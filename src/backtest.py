"""The three BackTrader strategies of Section 5.

EW-FE  equal weight, fixed entry thresholds
EW-OE  equal weight, optimised entry thresholds
OW-OE  optimised weight, optimised entry thresholds

They differ only in which signal set they read and how each pair is sized, so
one strategy class covers all three; the original repeated it three times.

    python src/backtest.py
"""

import datetime as dt

import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import numpy as np
import pandas as pd

from config import (COMMISSION, FIXED_ENTRY, FIXED_EXIT, OUT_DIR, PAIR_FILE,
                    QUOTE_DIR, STARTING_CASH)

pairs = pd.read_csv(PAIR_FILE, index_col=0)
PAIRS = pairs[["Ticker 1", "Ticker 2"]].to_numpy().tolist()
TICKERS = sorted({t for p in PAIRS for t in p})


def _read(name):
    return pd.read_csv(OUT_DIR / name, header=0, index_col=0)


def crossings(z_score, entry, exit_, is_long):
    """The four threshold crossings of Section 3, as the report defines them."""
    if is_long:
        return pd.DataFrame({
            "entry": (z_score < entry) & (z_score.shift(1) > entry),
            "exit": (z_score > exit_) & (z_score.shift(1) < exit_)})
    return pd.DataFrame({
        "entry": (z_score > entry) & (z_score.shift(1) < entry),
        "exit": (z_score < exit_) & (z_score.shift(1) > exit_)})


def signal_set(z_score, thresholds):
    """One signal frame per pair. thresholds is (LET, LETX, SET, SETX) per pair."""
    out = []
    for i in range(len(PAIRS)):
        z = z_score.iloc[:, i]
        let, letx, set_, setx = thresholds[i]
        long_ = crossings(z, let, letx, True)
        short = crossings(z, set_, setx, False)
        out.append(pd.DataFrame({
            "long entry": long_["entry"], "long exit": long_["exit"],
            "short entry": short["entry"], "short exit": short["exit"]}).dropna())
    return out


class HLOCV(btfeeds.GenericCSVData):
    params = (("nullvalue", 0.0), ("dtformat", "%Y-%m-%d"),
              ("datetime", 0), ("open", 1), ("high", 2), ("low", 3),
              ("close", 4), ("volume", 5), ("openinterest", -1))


class PairStrategy(bt.Strategy):
    """Signals are read off a precomputed frame, one row per trading day.

    Sizing gives each leg the same notional, the portfolio value times the
    pair's weight, so the two legs are dollar neutral.
    """

    params = (("signals", None), ("weights", None), ("dates", None))

    def __init__(self):
        self.trading_size = [[1, 1] for _ in PAIRS]

    def next(self):
        today = self.datas[0].datetime.date(0)
        if today not in self.p.dates:
            return
        row = self.p.dates.index(today)
        for i, (t0, t1) in enumerate(PAIRS):
            d0, d1 = self.getdatabyname(t0), self.getdatabyname(t1)
            weight = 1 / len(PAIRS)
            if self.p.weights is not None:
                w = self.p.weights.iloc[row, i]
                weight = weight if np.isnan(w) else w
            ratio = d0.close[0] / d1.close[0]
            size = self.broker.get_value() * weight / d0.close[0]
            signal = self.p.signals[i].iloc[row]

            if signal["long entry"]:
                self.sell(data=d0, size=size); self.buy(data=d1, size=size * ratio)
                self.trading_size[i] = [size, size * ratio]
            elif signal["long exit"]:
                self.buy(data=d0, size=self.trading_size[i][0])
                self.sell(data=d1, size=self.trading_size[i][1])
            elif signal["short entry"]:
                self.buy(data=d0, size=size); self.sell(data=d1, size=size * ratio)
                self.trading_size[i] = [size, size * ratio]
            elif signal["short exit"]:
                self.sell(data=d0, size=self.trading_size[i][0])
                self.buy(data=d1, size=self.trading_size[i][1])


def run(name, signals, dates, weights=None):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(STARTING_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    for ticker in TICKERS:
        data = HLOCV(dataname=str(QUOTE_DIR / f"{ticker}.csv"))
        data.plotinfo.plot = False
        cerebro.adddata(data, name=ticker)
    cerebro.addstrategy(PairStrategy, signals=signals, weights=weights, dates=dates)
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name="annual")
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.SQN, _name="sqn")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    strat = cerebro.run()[0]
    trades = strat.analyzers.trades.get_analysis()
    return {
        "strategy": name,
        "final value": cerebro.broker.getvalue(),
        "annual return": {k: round(100 * v, 2) for k, v in
                          strat.analyzers.annual.get_analysis().items()},
        "sharpe": strat.analyzers.sharpe.get_analysis().get("sharperatio"),
        "max drawdown": strat.analyzers.drawdown.get_analysis()["max"]["drawdown"],
        "sqn": strat.analyzers.sqn.get_analysis()["sqn"],
        "trades": trades.get("total", {}).get("closed", 0),
    }


def main():
    z_score = _read("test_z_score.csv")
    optim = _read("z_score_optim.csv")
    weights = _read("weight_p_df.csv")

    fixed = [(-FIXED_ENTRY, -FIXED_EXIT, FIXED_ENTRY, FIXED_EXIT)] * len(PAIRS)
    optimised = optim.iloc[:, 2:6].to_numpy().tolist()   # LET, LETX, SET, SETX

    dates = [dt.datetime.strptime(d, "%Y-%m-%d").date()
             for d in signal_set(z_score, fixed)[0].index]

    rows = [run("EW-FE", signal_set(z_score, fixed), dates),
            run("EW-OE", signal_set(z_score, optimised), dates),
            run("OW-OE", signal_set(z_score, optimised), dates, weights)]
    table = pd.DataFrame(rows).set_index("strategy")
    pd.set_option("display.width", 200)
    print(table.to_string())
    table.to_csv(OUT_DIR / "backtest_metrics.csv")


if __name__ == "__main__":
    main()
