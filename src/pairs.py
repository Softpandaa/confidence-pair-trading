"""Pair formation of Section 2: cointegration screen, Kalman hedge ratio, half life.

    python src/pairs.py

Reads data/prices.csv and writes the rebuilt pair list and spreads under
output/ with a _rebuilt suffix, so the committed files are never overwritten.

The committed pair_df.csv was produced from a 48-ticker panel. EA, USLV, TWTR
and ATVI are no longer obtainable, so this rerun works from 47 tickers and will
not reproduce the same 73 pairs; the overlap is printed.
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

from config import (COINT_LEVEL, KALMAN_DELTA, KALMAN_OBS_COV, OUT_DIR,
                    PAIR_FILE, PRICE_FILE, SMOOTHER_OBS_COV,
                    SMOOTHER_TRANS_COV, TRAIN_FRACTION)


def smooth(series):
    """Kalman rolling mean, used to damp price noise before the regression."""
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=0, initial_state_covariance=1,
                      observation_covariance=SMOOTHER_OBS_COV,
                      transition_covariance=SMOOTHER_TRANS_COV)
    means, _ = kf.filter(series.values)
    return pd.Series(means.flatten(), index=series.index)


def hedge_ratio(x, y):
    """Time varying beta of y on x, from a two state Kalman regression."""
    cov = KALMAN_DELTA / (1 - KALMAN_DELTA) * np.eye(2)
    observation = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=observation,
                      observation_covariance=KALMAN_OBS_COV,
                      transition_covariance=cov)
    means, _ = kf.filter(y.values)
    return means[:, 0]


def spread_of(frame, t0, t1):
    x, y = frame[t0], frame[t1]
    return y - x * hedge_ratio(smooth(x), smooth(y))


def half_life(spread):
    """Days for a deviation to decay by half, from the AR(1) coefficient."""
    lag = spread.shift(1); lag.iloc[0] = lag.iloc[1]
    delta = spread - lag; delta.iloc[0] = delta.iloc[1]
    beta = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
    return max(1, int(round(-np.log(2) / beta, 0)))


def main():
    price = pd.read_csv(PRICE_FILE, index_col="date", parse_dates=True).dropna(axis=1)
    split = int(len(price) * TRAIN_FRACTION)
    train, test = price.iloc[:split], price.iloc[split:]
    tickers = list(price.columns)
    print(f"{len(tickers)} tickers, training to {train.index[-1]:%Y-%m-%d}")

    found = []
    for i, t0 in enumerate(tickers):
        for j, t1 in enumerate(tickers):
            if i <= j:
                continue
            if coint(train[t0], train[t1])[1] < COINT_LEVEL:
                found.append([t0, t1])
    print(f"cointegrated pairs at {COINT_LEVEL:.0%}: {len(found)}")

    names = [f"Pair {i + 1}" for i in range(len(found))]
    pd.DataFrame(found, index=names, columns=["Ticker 1", "Ticker 2"]).to_csv(
        OUT_DIR / "pair_df_rebuilt.csv")
    for label, frame in (("train", train), ("test", test)):
        spreads = pd.DataFrame({n: spread_of(frame, *p) for n, p in zip(names, found)})
        spreads.to_csv(OUT_DIR / f"{label}_spread_rebuilt.csv")
        pd.DataFrame({"Half Life": [half_life(spreads[n]) for n in names]},
                     index=names).to_csv(OUT_DIR / f"{label}_half_life_rebuilt.csv")

    committed = pd.read_csv(PAIR_FILE, index_col=0)
    old = {frozenset(r) for r in committed[["Ticker 1", "Ticker 2"]].to_numpy()}
    new = {frozenset(p) for p in found}
    print(f"overlap with the committed 73 pairs: {len(old & new)}")


if __name__ == "__main__":
    main()
