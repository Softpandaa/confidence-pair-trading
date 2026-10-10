"""Confidence weights of Section 5, written as *_rebuilt.csv."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from config import LOOK_BACK, OUT_DIR, PAIR_FILE, PRICE_FILE, WEIGHT_CAP_MULTIPLE


def rolling_cointegration(price, pairs, dates):
    """p-value and |t-statistic| of each pair over a trailing window."""
    p = pd.DataFrame(index=dates, columns=range(len(pairs)), dtype=float)
    t = p.copy()
    position = {d: i for i, d in enumerate(price.index)}
    for row, date in enumerate(dates):
        end = position[date] + 1
        window = slice(end - LOOK_BACK, end)
        for j, (t0, t1) in enumerate(pairs):
            if t0 not in price or t1 not in price:
                continue
            stat, pval, _ = coint(price[t0].iloc[window], price[t1].iloc[window])
            p.iloc[row, j], t.iloc[row, j] = pval, abs(stat)
    return p, t


def allocate(confidence, n):
    """Section 5.2 weights: shares capped pass by pass, the rest at the smallest weight."""
    cap = WEIGHT_CAP_MULTIPLE / n
    out = np.full(confidence.shape, np.nan)
    conf = confidence.to_numpy(dtype=float)
    for row in range(len(out)):
        weight = out[row]
        while True:
            remaining = 1 - np.nansum(weight)
            free = np.isnan(weight)
            total = np.nansum(conf[row][free])
            if remaining <= 0 or total <= 0:
                break
            limit = remaining * cap
            with np.errstate(invalid="ignore"):
                hit = free & (conf[row] / total > limit)
            if not hit.any():
                break
            weight[hit] = limit
        fill = 1 / n if np.isnan(weight).all() else np.nanmin(weight)
        weight[np.isnan(weight)] = fill
        out[row] = weight / weight.sum()
    return pd.DataFrame(out, index=confidence.index)


def main():
    pairs = pd.read_csv(PAIR_FILE, index_col=0)[["Ticker 1", "Ticker 2"]].to_numpy().tolist()
    vol = pd.read_csv(OUT_DIR / "test_spread_vol.csv", index_col=0)
    price = pd.read_csv(PRICE_FILE, index_col="date", parse_dates=True)

    dates = pd.to_datetime(vol.index)
    usable = dates.isin(price.index)
    skipped = [f"{a}-{b}" for a, b in pairs if a not in price or b not in price]
    print(f"{len(pairs)} pairs, {usable.sum()} of {len(dates)} dates in the panel")
    if skipped:
        more = f" and {len(skipped) - 4} more" if len(skipped) > 4 else ""
        print(f"skipped, ticker no longer obtainable: {', '.join(skipped[:4])}{more}")

    p, t = rolling_cointegration(price, pairs, dates[usable])
    p.to_csv(OUT_DIR / "rolling_coint_pvalue_rebuilt.csv")
    t.to_csv(OUT_DIR / "rolling_coint_tstat_rebuilt.csv")

    v = vol.loc[usable].to_numpy()
    n = len(pairs)
    for label, confidence in (("p", pd.DataFrame(v / p.to_numpy(), index=p.index)),
                              ("t", pd.DataFrame(v * t.to_numpy(), index=t.index))):
        weight = allocate(confidence, n)
        weight.columns = vol.columns
        weight.to_csv(OUT_DIR / f"weight_{label}_df_rebuilt.csv")
        print(f"  weight_{label}: row sums {weight.sum(axis=1).min():.4f}"
              f"-{weight.sum(axis=1).max():.4f}, max weight {weight.to_numpy().max():.4f}"
              f" (cap {WEIGHT_CAP_MULTIPLE / n:.4f})")


if __name__ == "__main__":
    main()
