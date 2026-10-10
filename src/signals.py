"""Trading signal of Section 4, written as *_rebuilt.csv: python src/signals.py [sharpe]."""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import BOUNDS, OUT_DIR, START_GUESS, TRADING_DAYS

COLUMNS = ["objective", "long_entryZscore", "long_exitZscore",
           "short_entryZscore", "short_exitZscore"]


def crossings(z, entry, exit_, is_long):
    """Entry and exit crossings of Section 4.1."""
    prev = z.shift(1)
    if is_long:
        return (z < entry) & (prev > entry), (z > exit_) & (prev < exit_)
    return (z > entry) & (prev < entry), (z < exit_) & (prev > exit_)


def position(z_score, entry, exit_, is_long):
    """Units held, from an entry crossing until the exit crossing."""
    enter, leave = crossings(z_score, entry, exit_, is_long)
    units = pd.Series(np.nan, index=z_score.index)
    units[enter] = 1 if is_long else -1
    units[leave] = 0
    units.iloc[0] = 0
    return units.ffill()


def returns(spread, base, z_score, params):
    """Daily strategy return from the four thresholds."""
    let, letx, set_, setx = params
    units = position(z_score, let, letx, True) + position(z_score, set_, setx, False)
    change = (spread - spread.shift(1)) / base
    return change.to_numpy() * units.shift(1).to_numpy()


def terminal_value(params, spread, base, z_score):
    return -(1 + np.nansum(returns(spread, base, z_score, params)))


def sharpe(params, spread, base, z_score):
    r = returns(spread, base, z_score, params)
    sd = np.nanstd(r)
    return 0.0 if sd == 0 else -(np.nanmean(r) / sd * np.sqrt(TRADING_DAYS))


def optimise(objective, spread, base, z_score):
    result = minimize(objective, START_GUESS, args=(spread, base, z_score),
                      bounds=BOUNDS, method="Nelder-Mead")
    return [-result.fun, *result.x]


def main(which="profit"):
    objective = sharpe if which == "sharpe" else terminal_value
    z_score = pd.read_csv(OUT_DIR / "train_z_score.csv", index_col=0)
    spread = pd.read_csv(OUT_DIR / "train_spread.csv", index_col=0)
    base = pd.read_csv(OUT_DIR / "train_spread_base.csv", index_col=0)
    # The z score omits the first rows, consumed by the half-life rolling mean.
    offset = len(spread) - len(z_score)
    spread, base = spread.iloc[offset:], base.iloc[offset:]

    rows = []
    for i in range(z_score.shape[1]):
        rows.append(optimise(objective, spread.iloc[:, i], base.iloc[:, i], z_score.iloc[:, i]))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{z_score.shape[1]} pairs", flush=True)
    table = pd.DataFrame(rows, index=z_score.columns, columns=COLUMNS)
    name = "z_score_optim_sharpe_rebuilt.csv" if which == "sharpe" else "z_score_optim_rebuilt.csv"
    table.to_csv(OUT_DIR / name)
    print(table.describe().loc[["mean", "min", "max"]].to_string())
    print("wrote", OUT_DIR / name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "profit")
