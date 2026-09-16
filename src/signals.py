"""Z-score threshold optimisation of Section 3.2.

For each pair, the four thresholds (LET, LETX, SET, SETX) are chosen on the
training spread by Nelder-Mead, maximising either terminal value or the Sharpe
ratio of a long-short position held between crossings.

    python src/signals.py            maximise profit   -> output/z_score_optim_rebuilt.csv
    python src/signals.py sharpe     maximise Sharpe   -> output/z_score_optim_sharpe_rebuilt.csv

The original ran this as five near-identical notebooks. The position state,
the P&L and the two objectives are each written once here.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import BOUNDS, OUT_DIR, START_GUESS

warnings.filterwarnings("ignore")

COLUMNS = ["objective", "long_entryZscore", "long_exitZscore",
           "short_entryZscore", "short_exitZscore"]


def position(z_score, entry, exit_, is_long):
    """Units held, as a state machine: enter on a crossing, hold until the exit.

    The original wrote this block twice per notebook, once for each side.
    """
    unit = 1 if is_long else -1
    if is_long:
        enter = (z_score < entry) & (z_score.shift(1) > entry)
        leave = (z_score > exit_) & (z_score.shift(1) < exit_)
    else:
        enter = (z_score > entry) & (z_score.shift(1) < entry)
        leave = (z_score < exit_) & (z_score.shift(1) > exit_)
    units = pd.Series(np.nan, index=z_score.index)
    units[enter] = unit
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
    return 0.0 if sd == 0 else -(np.nanmean(r) / sd * np.sqrt(252))


def optimise(objective, spread, base, z_score):
    result = minimize(objective, START_GUESS, args=(spread, base, z_score),
                      bounds=BOUNDS, method="Nelder-Mead")
    return [-result.fun, *result.x]


def main(which="profit"):
    objective = sharpe if which == "sharpe" else terminal_value
    z_score = pd.read_csv(OUT_DIR / "train_z_score.csv", index_col=0)
    spread = pd.read_csv(OUT_DIR / "train_spread.csv", index_col=0)
    base = pd.read_csv(OUT_DIR / "train_spread_base.csv", index_col=0)
    # The z score starts 18 rows later than the spread: the rolling mean over
    # each pair's half life consumes the first observations.
    offset = len(spread) - len(z_score)
    spread, base = spread.iloc[offset:], base.iloc[offset:]

    rows = []
    for i, name in enumerate(z_score.columns):
        rows.append(optimise(objective, spread.iloc[:, i], base.iloc[:, i],
                             z_score.iloc[:, i]))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{z_score.shape[1]} pairs", flush=True)
    table = pd.DataFrame(rows, index=z_score.columns, columns=COLUMNS)
    name = "z_score_optim_sharpe_rebuilt.csv" if which == "sharpe" else "z_score_optim_rebuilt.csv"
    table.to_csv(OUT_DIR / name)
    print(table.describe().loc[["mean", "min", "max"]].to_string())
    print("wrote", OUT_DIR / name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "profit")
