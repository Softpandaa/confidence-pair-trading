"""Every path and parameter used by this project, declared once."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
QUOTE_DIR = DATA_DIR / "quotes"      # per ticker OHLCV, the backtest feed
OUT_DIR = ROOT / "output"            # committed pipeline inputs and outputs
FIG_DIR = ROOT / "figures"           # regenerated, not tracked

PRICE_FILE = DATA_DIR / "prices.csv"   # daily closes, one column per ticker
PAIR_FILE = OUT_DIR / "pair_df.csv"

# Sample. The split is 80 percent of the trading days; the training window ends
# 2021-08-05 and the testing window runs to the end of 2023.
START, END = "2012-01-01", "2023-12-31"
TRAIN_FRACTION = 0.8

# Pair formation.
COINT_LEVEL = 0.05
KALMAN_DELTA = 1e-3                  # how much the hedge ratio is allowed to wander
KALMAN_OBS_COV = 2.0
SMOOTHER_OBS_COV = 1.0               # KalmanFilterAverage, price smoothing
SMOOTHER_TRANS_COV = 0.01

# Signal thresholds. The fixed pair is used by EW-FE; the optimiser searches
# within BOUNDS from START_GUESS for the other two strategies.
FIXED_ENTRY, FIXED_EXIT = 0.8, -0.05
START_GUESS = [-0.7, 0.05, 0.7, -0.05]
BOUNDS = [(-1.1, 1.1)] * 4

# Portfolio weighting.
LOOK_BACK = 250                      # rolling cointegration window, trading days
WEIGHT_CAP_MULTIPLE = 4              # cap at 4 / n, about 5 percent at n = 73

# Backtest.
STARTING_CASH = 10 ** 9
COMMISSION = 0.001                   # 0.1 percent a side, no slippage
