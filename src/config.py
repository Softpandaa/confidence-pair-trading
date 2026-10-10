"""Paths and parameters of the Python steps, declared once."""

from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
QUOTE_DIR = DATA_DIR / "quotes"         # per ticker OHLCV, the backtest feed
OUT_DIR = ROOT / "output"
PRICE_FILE = DATA_DIR / "prices.csv"    # daily closes, one column per ticker
PAIR_FILE = OUT_DIR / "pair_df.csv"

# Sample ----------------------------------------------------------------------
TRAIN_FRACTION = 0.8
TRADING_DAYS = 252                      # per year

# Pair formation --------------------------------------------------------------
COINT_LEVEL = 0.05
KALMAN_DELTA = 1e-3                     # state noise scale
KALMAN_OBS_COV = 2.0
SMOOTHER_OBS_COV = 1.0
SMOOTHER_TRANS_COV = 0.01

# Signal thresholds -----------------------------------------------------------
FIXED_ENTRY, FIXED_EXIT = 0.8, -0.05    # EW-FE SET, SETX; long side negated
START_GUESS = [-0.7, 0.05, 0.7, -0.05]  # LET, LETX, SET, SETX
BOUNDS = [(-1.1, 1.1)] * 4

# Weights ---------------------------------------------------------------------
LOOK_BACK = 250                         # cointegration window, trading days
WEIGHT_CAP_MULTIPLE = 4                 # times the equal weight 1 / n

# Backtest --------------------------------------------------------------------
STARTING_CASH = 10 ** 9
COMMISSION = 0.001                      # of traded notional, each side
