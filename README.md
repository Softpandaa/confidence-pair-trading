# Confidence-Weighted Pair Trading via DCC-eGARCH

Cointegration-based pair trading on a pool of 45 US equities and ETFs, backtested in BackTrader over the 593 trading days from 23 August 2021 to 29 December 2023. Pairs are selected by the Engle-Granger test, the hedge ratio is estimated by a Kalman filter so it moves with the data, the spread is standardized by a DCC-eGARCH conditional volatility, and the four entry and exit z-score thresholds are optimized pair by pair. Three designs are compared to separate the contribution of threshold optimization from that of portfolio weighting.

## Findings

Optimizing the thresholds does not pay for itself. Against equal weight with fixed thresholds, equal weight with optimized thresholds earns 179.99% rather than 158.68%, but annualized volatility rises from 33.7% to 44.2% and maximum drawdown from 28.58% to 32.75%, so the Sharpe ratio falls from 1.32 to 1.17. The thresholds are fitted on a training sample ending more than two years before the window closes, and the extra return is bought with more risk than it is worth.

A confidence weighting built from the rolling cointegration statistic and the forecast volatility reverses this on an unchanged signal set. It earns 202.22%, pulls volatility back to 36.2% and drawdown to 25.59%, and records the highest Sharpe ratio at 1.43 on 393 trades against 386. Equal weight with fixed thresholds leads only on the system quality number, 3.68 against 3.28, and only because that statistic scales with the square root of trade count: per-trade expectancy is 0.158, 0.139 and 0.166 respectively. The contribution is the weighting rather than the threshold search.

Two results bound how far this should be read. Returns are net of 0.1% commission a side, which costs 27 to 30 percentage points of total return at a turnover of roughly 80 times capital a year, so the strategy would not survive a materially higher cost or any slippage. And the screen retains 73 pairs from 990 candidates at the 5% level, against the 50 that repeated testing alone would be expected to produce.

## Layout

```
data/     the screening price panel and the backtest feeds
src/      Python modules, every parameter declared once in config.py
R/        the DCC-eGARCH fit
output/   the pipeline's CSVs, committed so any stage can be run alone
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

The pipeline runs in five stages: `src/pairs.py` screens for cointegration and estimates the Kalman hedge ratio, spread and half life; `R/dcc_egarch.R` fits the volatility; `src/signals.py` optimizes the thresholds; `src/weights.py` computes the confidence scores and capped weights; `src/backtest.py` runs the three strategies.

## Data

`data/prices.csv` is the panel the cointegration screen runs on, daily closes from Yahoo Finance over 3 January 2012 to 29 December 2023, one column per ticker. Columns with any missing observation are dropped before screening. `data/quotes/` holds one OHLCV file per traded ticker over the testing window, 604 rows each, and these are the only data the backtest reads.

The study screened 45 tickers. `prices.csv` carries 47 columns of which 43 are complete, because EA, USLV, TWTR and ATVI can no longer be downloaded and META, EMLP, FSTA and UWT have incomplete histories. A rerun of the screen therefore cannot reproduce the original 45, which is why the upstream stages write to separate files.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/backtest.py
```

This reads the committed signals and weights in `output/`, reproduces the metrics quoted above and writes `output/backtest_metrics.csv`.

The upstream stages can be rerun in order, writing to `_rebuilt` names so the committed files are never overwritten. `pairs.py` recovers 47 of the 73 committed pairs for the reason above, `weights.py` skips the 13 pairs that use USLV, and `signals.py` matches the committed thresholds to a median absolute difference of about 2e-4, the residue being Nelder-Mead landing on a different local optimum of a step objective. `R/dcc_egarch.R` needs `rugarch` and `rmgarch`.
