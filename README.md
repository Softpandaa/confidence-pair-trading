# Confidence-Weighted Pair Trading via DCC-eGARCH

Cointegration-based pair trading on a pool of 45 US equities and ETFs, backtested in BackTrader from 23 August 2021 to 29 December 2023. Pairs are selected by the Engle-Granger test, the time-varying hedge ratio is estimated by a Kalman filter, the spread is standardized and modeled by a DCC-eGARCH conditional volatility. We optimize the entry and exit thresholds, and propose a confidence-weighting scheme based on the rolling cointegration statistic and forecast volatility. 

## Findings

Optimizing the thresholds does not pay for itself. Against equal weight with fixed thresholds, equal weight with optimized thresholds improves returns but increases volatility and drawdown by more, so the Sharpe ratio falls from 1.32 to 1.17. Nevertheless, combined with a confidence score based on the rolling cointegration statistic and forecast volatility, it delivers the highest total return and the lowest drawdown. The Sharpe ratio is also highest at 1.43.


## Layout

```
data/     the screening price panel and the backtest feeds
src/      Python modules, every parameter declared once in config.py
R/        the DCC-eGARCH fit
output/   the pipeline's CSVs, committed so any stage can be run alone
latex/    report source and figures
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/prices.csv` are the daily closing price from Yahoo Finance over 3 January 2012 to 29 December 2023. `data/quotes/` holds one OHLCV file per traded ticker over the testing window, and these are the only data the backtest reads.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/backtest.py
```

This reads the committed signals and weights in `output/`, reproduces the metrics quoted above and writes `output/backtest_metrics.csv`. `R/dcc_egarch.R` needs `rugarch` and `rmgarch`.
