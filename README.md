# Confidence-Weighted Pairs Trading via DCC-EGARCH

This study asks whether optimizing the entry and exit thresholds pair by pair and weighting the pairs by a confidence score improve a pairs trading portfolio of 45 US equities and ETFs, backtested from August 2021 to December 2023. Pairs come from the Engle–Granger cointegration test, hedge ratios from a Kalman filter and spread volatility from a DCC-EGARCH model.

## Findings

Net of transaction costs, optimizing the thresholds raises the total return from 158.68% to 179.99% but lowers the Sharpe ratio from 1.32 to 1.17, as volatility and drawdown rise by more. Adding confidence weights based on the rolling cointegration p-value and forecast volatility delivers the highest total return, the highest Sharpe ratio at 1.43 and the lowest drawdown.

## Layout

```
data/     the screening price panel and the backtest feeds
src/      Python modules, every parameter declared once in config.py
R/        the DCC-eGARCH fit
output/   the pipeline's CSVs
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/prices.csv` holds the daily closing prices from Yahoo Finance over 3 January 2012 to 29 December 2023. `data/quotes/` holds one OHLCV file per traded ticker over the testing window, and these are the only data the backtest reads.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/backtest.py
```

This reads the committed signals and weights in `output/`, reproduces Table 2 of the report except the Sharpe ratio and writes `output/backtest_metrics.csv`. `R/dcc_egarch.R` needs `rugarch` and `rmgarch`.