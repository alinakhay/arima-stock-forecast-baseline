# ARIMA Stock-Forecasting Baseline

An honest time-series baseline for forecasting Alphabet's split-adjusted closing price with an ARIMA model.

The repository contains 1,957 observations from March 2014 through December 2021. The original notebook explores stationarity, selects an ARIMA order and visualises historical and forward predictions. A separate evaluation script now provides the more important portfolio signal: a chronological, leakage-free comparison against a naive last-value forecast.

## What this demonstrates

- Chronological train/test splitting for market data
- Augmented Dickey-Fuller stationarity diagnostics
- ARIMA specification and multi-step forecasting
- Comparison with a simple baseline rather than reporting model metrics in isolation
- Awareness of the limits of price-level forecasting

## Reproduce the evaluation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py
```

The script reserves the final 100 observations for testing, fits only on the earlier data and reports RMSE, MAE and R-squared for both ARIMA(5,2,0) and a last-observation baseline.

## Corrected out-of-sample result

| Model | RMSE | MAE | R-squared |
| --- | ---: | ---: | ---: |
| ARIMA(5,2,0) | 18.77 | 15.78 | -19.68 |
| Last observed close | **6.69** | **5.75** | **-1.62** |

The ARIMA specification underperforms the naive forecast decisively. That negative result is the main conclusion: fitting a sophisticated time-series model does not make raw equity-price levels predictably tradable, and leakage-free validation changes the interpretation of the original notebook.

## Repository guide

| File | Purpose |
| --- | --- |
| `evaluate.py` | Leakage-free chronological evaluation and baseline comparison |
| `Final_project.ipynb` | Original exploratory analysis, diagnostics and visualisations |
| `Google_stock_price.csv` | Historical OHLCV observations used by the analysis |
| `Google_stock_price-future_values.csv` | Small subsequent-period comparison sample |

## Important limitations

This is a statistical forecasting exercise, not a trading strategy or evidence of investment performance. It does not model corporate actions, transaction costs, execution, exogenous variables or uncertainty intervals. The original notebook fits its displayed evaluation model on the full series, so those saved notebook scores should be treated as in-sample diagnostics; `evaluate.py` is the corrected out-of-sample reference.
