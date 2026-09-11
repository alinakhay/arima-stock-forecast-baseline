"""Chronological, leakage-free evaluation for the repository's ARIMA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


def evaluate(csv_path: Path, holdout: int = 100) -> dict[str, object]:
    """Fit on the past and compare the ARIMA forecast with a last-value baseline."""
    frame = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    close = frame["Close"].astype(float).reset_index(drop=True)

    if holdout <= 0 or holdout >= len(close):
        raise ValueError("holdout must be positive and smaller than the dataset")

    train, test = close.iloc[:-holdout], close.iloc[-holdout:]
    forecast = ARIMA(train, order=(5, 2, 0)).fit().forecast(steps=holdout)
    forecast.index = test.index
    naive = pd.Series(train.iloc[-1], index=test.index)

    def metrics(prediction: pd.Series) -> dict[str, float]:
        return {
            "rmse": float(mean_squared_error(test, prediction) ** 0.5),
            "mae": float(mean_absolute_error(test, prediction)),
            "r2": float(r2_score(test, prediction)),
        }

    return {
        "observations": len(close),
        "train_observations": len(train),
        "test_observations": len(test),
        "train_end": frame["Date"].iloc[-holdout - 1].date().isoformat(),
        "test_start": frame["Date"].iloc[-holdout].date().isoformat(),
        "test_end": frame["Date"].iloc[-1].date().isoformat(),
        "arima_5_2_0": metrics(forecast),
        "last_value_baseline": metrics(naive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Google_stock_price.csv"))
    parser.add_argument("--holdout", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.holdout), indent=2))


if __name__ == "__main__":
    main()

