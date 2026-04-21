"""
backtest.py
===========
Walk-forward cross-validation untuk memilih model terbaik per komoditas.

Strategi:
1. Split data: train = semua kecuali N hari terakhir, test = N hari terakhir
2. Fit model pada train, predict N hari
3. Hitung MAPE, RMSE, MAE
4. Ulangi untuk beberapa cutoff (opsional sliding window)

Output:
- DataFrame metrik per model
- Info model terbaik (lowest MAPE)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import (
    ProphetForecaster, LightGBMForecaster, XGBoostForecaster,
    EnsembleForecaster, Forecaster,
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def backtest_single(
    model_cls: type[Forecaster],
    ts: pd.DataFrame,
    horizon: int = 30,
) -> dict:
    """Backtest 1 model pada 1 time series dengan single holdout."""
    if len(ts) < horizon + 60:
        return {"model": model_cls.__name__, "error": "Data tidak cukup"}

    train = ts.iloc[:-horizon].copy()
    test = ts.iloc[-horizon:].copy()

    try:
        model = model_cls()
        model.fit(train)
        pred = model.predict(horizon)
        # Join pada ds biar aman
        pred = pred.rename(columns={"ds": "ds"})
        merged = test.merge(pred, on="ds", how="inner")

        y_true = merged["y"].values
        y_pred = merged["yhat"].values
        return {
            "model": model.name,
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "coverage": float(
                np.mean(
                    (merged["yhat_lower"] <= merged["y"])
                    & (merged["y"] <= merged["yhat_upper"])
                ) * 100
            ),
        }
    except Exception as e:
        return {"model": model_cls.__name__, "error": str(e)[:100]}


def backtest_all(
    ts: pd.DataFrame,
    horizon: int = 30,
    models: list[type[Forecaster]] | None = None,
) -> pd.DataFrame:
    """Backtest semua model, return DataFrame ranked by MAPE."""
    if models is None:
        models = [
            ProphetForecaster, LightGBMForecaster,
            XGBoostForecaster, EnsembleForecaster,
        ]
    results = [backtest_single(m, ts, horizon) for m in models]
    df = pd.DataFrame(results)
    if "mape" in df.columns:
        df = df.sort_values("mape", na_position="last").reset_index(drop=True)
    return df


def best_model_name(backtest_df: pd.DataFrame) -> str:
    """Ambil nama model terbaik dari hasil backtest."""
    valid = backtest_df.dropna(subset=["mape"]) if "mape" in backtest_df.columns else pd.DataFrame()
    if valid.empty:
        return "Prophet"  # fallback
    return valid.iloc[0]["model"]


if __name__ == "__main__":
    from data_loader import load_cache, get_timeseries
    df = load_cache()
    ts = get_timeseries(df, "Cabai Rawit Merah")
    print(f"Backtest pada Cabai Rawit Merah ({len(ts)} hari)...")
    result = backtest_all(ts, horizon=30)
    print(result.to_string(index=False))
    print(f"\nModel terbaik: {best_model_name(result)}")
