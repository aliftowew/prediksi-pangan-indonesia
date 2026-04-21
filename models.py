"""
models.py
=========
Model forecasting harga pangan dengan interface seragam.

Semua forecaster menerima DataFrame (ds, y) dan output
DataFrame prediksi (ds, yhat, yhat_lower, yhat_upper).

Model tersedia:
- ProphetForecaster   : Baseline, baik untuk seasonality + Lebaran
- LightGBMForecaster  : Gradient boosting + lag features, juara M5 competitions
- XGBoostForecaster   : Alternatif LightGBM
- EnsembleForecaster  : Rata-rata prediksi dari beberapa model

Uncertainty interval:
- Prophet: built-in Bayesian
- LightGBM/XGBoost: quantile regression (train model terpisah untuk 0.1 & 0.9)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==========================================================
# Lebaran holidays (untuk Prophet)
# ==========================================================
LEBARAN_DATES = pd.to_datetime([
    "2022-05-02",  # Lebaran 1443 H
    "2023-04-22",  # Lebaran 1444 H
    "2024-04-10",  # Lebaran 1445 H
    "2025-03-31",  # Lebaran 1446 H
    "2026-03-20",  # Lebaran 1447 H
    "2027-03-09",  # Lebaran 1448 H (proyeksi)
])


def build_lebaran_calendar() -> pd.DataFrame:
    return pd.DataFrame({
        "holiday": "Lebaran",
        "ds": LEBARAN_DATES,
        "lower_window": -30,   # 30 hari sebelum: biasanya harga naik
        "upper_window": 7,     # 7 hari setelah: normalisasi
    })


# ==========================================================
# Base interface
# ==========================================================
class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "Forecaster":
        """df harus punya kolom 'ds' (datetime) dan 'y' (float)."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Return DataFrame(ds, yhat, yhat_lower, yhat_upper)."""


# ==========================================================
# Feature engineering untuk model tree-based
# ==========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["ds"].dt.dayofweek
    out["day"] = out["ds"].dt.day
    out["month"] = out["ds"].dt.month
    out["dayofyear"] = out["ds"].dt.dayofyear
    out["weekofyear"] = out["ds"].dt.isocalendar().week.astype(int)
    out["year"] = out["ds"].dt.year
    out["quarter"] = out["ds"].dt.quarter
    # Hari ke Lebaran terdekat (signal musiman Ramadhan)
    days_to_lebaran = np.array([
        (out["ds"] - d).dt.days.values for d in LEBARAN_DATES
    ])
    # Ambil jarak minimum absolut (bisa +/-)
    min_abs_idx = np.argmin(np.abs(days_to_lebaran), axis=0)
    out["days_to_lebaran"] = days_to_lebaran[
        min_abs_idx, np.arange(len(out))
    ]
    return out


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 7, 14, 30],
    rolling_windows: list[int] = [7, 30],
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    for w in rolling_windows:
        # Shift 1 dulu supaya feature tidak leak value hari ini
        out[f"rollmean_{w}"] = out["y"].shift(1).rolling(w).mean()
        out[f"rollstd_{w}"] = out["y"].shift(1).rolling(w).std()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(add_time_features(df))


def _recursive_forecast_fast(
    model_median, model_lower, model_upper,
    train_df: pd.DataFrame, horizon: int,
) -> pd.DataFrame:
    """
    Optimisasi recursive forecast: time features dihitung sekali saja,
    lag features diupdate incremental menggunakan numpy array.
    """
    last_date = train_df["ds"].max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    full = pd.concat([
        train_df[["ds", "y"]],
        pd.DataFrame({"ds": future_dates, "y": [np.nan] * horizon})
    ], ignore_index=True)
    full = add_time_features(full)

    y_array = full["y"].values.copy()
    preds = []
    lags = [1, 2, 3, 7, 14, 30]
    n_hist = len(train_df)

    for step in range(n_hist, n_hist + horizon):
        next_date = full["ds"].iloc[step]
        row_features = {}
        for lag in lags:
            row_features[f"lag_{lag}"] = (
                y_array[step - lag] if step - lag >= 0 else np.nan
            )
        for w in [7, 30]:
            window = y_array[max(0, step - w):step]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                row_features[f"rollmean_{w}"] = float(np.mean(window))
                row_features[f"rollstd_{w}"] = (
                    float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
                )
            else:
                row_features[f"rollmean_{w}"] = np.nan
                row_features[f"rollstd_{w}"] = np.nan
        for c in ["dayofweek", "day", "month", "dayofyear", "weekofyear",
                  "year", "quarter", "days_to_lebaran"]:
            row_features[c] = full[c].iloc[step]

        X_new = pd.DataFrame([row_features])[FEATURE_COLS]

        if X_new.isna().any(axis=1).iloc[0]:
            last_valid = y_array[step - 1] if step > 0 else train_df["y"].iloc[-1]
            yhat = float(last_valid)
            ylow, yup = yhat * 0.95, yhat * 1.05
        else:
            yhat = float(model_median.predict(X_new)[0])
            ylow = float(model_lower.predict(X_new)[0])
            yup = float(model_upper.predict(X_new)[0])
            ylow, yup = min(ylow, yup), max(ylow, yup)

        preds.append({
            "ds": next_date,
            "yhat": max(yhat, 0),
            "yhat_lower": max(ylow, 0),
            "yhat_upper": max(yup, 0),
        })
        y_array[step] = yhat

    return pd.DataFrame(preds)


FEATURE_COLS = [
    "dayofweek", "day", "month", "dayofyear", "weekofyear",
    "year", "quarter", "days_to_lebaran",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "rollmean_7", "rollstd_7", "rollmean_30", "rollstd_30",
]


# ==========================================================
# ProphetForecaster
# ==========================================================
class ProphetForecaster(Forecaster):
    name = "Prophet"

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        holidays_prior_scale: float = 0.5,
        changepoint_prior_scale: float = 0.05,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self._train = None

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet
        self._train = df.copy()
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
            holidays=build_lebaran_calendar(),
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=0.8,
        )
        self.model.add_country_holidays(country_name="ID")
        self.model.fit(df[["ds", "y"]])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon)
        fcst = self.model.predict(future)
        out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
        # Pastikan harga tidak negatif
        for c in ["yhat", "yhat_lower", "yhat_upper"]:
            out[c] = out[c].clip(lower=0)
        return out.reset_index(drop=True)


# ==========================================================
# LightGBMForecaster (recursive multi-step)
# ==========================================================
class LightGBMForecaster(Forecaster):
    name = "LightGBM"

    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self._train = None

    def _make_xy(self, df_feat: pd.DataFrame):
        df_feat = df_feat.dropna()
        X = df_feat[FEATURE_COLS]
        y = df_feat["y"]
        return X, y

    def fit(self, df: pd.DataFrame) -> "LightGBMForecaster":
        import lightgbm as lgb
        self._train = df.copy().reset_index(drop=True)
        df_feat = build_features(self._train)
        X, y = self._make_xy(df_feat)

        common = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31, min_child_samples=10,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            verbose=-1,
        )
        self.model_median = lgb.LGBMRegressor(objective="regression", **common).fit(X, y)
        self.model_lower = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **common).fit(X, y)
        self.model_upper = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **common).fit(X, y)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        return _recursive_forecast_fast(
            self.model_median, self.model_lower, self.model_upper,
            self._train, horizon,
        )


# ==========================================================
# XGBoostForecaster (struktur sama, engine beda)
# ==========================================================
class XGBoostForecaster(Forecaster):
    name = "XGBoost"

    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self._train = None

    def _make_xy(self, df_feat: pd.DataFrame):
        df_feat = df_feat.dropna()
        return df_feat[FEATURE_COLS], df_feat["y"]

    def fit(self, df: pd.DataFrame) -> "XGBoostForecaster":
        import xgboost as xgb
        self._train = df.copy().reset_index(drop=True)
        df_feat = build_features(self._train)
        X, y = self._make_xy(df_feat)

        common = dict(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=6, subsample=0.9, colsample_bytree=0.9, verbosity=0,
        )
        self.model_median = xgb.XGBRegressor(
            objective="reg:squarederror", **common,
        ).fit(X, y)
        self.model_lower = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.1, **common,
        ).fit(X, y)
        self.model_upper = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.9, **common,
        ).fit(X, y)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        return _recursive_forecast_fast(
            self.model_median, self.model_lower, self.model_upper,
            self._train, horizon,
        )


# ==========================================================
# EnsembleForecaster
# ==========================================================
class EnsembleForecaster(Forecaster):
    name = "Ensemble"

    def __init__(self, models: list[Forecaster] | None = None, weights: list[float] | None = None):
        self.models = models if models is not None else [
            ProphetForecaster(), LightGBMForecaster(), XGBoostForecaster()
        ]
        self.weights = weights if weights is not None else [1 / len(self.models)] * len(self.models)

    def fit(self, df: pd.DataFrame) -> "EnsembleForecaster":
        for m in self.models:
            m.fit(df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        preds = [m.predict(horizon) for m in self.models]
        base = preds[0][["ds"]].copy()

        yhat = np.zeros(horizon)
        ylow = np.zeros(horizon)
        yup = np.zeros(horizon)
        for w, p in zip(self.weights, preds):
            yhat += w * p["yhat"].values
            ylow += w * p["yhat_lower"].values
            yup += w * p["yhat_upper"].values

        base["yhat"] = yhat
        base["yhat_lower"] = ylow
        base["yhat_upper"] = yup
        return base


# ==========================================================
# Factory
# ==========================================================
def make_forecaster(name: str) -> Forecaster:
    name = name.lower().replace(" ", "")
    if name in ("prophet",):
        return ProphetForecaster()
    if name in ("lightgbm", "lgbm"):
        return LightGBMForecaster()
    if name in ("xgboost", "xgb"):
        return XGBoostForecaster()
    if name in ("ensemble",):
        return EnsembleForecaster()
    raise ValueError(f"Unknown forecaster: {name}")


AVAILABLE_MODELS = ["Prophet", "LightGBM", "XGBoost", "Ensemble"]
