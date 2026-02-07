"""
Evaluation utilities for probabilistic NHL total-goals forecasts.

Focus:
- Time-series cross-validation (expanding window)
- Proper scoring rules (NLL, CRPS)
- Event-based evaluation (Brier/log loss for over/under)
- Calibration / reliability curves
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import config
from .conformal import split_conformal_interval
from .model import get_feature_columns
from .probabilistic import (
    crps_from_pmf,
    fit_nb2_alpha,
    fit_poisson_mixture,
    nb2_pmf_matrix,
    poisson_mixture_pmf_matrix,
    poisson_nll,
    poisson_pmf_matrix,
    prob_over_from_pmf,
    randomized_pit,
    reliability_curve,
)
from .team_strength import TeamStrengthPoissonModel

PointModel = Literal["xgb", "poisson_glm", "team_strength"]
DistModel = Literal["poisson", "nb2", "poisson_mixture"]


def _brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


@dataclass(frozen=True)
class FoldForecastMetrics:
    fold: int
    n_train: int
    n_cal: int
    n_test: int
    mae: float
    rmse: float
    poisson_nll: float
    dist_nll: float
    crps: float
    over_brier: float
    over_log_loss: float
    conformal_q: float


@dataclass(frozen=True)
class CVForecastResult:
    point_model: PointModel
    dist_model: DistModel
    threshold: float
    max_goals: int
    folds: list[FoldForecastMetrics]
    reliability_bins: list
    pit_values: list[float]

    @property
    def metrics_mean(self) -> dict[str, float]:
        keys = [
            "mae",
            "rmse",
            "poisson_nll",
            "dist_nll",
            "crps",
            "over_brier",
            "over_log_loss",
            "conformal_q",
        ]
        out: dict[str, float] = {}
        for k in keys:
            out[k] = float(np.mean([getattr(f, k) for f in self.folds]))
        return out


def _fit_point_model(
    df_fit: pd.DataFrame,
    *,
    point_model: PointModel,
    feature_cols: Optional[list[str]],
) -> tuple[object, Optional[StandardScaler], list[str]]:
    if point_model == "team_strength":
        model = TeamStrengthPoissonModel().fit(df_fit)
        return model, None, []

    if feature_cols is None:
        feature_cols = get_feature_columns(df_fit)
    if not feature_cols:
        raise ValueError("No engineered feature columns found; run add_features() first.")

    required_cols = feature_cols + ["totalGoals"]
    df_fit = df_fit.dropna(subset=required_cols).copy()
    if df_fit.empty:
        raise ValueError("No rows with complete feature/target data in this fold.")

    X = df_fit[feature_cols].to_numpy(dtype=float)
    y = df_fit["totalGoals"].to_numpy(dtype=float)

    if point_model == "poisson_glm":
        from sklearn.linear_model import PoissonRegressor

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = PoissonRegressor(alpha=1.0, max_iter=2000)
        model.fit(Xs, y)
        return model, scaler, feature_cols

    if point_model == "xgb":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("point_model='xgb' requires xgboost") from e

        params = dict(config.model.xgb_params)
        params.update({"random_state": config.model.random_state, "n_jobs": -1, "verbosity": 0})
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        return model, None, feature_cols

    raise ValueError(f"Unknown point_model: {point_model}")


def _predict_mu(
    model: object,
    df: pd.DataFrame,
    *,
    point_model: PointModel,
    scaler: Optional[StandardScaler],
    feature_cols: list[str],
) -> np.ndarray:
    if point_model == "team_strength":
        mu = model.predict_mu(df)  # type: ignore[attr-defined]
        return np.clip(mu, 1e-9, None)

    X = df[feature_cols].to_numpy(dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    mu = np.asarray(model.predict(X), dtype=float)
    return np.clip(mu, 1e-9, None)


def time_series_cv_forecast(
    df: pd.DataFrame,
    *,
    point_model: PointModel = "xgb",
    dist_model: DistModel = "nb2",
    threshold: float = 6.5,
    n_splits: int = 5,
    cal_fraction: float = 0.2,
    max_goals: int = 20,
    feature_cols: Optional[list[str]] = None,
    random_state: int = 42,
) -> CVForecastResult:
    """Run expanding-window CV with an inner calibration split per fold."""
    if not (0.05 <= cal_fraction <= 0.5):
        raise ValueError("cal_fraction must be in [0.05, 0.5]")

    df = df.sort_values("date").reset_index(drop=True)

    if point_model != "team_strength":
        if feature_cols is None:
            feature_cols = get_feature_columns(df)
        if not feature_cols:
            raise ValueError("No engineered feature columns found; run add_features() first.")
        required_cols = feature_cols + ["totalGoals"]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows with complete feature/target data after filtering.")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_over_p: list[float] = []
    all_over_y: list[int] = []
    pit_values: list[float] = []
    folds: list[FoldForecastMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Inner calibration split from the end of the training fold
        n_train = len(train_df)
        cal_size = max(1, int(cal_fraction * n_train))
        fit_df = train_df.iloc[: n_train - cal_size].copy()
        cal_df = train_df.iloc[n_train - cal_size :].copy()

        model, scaler, used_features = _fit_point_model(
            fit_df,
            point_model=point_model,
            feature_cols=feature_cols,
        )

        mu_cal = _predict_mu(
            model,
            cal_df,
            point_model=point_model,
            scaler=scaler,
            feature_cols=used_features,
        )
        y_cal = cal_df["totalGoals"].to_numpy(dtype=float)

        mu_test = _predict_mu(
            model,
            test_df,
            point_model=point_model,
            scaler=scaler,
            feature_cols=used_features,
        )
        y_test = test_df["totalGoals"].to_numpy(dtype=float)

        # Fit distribution calibration params on calibration slice only
        dist_nll = float("nan")
        pmf_test: np.ndarray
        if dist_model == "poisson":
            pmf_test = poisson_pmf_matrix(mu_test, max_goals=max_goals)
            dist_nll = poisson_nll(y_test, mu_test)
        elif dist_model == "nb2":
            alpha_hat = fit_nb2_alpha(y_cal, mu_cal)
            pmf_test = nb2_pmf_matrix(mu_test, alpha=alpha_hat, max_goals=max_goals)
            # NLL under NB2 via pmf lookup (avoids recomputing lgamma for y_test)
            p_y = np.take_along_axis(pmf_test, y_test.astype(int)[:, None].clip(0, max_goals), axis=1).squeeze(1)
            dist_nll = float(-np.mean(np.log(np.clip(p_y, 1e-12, 1.0))))
        elif dist_model == "poisson_mixture":
            w_hat, m_hat = fit_poisson_mixture(y_cal, mu_cal, max_goals=max_goals)
            pmf_test = poisson_mixture_pmf_matrix(mu_test, weight=w_hat, multiplier=m_hat, max_goals=max_goals)
            p_y = np.take_along_axis(pmf_test, y_test.astype(int)[:, None].clip(0, max_goals), axis=1).squeeze(1)
            dist_nll = float(-np.mean(np.log(np.clip(p_y, 1e-12, 1.0))))
        else:
            raise ValueError(f"Unknown dist_model: {dist_model}")

        # Point metrics
        mae = float(mean_absolute_error(y_test, mu_test))
        rmse = float(root_mean_squared_error(y_test, mu_test))
        p_nll = poisson_nll(y_test, mu_test)

        # Proper scoring rule for the chosen distribution
        crps = crps_from_pmf(pmf_test, y_test.astype(int))
        pit_values.extend(randomized_pit(pmf_test, y_test.astype(int)).tolist())

        # Over/under event evaluation
        p_over = prob_over_from_pmf(pmf_test, threshold=threshold)
        y_over = (y_test > threshold).astype(int)
        over_brier = _brier_score(p_over, y_over)
        over_ll = _log_loss(p_over, y_over)

        all_over_p.extend(p_over.tolist())
        all_over_y.extend(y_over.tolist())

        # Conformal interval based on calibration residuals (split-conformal)
        lo, hi, q = split_conformal_interval(y_cal, mu_cal, mu_test, alpha=0.1, clip_lower=0.0)
        _ = lo, hi  # intervals can be returned by higher-level code if needed

        folds.append(
            FoldForecastMetrics(
                fold=fold,
                n_train=len(fit_df),
                n_cal=len(cal_df),
                n_test=len(test_df),
                mae=mae,
                rmse=rmse,
                poisson_nll=p_nll,
                dist_nll=dist_nll,
                crps=crps,
                over_brier=over_brier,
                over_log_loss=over_ll,
                conformal_q=float(q),
            )
        )

    bins = reliability_curve(np.asarray(all_over_p), np.asarray(all_over_y), n_bins=10)

    return CVForecastResult(
        point_model=point_model,
        dist_model=dist_model,
        threshold=threshold,
        max_goals=max_goals,
        folds=folds,
        reliability_bins=bins,
        pit_values=pit_values,
    )
