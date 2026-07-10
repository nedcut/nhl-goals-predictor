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
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import config
from .conformal import split_conformal_interval
from .model import get_feature_columns
from .probabilistic import (
    crps_per_game_from_pmf,
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
from .double_poisson import DoublePoissonModel
from .team_strength import TeamStrengthPoissonModel

PointModel = Literal["xgb", "poisson_glm", "team_strength", "double_poisson"]
DistModel = Literal["poisson", "nb2", "poisson_mixture"]

# Point models that use only team IDs (no engineered rolling features).
_TEAM_ID_POINT_MODELS = frozenset({"team_strength", "double_poisson"})


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


def _threshold_label(threshold: float) -> str:
    """Stable string key for a total-goals line (e.g. 6.5 -> '6.5')."""
    return f"{float(threshold):g}"


@dataclass(frozen=True)
class CVForecastResult:
    point_model: PointModel
    dist_model: DistModel
    threshold: float
    max_goals: int
    folds: list[FoldForecastMetrics]
    reliability_bins: list
    pit_values: list[float]
    diagnostics: list[dict[str, Any]] | None = None
    # Per-game score components pooled across all folds, keyed by metric name.
    # game_key aligns the same game across models so scores can be paired.
    per_game: dict[str, Any] | None = None
    # Multi-threshold over/under scores: label -> {brier, log_loss}.
    # Mean of per-fold metrics (same aggregation as metrics_mean over_brier).
    threshold_metrics: dict[str, dict[str, float]] | None = None
    # Reliability curves per threshold label (optional diagnostic).
    reliability_by_threshold: dict[str, list] | None = None

    _METRIC_KEYS = (
        "mae",
        "rmse",
        "poisson_nll",
        "dist_nll",
        "crps",
        "over_brier",
        "over_log_loss",
        "conformal_q",
    )

    @property
    def metrics_mean(self) -> dict[str, float]:
        return {k: float(np.mean([getattr(f, k) for f in self.folds])) for k in self._METRIC_KEYS}

    @property
    def metrics_std(self) -> dict[str, float]:
        """Across-fold standard deviation for each metric (sampling noise proxy)."""
        return {k: float(np.std([getattr(f, k) for f in self.folds])) for k in self._METRIC_KEYS}


def _fit_point_model(
    df_fit: pd.DataFrame,
    *,
    point_model: PointModel,
    feature_cols: Optional[list[str]],
    xgb_params: Optional[dict[str, Any]] = None,
) -> tuple[object, Optional[StandardScaler], list[str]]:
    if point_model == "team_strength":
        model = TeamStrengthPoissonModel().fit(df_fit)
        return model, None, []

    if point_model == "double_poisson":
        model = DoublePoissonModel().fit(df_fit)
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
        if xgb_params:
            params.update(xgb_params)
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
    if point_model in _TEAM_ID_POINT_MODELS:
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
    thresholds: list[float] | None = None,
    n_splits: int = 5,
    cal_fraction: float = 0.2,
    max_goals: int = 20,
    feature_cols: Optional[list[str]] = None,
    random_state: int = 42,
    return_diagnostics: bool = False,
    xgb_params: Optional[dict[str, Any]] = None,
) -> CVForecastResult:
    """Run expanding-window CV with an inner calibration split per fold.

    Parameters
    ----------
    threshold : float
        Primary over/under line used for fold metrics, per-game Brier, and the
        top-level reliability curve (champion weighting stays on this line).
    thresholds : list of float, optional
        Additional lines to score (Brier + log-loss + reliability). If None,
        only ``threshold`` is reported under ``threshold_metrics``.
    return_diagnostics : bool
        If True, include row-level fold diagnostics in the result payload.
    xgb_params : dict, optional
        Per-run XGBoost parameter override for point_model="xgb".
    """
    if not (0.05 <= cal_fraction <= 0.5):
        raise ValueError("cal_fraction must be in [0.05, 0.5]")

    eval_thresholds = list(thresholds) if thresholds is not None else [threshold]
    if not eval_thresholds:
        raise ValueError("thresholds must be non-empty when provided")

    df = df.sort_values("date").reset_index(drop=True)

    if point_model not in _TEAM_ID_POINT_MODELS:
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
    diagnostics_rows: list[dict[str, Any]] = []

    # Per-threshold accumulators (label -> fold scores / pooled probs).
    thresh_fold_brier: dict[str, list[float]] = {_threshold_label(t): [] for t in eval_thresholds}
    thresh_fold_ll: dict[str, list[float]] = {_threshold_label(t): [] for t in eval_thresholds}
    thresh_all_p: dict[str, list[float]] = {_threshold_label(t): [] for t in eval_thresholds}
    thresh_all_y: dict[str, list[int]] = {_threshold_label(t): [] for t in eval_thresholds}

    # Per-game score components pooled across folds (for paired significance tests).
    per_game_keys: list[str] = []
    per_game_abs_err: list[float] = []
    per_game_crps: list[float] = []
    per_game_nll: list[float] = []
    per_game_brier: list[float] = []

    def _rest_bucket(rest_diff: float) -> str:
        if pd.isna(rest_diff):
            return "unknown"
        if rest_diff <= -2:
            return "away_rest_advantage"
        if rest_diff >= 2:
            return "home_rest_advantage"
        return "even_rest"

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
            xgb_params=xgb_params,
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
        pmf_test: np.ndarray
        if dist_model == "poisson":
            pmf_test = poisson_pmf_matrix(mu_test, max_goals=max_goals)
        elif dist_model == "nb2":
            alpha_hat = fit_nb2_alpha(y_cal, mu_cal)
            pmf_test = nb2_pmf_matrix(mu_test, alpha=alpha_hat, max_goals=max_goals)
        elif dist_model == "poisson_mixture":
            w_hat, m_hat = fit_poisson_mixture(y_cal, mu_cal, max_goals=max_goals)
            pmf_test = poisson_mixture_pmf_matrix(mu_test, weight=w_hat, multiplier=m_hat, max_goals=max_goals)
        else:
            raise ValueError(f"Unknown dist_model: {dist_model}")

        # Per-game NLL under the fitted distribution, via PMF lookup. Using the
        # same (renormalized) PMF that drives CRPS/over-probabilities keeps every
        # score consistent with the distribution actually deployed.
        y_idx = y_test.astype(int)[:, None].clip(0, max_goals)
        p_y = np.take_along_axis(pmf_test, y_idx, axis=1).squeeze(1)
        nll_per_game = -np.log(np.clip(p_y, 1e-12, 1.0))
        dist_nll = float(np.mean(nll_per_game))

        # Point metrics
        abs_err = np.abs(y_test - mu_test)
        mae = float(np.mean(abs_err))
        rmse = float(root_mean_squared_error(y_test, mu_test))
        p_nll = poisson_nll(y_test, mu_test)

        # Proper scoring rule for the chosen distribution
        crps_per_game = crps_per_game_from_pmf(pmf_test, y_idx.squeeze(1))
        crps = float(np.mean(crps_per_game))
        pit_values.extend(randomized_pit(pmf_test, y_test.astype(int)).tolist())

        # Primary over/under event evaluation (champion / fold metrics)
        p_over = prob_over_from_pmf(pmf_test, threshold=threshold)
        y_over = (y_test > threshold).astype(int)
        brier_per_game = (p_over - y_over) ** 2
        over_brier = float(np.mean(brier_per_game))
        over_ll = _log_loss(p_over, y_over)

        all_over_p.extend(p_over.tolist())
        all_over_y.extend(y_over.tolist())

        # Multi-threshold over/under (Brier + log-loss per line)
        for t in eval_thresholds:
            label = _threshold_label(t)
            p_t = prob_over_from_pmf(pmf_test, threshold=t)
            y_t = (y_test > t).astype(int)
            thresh_fold_brier[label].append(_brier_score(p_t, y_t))
            thresh_fold_ll[label].append(_log_loss(p_t, y_t))
            thresh_all_p[label].extend(p_t.tolist())
            thresh_all_y[label].extend(y_t.tolist())

        # Stable per-game key so the same game can be paired across models.
        game_keys = (
            test_df["date"].astype(str)
            + "|"
            + test_df.get("homeTeam", pd.Series([""] * len(test_df))).astype(str)
            + "|"
            + test_df.get("awayTeam", pd.Series([""] * len(test_df))).astype(str)
        ).to_numpy()
        per_game_keys.extend(game_keys.tolist())
        per_game_abs_err.extend(abs_err.tolist())
        per_game_crps.extend(crps_per_game.tolist())
        per_game_nll.extend(nll_per_game.tolist())
        per_game_brier.extend(brier_per_game.tolist())

        # Conformal interval radius based on calibration residuals (split-conformal).
        # Only the quantile radius q feeds the reported metric; bounds are derivable.
        _, _, q = split_conformal_interval(y_cal, mu_cal, mu_test, alpha=0.1, clip_lower=0.0)

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

        if return_diagnostics:
            p_over_65 = prob_over_from_pmf(pmf_test, threshold=6.5)
            for i, (_, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
                y_true = float(y_test[i])
                mu_pred = float(mu_test[i])
                abs_err = abs(y_true - mu_pred)
                diagnostics_rows.append(
                    {
                        "fold": fold,
                        "date": str(row.get("date", "")),
                        "season": str(row.get("season", "")),
                        "homeTeam": str(row.get("homeTeam", "")),
                        "awayTeam": str(row.get("awayTeam", "")),
                        "y_true": y_true,
                        "mu_pred": mu_pred,
                        "p_over_6_5": float(p_over_65[i]),
                        "abs_error": float(abs_err),
                        "squared_error": float(abs_err**2),
                        "home_is_back_to_back": int(row.get("home_is_back_to_back", 0) or 0),
                        "away_is_back_to_back": int(row.get("away_is_back_to_back", 0) or 0),
                        "is_back_to_back": int(
                            int(row.get("home_is_back_to_back", 0) or 0)
                            or int(row.get("away_is_back_to_back", 0) or 0)
                        ),
                        "month": int(pd.to_datetime(row.get("date")).month) if row.get("date") else -1,
                        "rest_diff_bucket": _rest_bucket(
                            float(row.get("home_rest_days", np.nan))
                            - float(row.get("away_rest_days", np.nan))
                        ),
                    }
                )

    bins = reliability_curve(np.asarray(all_over_p), np.asarray(all_over_y), n_bins=10)

    threshold_metrics = {
        label: {
            "brier": float(np.mean(thresh_fold_brier[label])),
            "log_loss": float(np.mean(thresh_fold_ll[label])),
        }
        for label in thresh_fold_brier
    }
    reliability_by_threshold = {
        label: reliability_curve(
            np.asarray(thresh_all_p[label]),
            np.asarray(thresh_all_y[label]),
            n_bins=10,
        )
        for label in thresh_all_p
    }

    per_game = {
        "game_key": np.asarray(per_game_keys, dtype=object),
        "abs_error": np.asarray(per_game_abs_err, dtype=float),
        "crps": np.asarray(per_game_crps, dtype=float),
        "dist_nll": np.asarray(per_game_nll, dtype=float),
        "over_brier": np.asarray(per_game_brier, dtype=float),
    }

    return CVForecastResult(
        point_model=point_model,
        dist_model=dist_model,
        threshold=threshold,
        max_goals=max_goals,
        folds=folds,
        reliability_bins=bins,
        pit_values=pit_values,
        diagnostics=diagnostics_rows if return_diagnostics else None,
        per_game=per_game,
        threshold_metrics=threshold_metrics,
        reliability_by_threshold=reliability_by_threshold,
    )
