"""
Model training and evaluation utilities.

This module defines functions to train regression models that predict the
total goals scored in an NHL game. It supports:
- Random Forest (baseline)
- XGBoost (gradient boosting)
- Poisson regression (captures count data nature)
- Ensemble stacking (combines multiple models)
- Time-series cross-validation
- Model comparison and visualization

Optimized XGBoost parameters (beats baseline by ~0.6%):
    max_depth=2, learning_rate=0.01, n_estimators=150,
    reg_alpha=1.0, reg_lambda=2.0, subsample=0.7,
    colsample_bytree=0.7, min_child_weight=7

Usage:
    from src.features import add_features
    from src.model import train_xgboost, train_poisson, train_ensemble, compare_models

    df_features = add_features(df, include_goalies=True)
    result = train_xgboost(df_features)
    print(f"MAE: {result.mae:.4f} vs Baseline: {result.baseline_mae:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import config
from .logging_config import get_logger
from .validation import validate_features, validate_target

logger = get_logger(__name__)

# Optional XGBoost import
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class TrainingResult:
    """Container for training results and artifacts."""
    model: Any  # RandomForestRegressor or XGBRegressor
    model_type: str
    feature_names: List[str]
    y_test: pd.Series
    y_pred: np.ndarray
    mae: float
    rmse: float
    baseline_mae: float = 0.0


@dataclass
class CVResult:
    """Container for cross-validation results."""
    model_type: str
    mae_scores: List[float]
    rmse_scores: List[float]
    mae_mean: float = field(init=False)
    mae_std: float = field(init=False)
    rmse_mean: float = field(init=False)
    rmse_std: float = field(init=False)

    def __post_init__(self):
        self.mae_mean = np.mean(self.mae_scores)
        self.mae_std = np.std(self.mae_scores)
        self.rmse_mean = np.mean(self.rmse_scores)
        self.rmse_std = np.std(self.rmse_scores)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect feature columns from the DataFrame.

    Looks for columns matching the pattern from add_features():
    home_* and away_* columns that are numeric features.
    """
    feature_prefixes = (
        # Basic rolling features
        "home_avg_", "away_avg_", "home_win_", "away_win_",
        "home_rest_", "away_rest_", "home_is_", "away_is_",
        "home_games_", "away_games_",
        # Multi-window features (e.g., home_avg_GF_5g, home_std_GF_10g)
        "home_std_", "away_std_",
        # Goalie features (numeric only)
        "home_goalie_sv_", "away_goalie_sv_",
        "home_goalie_gaa", "away_goalie_gaa",
        "home_goalie_vs_", "away_goalie_vs_",
        # Head-to-head and venue features
        "h2h_", "venue_",
        # Interaction features
        "scoring_", "opponent_", "rest_advantage", "form_diff", "combined_",
        "opp_threat", "xg_", "home_xg_", "away_xg_",
        # Temporal features
        "month", "day_of_week", "is_weekend", "days_into_season",
        "season_progress", "is_late_season", "is_early_season",
    )
    # Also include legacy column names for backwards compatibility
    legacy_prefixes = ("homeTeam_avg_", "awayTeam_avg_")

    feature_cols = []
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in feature_prefixes + legacy_prefixes):
            feature_cols.append(col)

    return feature_cols


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare the feature matrix and target vector for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and target column `totalGoals`.
    feature_cols : iterable of str, optional
        Columns to use as features. If None, auto-detects feature columns.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (total goals).
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if not feature_cols:
        raise ValueError("No feature columns found. Run add_features() first.")

    X = df[list(feature_cols)].copy()
    y = df["totalGoals"].copy()
    return X, y


def _prepare_clean_feature_frame(
    df: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return chronologically sorted feature/target data with scoped NaN dropping.

    Rows are dropped only when required model inputs (selected features + target)
    are missing. Unrelated nullable columns must not remove training samples.
    """
    validate_features(df)
    validate_target(df)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_names = get_feature_columns(df_sorted)
    else:
        feature_names = list(feature_cols)

    if not feature_names:
        raise ValueError("No feature columns found. Run add_features() first.")

    missing_features = [c for c in feature_names if c not in df_sorted.columns]
    if missing_features:
        raise ValueError(f"Missing requested feature columns: {missing_features[:5]}")

    required_cols = feature_names + ["totalGoals"]
    n_before = len(df_sorted)
    df_clean = df_sorted.dropna(subset=required_cols)
    n_dropped = n_before - len(df_clean)
    if n_dropped > 0:
        logger.info(
            "Dropped %d rows with missing model inputs (%.1f%% of data)",
            n_dropped, 100 * n_dropped / n_before
        )

    if df_clean.empty:
        raise ValueError("No rows with complete feature/target data.")

    X = df_clean[feature_names].copy()
    y = df_clean["totalGoals"].copy()
    return X, y, feature_names


def prepare_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Prepare and split data for training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float, optional
        Fraction of data for testing (from end of time series).
        Defaults to config value.
    feature_cols : list of str, optional
        Feature columns to use.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Apply config defaults
    if test_size is None:
        test_size = config.model.test_size

    X, y, feature_names = _prepare_clean_feature_frame(df, feature_cols)

    # Chronological split
    n = len(X)
    split_idx = int((1.0 - test_size) * n)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Invalid test_size.")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_names


def train_random_forest(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> TrainingResult:
    """Train a Random Forest regressor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing.
    n_estimators : int
        Number of trees.
    max_depth : int, optional
        Maximum tree depth.
    random_state : int
        Random seed.

    Returns
    -------
    TrainingResult
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, test_size)
    print(f"Training Random Forest on {len(X_train)} games, testing on {len(X_test)} games")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    baseline_pred = np.full_like(y_pred, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"Random Forest MAE: {mae:.3f} | RMSE: {rmse:.3f} | Baseline MAE: {baseline_mae:.3f}")

    return TrainingResult(
        model=model,
        model_type="RandomForest",
        feature_names=feature_names,
        y_test=y_test,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
    )


def train_xgboost(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> TrainingResult:
    """Train an XGBoost regressor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Boosting learning rate.
    random_state : int
        Random seed.

    Returns
    -------
    TrainingResult
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, test_size)
    print(f"Training XGBoost on {len(X_train)} games, testing on {len(X_test)} games")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    baseline_pred = np.full_like(y_pred, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"XGBoost MAE: {mae:.3f} | RMSE: {rmse:.3f} | Baseline MAE: {baseline_mae:.3f}")

    return TrainingResult(
        model=model,
        model_type="XGBoost",
        feature_names=feature_names,
        y_test=y_test,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
    )


def train_poisson(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    alpha: float = 1.0,
) -> TrainingResult:
    """Train a Poisson regressor.

    Poisson regression is appropriate for count data like goals.
    It models the log of the expected count as a linear function of features.

    This is particularly useful for hockey because:
    - Goals are count data (non-negative integers)
    - The Poisson distribution naturally handles the variance structure
    - It enforces non-negative predictions

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing.
    alpha : float
        Regularization strength (L2 penalty).

    Returns
    -------
    TrainingResult
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, test_size)
    print(f"Training Poisson Regression on {len(X_train)} games, testing on {len(X_test)} games")

    # Bundle preprocessing + estimator so persisted artifacts are inference-safe.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    baseline_pred = np.full_like(y_pred, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"Poisson MAE: {mae:.3f} | RMSE: {rmse:.3f} | Baseline MAE: {baseline_mae:.3f}")

    return TrainingResult(
        model=model,
        model_type="Poisson",
        feature_names=feature_names,
        y_test=y_test,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
    )


def train_ensemble(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train a stacked ensemble combining multiple model types.

    The ensemble uses:
    - Level 1 (base models):
      - Random Forest: Captures non-linear patterns
      - Ridge Regression: Linear baseline with regularization
      - XGBoost (if available): Gradient boosting
    - Level 2 (meta-learner):
      - Ridge Regression: Learns optimal model combination

    This approach often outperforms individual models by combining
    their strengths and reducing variance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    TrainingResult
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, test_size)
    print(f"Training Stacked Ensemble on {len(X_train)} games, testing on {len(X_test)} games")

    # Define base estimators
    estimators = [
        ("rf", RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        )),
        ("ridge", Ridge(alpha=1.0)),
        ("poisson", PoissonRegressor(alpha=1.0, max_iter=500)),
    ]

    # Add XGBoost if available
    if HAS_XGBOOST:
        estimators.append(("xgb", xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )))

    # Create stacking regressor with Ridge as meta-learner
    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,  # Use 5-fold CV to generate meta-features
        n_jobs=-1,
    )

    # Bundle preprocessing + estimator so persisted artifacts are inference-safe.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("stack", stack_model),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    baseline_pred = np.full_like(y_pred, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"Ensemble MAE: {mae:.3f} | RMSE: {rmse:.3f} | Baseline MAE: {baseline_mae:.3f}")

    return TrainingResult(
        model=model,
        model_type="Ensemble",
        feature_names=feature_names,
        y_test=y_test,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
    )


def cross_validate(
    df: pd.DataFrame,
    model_type: Literal["rf", "xgb"] = "rf",
    n_splits: int = 5,
    **model_kwargs,
) -> CVResult:
    """Perform time-series cross-validation.

    Uses an expanding window approach where each fold uses all prior data
    for training and a subsequent chunk for testing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    model_type : {"rf", "xgb"}
        Model type: "rf" for Random Forest, "xgb" for XGBoost.
    n_splits : int
        Number of CV folds.
    **model_kwargs
        Additional arguments passed to the model constructor.

    Returns
    -------
    CVResult
    """
    if model_type == "xgb" and not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    # Prepare data
    X, y, _ = _prepare_clean_feature_frame(df)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    rmse_scores = []

    model_name = "XGBoost" if model_type == "xgb" else "Random Forest"
    print(f"Running {n_splits}-fold time-series CV for {model_name}...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_type == "xgb":
            model = xgb.XGBRegressor(
                n_estimators=model_kwargs.get("n_estimators", 200),
                max_depth=model_kwargs.get("max_depth", 6),
                learning_rate=model_kwargs.get("learning_rate", 0.1),
                random_state=model_kwargs.get("random_state", 42),
                n_jobs=-1,
                verbosity=0,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=model_kwargs.get("n_estimators", 200),
                max_depth=model_kwargs.get("max_depth", None),
                random_state=model_kwargs.get("random_state", 42),
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"  Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f} (train={len(X_train)}, test={len(X_test)})")

    result = CVResult(model_type=model_name, mae_scores=mae_scores, rmse_scores=rmse_scores)
    print(f"\n{model_name} CV Results: MAE={result.mae_mean:.3f} ± {result.mae_std:.3f}")

    return result


def optimize_hyperparameters(
    df: pd.DataFrame,
    n_trials: int = 100,
    n_cv_folds: int | None = None,
    timeout: int | None = None,
    show_progress: bool = True,
    objective_metric: Literal["mae", "weighted_prob"] = "mae",
    dist_model: Literal["nb2", "poisson", "poisson_mixture"] = "nb2",
    threshold: float = 6.5,
    tune_splits: int = 3,
) -> Dict[str, Any]:
    """Use Optuna to find optimal XGBoost hyperparameters.

    Supports:
    - ``objective_metric="mae"``: classic MAE minimization with time-series CV.
    - ``objective_metric="weighted_prob"``: weighted probabilistic objective
      normalized to a team-strength baseline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    n_trials : int
        Number of optimization trials to run.
    n_cv_folds : int, optional
        Number of cross-validation folds. Defaults to config value.
    timeout : int, optional
        Timeout in seconds. If provided, stops after this time.
    show_progress : bool
        If True, show progress bar during optimization.
    objective_metric : {"mae", "weighted_prob"}
        Optimization objective.
    dist_model : {"nb2", "poisson", "poisson_mixture"}
        Distribution used when objective_metric="weighted_prob".
    threshold : float
        Over/under threshold used for Brier component in weighted objective.
    tune_splits : int
        Number of CV splits used in weighted-probability tuning.

    Returns
    -------
    dict
        Best hyperparameters found.

    Raises
    ------
    ImportError
        If Optuna is not installed.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter optimization. "
            "Install it with: pip install optuna"
        )

    if not HAS_XGBOOST:
        raise ImportError("XGBoost is required for hyperparameter optimization.")

    if n_cv_folds is None:
        n_cv_folds = config.model.cv_folds

    logger.info(
        "Starting hyperparameter optimization with %d trials (objective=%s)",
        n_trials,
        objective_metric,
    )

    baseline_metrics: Dict[str, float] | None = None
    if objective_metric == "weighted_prob":
        from .evaluation import time_series_cv_forecast

        baseline = time_series_cv_forecast(
            df,
            point_model="team_strength",
            dist_model=dist_model,
            threshold=threshold,
            n_splits=tune_splits,
            max_goals=20,
        )
        baseline_metrics = baseline.metrics_mean
        logger.info("Weighted-probability baseline metrics: %s", baseline_metrics)
    else:
        # Prepare data only for MAE objective path.
        X, y, _ = _prepare_clean_feature_frame(df)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        if objective_metric == "mae":
            # Time-series cross-validation MAE objective.
            tscv = TimeSeriesSplit(n_splits=n_cv_folds)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBRegressor(
                    **params,
                    random_state=config.model.random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, pred))

            return float(np.mean(scores))

        from .evaluation import time_series_cv_forecast

        result = time_series_cv_forecast(
            df,
            point_model="xgb",
            dist_model=dist_model,
            threshold=threshold,
            n_splits=tune_splits,
            xgb_params=params,
            max_goals=20,
        )
        metrics = result.metrics_mean
        base = baseline_metrics or metrics

        def _safe_ratio(metric_name: str) -> float:
            denom = max(float(base[metric_name]), 1e-9)
            return float(metrics[metric_name]) / denom

        weighted_score = (
            0.35 * _safe_ratio("mae")
            + 0.30 * _safe_ratio("crps")
            + 0.20 * _safe_ratio("dist_nll")
            + 0.15 * _safe_ratio("over_brier")
        )
        return float(weighted_score)

    # Configure Optuna logging
    if not show_progress:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=config.model.random_state),
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
    )

    best_params = study.best_params
    logger.info("Best objective value: %.4f", study.best_value)
    logger.info("Best parameters: %s", best_params)

    return best_params


def compare_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    include_ensemble: bool = True,
) -> pd.DataFrame:
    """Train and compare multiple models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing.
    include_ensemble : bool
        If True, include the stacked ensemble (slower but often best).

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model.
    """
    results = []

    # Random Forest
    print("\n" + "=" * 50)
    rf_result = train_random_forest(df, test_size=test_size)
    results.append({
        "Model": "Random Forest",
        "MAE": rf_result.mae,
        "RMSE": rf_result.rmse,
        "vs Baseline": f"{(1 - rf_result.mae / rf_result.baseline_mae) * 100:.1f}% better",
    })

    # XGBoost
    if HAS_XGBOOST:
        print("\n" + "-" * 50)
        xgb_result = train_xgboost(df, test_size=test_size)
        results.append({
            "Model": "XGBoost",
            "MAE": xgb_result.mae,
            "RMSE": xgb_result.rmse,
            "vs Baseline": f"{(1 - xgb_result.mae / xgb_result.baseline_mae) * 100:.1f}% better",
        })
    else:
        print("XGBoost not installed, skipping.")

    # Poisson Regression
    print("\n" + "-" * 50)
    try:
        poisson_result = train_poisson(df, test_size=test_size)
        results.append({
            "Model": "Poisson",
            "MAE": poisson_result.mae,
            "RMSE": poisson_result.rmse,
            "vs Baseline": f"{(1 - poisson_result.mae / poisson_result.baseline_mae) * 100:.1f}% better",
        })
    except Exception as e:
        print(f"Poisson regression failed: {e}")

    # Stacked Ensemble (optional, slower)
    if include_ensemble:
        print("\n" + "-" * 50)
        try:
            ensemble_result = train_ensemble(df, test_size=test_size)
            results.append({
                "Model": "Ensemble",
                "MAE": ensemble_result.mae,
                "RMSE": ensemble_result.rmse,
                "vs Baseline": f"{(1 - ensemble_result.mae / ensemble_result.baseline_mae) * 100:.1f}% better",
            })
        except Exception as e:
            print(f"Ensemble training failed: {e}")

    # Baseline
    results.append({
        "Model": "Baseline (mean)",
        "MAE": rf_result.baseline_mae,
        "RMSE": np.nan,
        "vs Baseline": "-",
    })

    comparison = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    return comparison


# Legacy alias for backwards compatibility
def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
) -> TrainingResult:
    """Train a model and evaluate. Legacy function for backwards compatibility."""
    return train_random_forest(
        df,
        test_size=test_size,
        n_estimators=n_estimators,
        random_state=random_state,
    )


def plot_feature_importance(
    result: TrainingResult,
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> None:
    """Plot feature importances from a trained model.

    Parameters
    ----------
    result : TrainingResult
        The result from training.
    top_n : int
        Number of top features to show.
    save_path : Path, optional
        If provided, save the figure to this path.
    """
    if hasattr(result.model, "feature_importances_"):
        importances = result.model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute.")
        return

    # Sort and take top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [result.feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances ({result.model_type})")
    plt.barh(range(len(top_importances)), top_importances[::-1], align="center")
    plt.yticks(range(len(top_importances)), top_names[::-1])
    plt.xlabel("Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_predictions(result: TrainingResult, save_path: Optional[Path] = None) -> None:
    """Plot predicted vs actual total goals.

    Parameters
    ----------
    result : TrainingResult
        The result from training.
    save_path : Path, optional
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(result.y_test, result.y_pred, alpha=0.5, edgecolors="none")

    min_val = min(result.y_test.min(), result.y_pred.min())
    max_val = max(result.y_test.max(), result.y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    ax1.set_xlabel("Actual Total Goals")
    ax1.set_ylabel("Predicted Total Goals")
    ax1.set_title(f"{result.model_type}: Predictions vs Actuals (MAE={result.mae:.2f})")
    ax1.legend()

    # Residual distribution
    ax2 = axes[1]
    residuals = result.y_pred - result.y_test.values
    ax2.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="r", linestyle="--", label="Zero error")
    ax2.set_xlabel("Prediction Error (Predicted - Actual)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved predictions plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cv_results(cv_results: List[CVResult], save_path: Optional[Path] = None) -> None:
    """Plot cross-validation results comparison.

    Parameters
    ----------
    cv_results : list of CVResult
        Results from cross_validate() for each model.
    save_path : Path, optional
        If provided, save the figure to this path.
    """
    models = [r.model_type for r in cv_results]
    mae_means = [r.mae_mean for r in cv_results]
    mae_stds = [r.mae_std for r in cv_results]

    x = np.arange(len(models))
    plt.figure(figsize=(8, 5))
    plt.bar(x, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
    plt.xticks(x, models)
    plt.ylabel("MAE")
    plt.title("Cross-Validation Results (mean ± std)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved CV results plot to {save_path}")
    else:
        plt.show()
    plt.close()


def save_model(
    result: TrainingResult,
    path: Path,
    seasons: Optional[List[str]] = None,
    use_artifact: bool = True,
) -> None:
    """Save a trained model to disk.

    Parameters
    ----------
    result : TrainingResult
        The training result to save.
    path : Path
        Path to save the model (without extension if use_artifact=True).
    seasons : list of str, optional
        Seasons used for training data (for metadata).
    use_artifact : bool
        If True, save as artifact with metadata (recommended).
        If False, use legacy format for backwards compatibility.
    """
    path = Path(path)

    if use_artifact:
        from .artifacts import ModelArtifact
        artifact = ModelArtifact.from_training_result(result, seasons=seasons)
        artifact.save(path)
        logger.info("Model artifact saved to %s", path)
    else:
        # Legacy format for backwards compatibility
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": result.model,
            "model_type": result.model_type,
            "feature_names": result.feature_names,
        }, path)
        logger.info("Model saved to %s (legacy format)", path)


def load_model(path: Path) -> Tuple[Any, str, List[str]]:
    """Load a trained model from disk.

    Supports both legacy format (.joblib) and new artifact format.

    Parameters
    ----------
    path : Path
        Path to the model file.

    Returns
    -------
    model, model_type, feature_names
    """
    path = Path(path)

    # Check if this is an artifact (has .json metadata file)
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists() or not path.suffix:
        # Try loading as artifact
        try:
            from .artifacts import ModelArtifact
            artifact = ModelArtifact.load(path)
            return artifact.model, artifact.metadata.model_type, artifact.metadata.feature_names
        except Exception:
            pass

    # Fall back to legacy format
    if path.suffix != ".joblib":
        path = path.with_suffix(".joblib")

    data = joblib.load(path)
    return data["model"], data.get("model_type", "Unknown"), data["feature_names"]


def load_artifact(path: Path) -> "ModelArtifact":
    """Load a model artifact with full metadata.

    Parameters
    ----------
    path : Path
        Path to the artifact (without extension).

    Returns
    -------
    ModelArtifact
        Complete artifact with model and metadata.
    """
    from .artifacts import ModelArtifact
    return ModelArtifact.load(path)
