"""
Model training and evaluation utilities.

This module defines functions to train regression models that predict the
total goals scored in an NHL game. It supports:
- Random Forest (baseline)
- XGBoost (gradient boosting)
- Time-series cross-validation
- Model comparison and visualization

Optimized XGBoost parameters (beats baseline by ~0.6%):
    max_depth=2, learning_rate=0.01, n_estimators=150,
    reg_alpha=1.0, reg_lambda=2.0, subsample=0.7,
    colsample_bytree=0.7, min_child_weight=7

Usage:
    from src.features import add_features
    from src.model import train_xgboost, compare_models

    df_features = add_features(df, window=20, include_goalies=True)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

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
        "home_avg_", "away_avg_", "home_win_", "away_win_",
        "home_rest_", "away_rest_", "home_is_", "away_is_",
        "home_games_", "away_games_",
        # Goalie features (numeric only)
        "home_goalie_sv_", "away_goalie_sv_",
        "home_goalie_gaa", "away_goalie_gaa",
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


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Prepare and split data for training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing (from end of time series).
    feature_cols : list of str, optional
        Feature columns to use.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Ensure chronological ordering
    df_sorted = df.sort_values("date").reset_index(drop=True)

    # Drop rows with missing features
    n_before = len(df_sorted)
    df_clean = df_sorted.dropna()
    n_dropped = n_before - len(df_clean)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing features ({n_dropped/n_before:.1%} of data)")

    if df_clean.empty:
        raise ValueError("No rows with complete feature data.")

    X, y = prepare_features(df_clean, feature_cols)
    feature_names = list(X.columns)

    # Chronological split
    n = len(df_clean)
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
    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_clean = df_sorted.dropna()
    X, y = prepare_features(df_clean)

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


def compare_models(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """Train and compare multiple models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    test_size : float
        Fraction of data for testing.

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model.
    """
    results = []

    # Random Forest
    rf_result = train_random_forest(df, test_size=test_size)
    results.append({
        "Model": "Random Forest",
        "MAE": rf_result.mae,
        "RMSE": rf_result.rmse,
        "vs Baseline": f"{(1 - rf_result.mae / rf_result.baseline_mae) * 100:.1f}% better",
    })

    # XGBoost
    if HAS_XGBOOST:
        xgb_result = train_xgboost(df, test_size=test_size)
        results.append({
            "Model": "XGBoost",
            "MAE": xgb_result.mae,
            "RMSE": xgb_result.rmse,
            "vs Baseline": f"{(1 - xgb_result.mae / xgb_result.baseline_mae) * 100:.1f}% better",
        })
    else:
        print("XGBoost not installed, skipping.")

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


def save_model(result: TrainingResult, path: Path) -> None:
    """Save a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": result.model,
        "model_type": result.model_type,
        "feature_names": result.feature_names,
    }, path)
    print(f"Model saved to {path}")


def load_model(path: Path) -> Tuple[Any, str, List[str]]:
    """Load a trained model from disk.

    Returns
    -------
    model, model_type, feature_names
    """
    data = joblib.load(path)
    return data["model"], data.get("model_type", "Unknown"), data["feature_names"]
