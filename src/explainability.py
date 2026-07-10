"""
Explainability and feature-stability utilities.

Goals:
- SHAP explanations (when optional dependency `shap` is installed)
- Robust fallbacks (permutation importance) when SHAP isn't available
- Feature stability across seasons (rank correlations of importances)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import config
from .model import get_feature_columns


@dataclass(frozen=True)
class SeasonImportance:
    season: str
    importances: pd.Series  # index=feature, value=importance


def xgb_gain_importance(model) -> pd.Series:
    """Extract XGBoost gain-based feature importances as a pandas Series."""
    booster = getattr(model, "get_booster", None)
    if booster is None:
        raise ValueError("Model does not look like an XGBoost estimator")
    score = model.get_booster().get_score(importance_type="gain")
    if not score:
        # Fall back to sklearn-style feature_importances_
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
            names = list(getattr(model, "feature_names_in_", []))
            if not names:
                names = [f"f{i}" for i in range(len(values))]
            return pd.Series(values, index=names).sort_values(ascending=False)
        raise ValueError("Could not extract feature importances")
    return pd.Series(score).sort_values(ascending=False)


def permutation_importance_df(
    model, X: pd.DataFrame, y: pd.Series, *, n_repeats: int = 5
) -> pd.Series:
    """Permutation importance as a fallback explanation method."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=config.model.random_state,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def shap_summary_plot_xgb(model, X: pd.DataFrame, *, out_path: Path, max_display: int = 25) -> None:
    """Create a SHAP summary plot for an XGBoost model (requires `shap`)."""
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is not installed. Run: pip install shap") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def feature_stability_matrix(importances: pd.DataFrame) -> pd.DataFrame:
    """Compute season-by-season Spearman rank correlation matrix."""
    ranked = importances.rank(axis=0, ascending=False, method="average")
    # Spearman correlation = Pearson correlation on ranks
    return ranked.corr(method="pearson")


def plot_stability_heatmap(stability: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(stability.to_numpy(), vmin=-1, vmax=1)
    plt.colorbar(label="Spearman correlation")
    plt.xticks(range(len(stability.columns)), stability.columns, rotation=45, ha="right")
    plt.yticks(range(len(stability.index)), stability.index)
    plt.title("Feature Stability Across Seasons")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def train_xgb_per_season_importance(
    df: pd.DataFrame,
    *,
    seasons: list[str],
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Train one XGBoost model per season and return gain importances.

    Returns a DataFrame with index=feature and columns=season.
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("This function requires xgboost") from e

    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No engineered feature columns found; run add_features() first.")

    params = dict(config.model.xgb_params)
    params.update({"random_state": config.model.random_state, "n_jobs": -1, "verbosity": 0})

    out: dict[str, pd.Series] = {}
    for season in seasons:
        df_s = df[df["season"] == season].dropna().copy()
        if df_s.empty:
            continue
        X = df_s[feature_cols]
        y = df_s["totalGoals"].astype(float)
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        imp = xgb_gain_importance(model)
        # Ensure every season shares the same feature index
        out[season] = imp

    if not out:
        raise ValueError("No seasons produced importances (check season codes and data).")

    imp_df = pd.DataFrame(out).fillna(0.0)
    return imp_df
