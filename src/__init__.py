"""Top-level package for the NHL total goals prediction project.

Heavy modules are exported lazily so importing `src` (or `src.config`) does not
force optional ML dependencies to import at package load time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import config

__all__ = [
    # Config
    "config",
    # Data
    "build_dataset",
    # Features
    "add_features",
    # Model
    "train_xgboost",
    "train_random_forest",
    "cross_validate",
    "optimize_hyperparameters",
    "save_model",
    "load_model",
    "TrainingResult",
    "CVResult",
    # Probabilistic evaluation
    "time_series_cv_forecast",
    "CVForecastResult",
    # Distribution helpers
    "poisson_pmf_matrix",
    "nb2_pmf_matrix",
    "poisson_mixture_pmf_matrix",
    "prob_over_from_pmf",
    # Championing
    "weighted_score",
    "rank_candidates",
    "choose_champion",
    "per_game_weighted_scores",
    "compare_models_significance",
    # Significance
    "paired_bootstrap",
    "PairedComparison",
    # Portfolio orchestration
    "run_portfolio_pipeline",
    # Baselines
    "TeamStrengthPoissonModel",
    "TeamStrengthConfig",
    # Artifacts
    "ModelArtifact",
    "ModelMetadata",
    # Registry
    "ModelRegistry",
    "get_registry",
    # Validation
    "ValidationError",
    "validate_game_data",
    "validate_features",
    # Monitoring & drift
    "log_predictions",
    "load_prediction_log",
    "reconcile_outcomes",
    "realized_metrics",
    "population_stability_index",
    "feature_drift",
    "assess_overall_drift",
    "monitoring_summary",
]

_LAZY_ATTR_MODULES = {
    "build_dataset": ".data",
    "add_features": ".features",
    "ValidationError": ".validation",
    "validate_game_data": ".validation",
    "validate_features": ".validation",
    "train_xgboost": ".model",
    "train_random_forest": ".model",
    "cross_validate": ".model",
    "optimize_hyperparameters": ".model",
    "save_model": ".model",
    "load_model": ".model",
    "TrainingResult": ".model",
    "CVResult": ".model",
    "time_series_cv_forecast": ".evaluation",
    "CVForecastResult": ".evaluation",
    "poisson_pmf_matrix": ".probabilistic",
    "nb2_pmf_matrix": ".probabilistic",
    "poisson_mixture_pmf_matrix": ".probabilistic",
    "prob_over_from_pmf": ".probabilistic",
    "weighted_score": ".champion",
    "rank_candidates": ".champion",
    "choose_champion": ".champion",
    "per_game_weighted_scores": ".champion",
    "compare_models_significance": ".champion",
    "paired_bootstrap": ".significance",
    "PairedComparison": ".significance",
    "run_portfolio_pipeline": ".portfolio",
    "TeamStrengthPoissonModel": ".team_strength",
    "TeamStrengthConfig": ".team_strength",
    "ModelArtifact": ".artifacts",
    "ModelMetadata": ".artifacts",
    "ModelRegistry": ".registry",
    "get_registry": ".registry",
    "log_predictions": ".monitoring",
    "load_prediction_log": ".monitoring",
    "reconcile_outcomes": ".monitoring",
    "realized_metrics": ".monitoring",
    "population_stability_index": ".monitoring",
    "feature_drift": ".monitoring",
    "assess_overall_drift": ".monitoring",
    "monitoring_summary": ".monitoring",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports declared in __all__."""
    if name in _LAZY_ATTR_MODULES:
        module = import_module(_LAZY_ATTR_MODULES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
