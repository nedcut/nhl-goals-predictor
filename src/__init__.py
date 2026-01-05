"""Top-level package for the NHL total goals prediction project.

Main modules:
- data: Fetch game data from NHL API
- features: Feature engineering with rolling stats
- goalies: Goalie data and features
- model: Model training, evaluation, and optimization
- predict: CLI for making predictions
- api: REST API for predictions
- artifacts: Model persistence with metadata
- registry: Model versioning and management
- config: Centralized configuration
- validation: Input validation utilities
"""

from .config import config
from .data import build_dataset
from .features import add_features
from .model import (
    train_xgboost,
    train_random_forest,
    cross_validate,
    optimize_hyperparameters,
    save_model,
    load_model,
    TrainingResult,
    CVResult,
)
from .artifacts import ModelArtifact, ModelMetadata
from .registry import ModelRegistry, get_registry
from .validation import ValidationError, validate_game_data, validate_features

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
]