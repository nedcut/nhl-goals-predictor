"""
Centralized configuration for NHL goals prediction pipeline.

This module contains all configurable parameters used across the codebase.
Import and use `config` to access settings.

Usage:
    from src.config import config

    # Access data settings
    print(config.data.api_base)
    print(config.data.cache_dir)

    # Access feature settings
    print(config.features.rolling_window)

    # Access model settings
    print(config.model.xgb_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class DataConfig:
    """Configuration for data fetching and caching."""

    # API settings
    api_base: str = "https://api-web.nhle.com/v1"
    request_timeout: int = 30

    # Cache directories
    cache_dir: Path = field(default_factory=lambda: Path("data/raw"))
    goalie_cache_dir: Path = field(default_factory=lambda: Path("data/goalies"))
    goalie_cache_file: str = "goalie_stats.csv"

    # Request throttling
    request_delay: float = 0.2  # Delay between game data requests
    goalie_request_delay: float = 0.05  # Delay between goalie boxscore requests

    # Season date ranges
    season_start_month: int = 10  # October
    season_end_month: int = 6  # June

    @property
    def goalie_cache_path(self) -> Path:
        """Full path to goalie cache file."""
        return self.goalie_cache_dir / self.goalie_cache_file


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Rolling window sizes
    rolling_window: int = 20  # Window for team rolling stats (20 is optimal)
    goalie_window: int = 10  # Window for goalie rolling stats

    # Minimum history requirements
    min_games: int = 3  # Minimum games before features are valid

    # Feature flags
    include_goalies: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    # Train/test split
    test_size: float = 0.2

    # Cross-validation
    cv_folds: int = 5

    # Random state for reproducibility
    random_state: int = 42

    # Optimized XGBoost hyperparameters (beats baseline by ~0.6%)
    xgb_params: Dict[str, float | int] = field(default_factory=lambda: {
        "max_depth": 2,
        "learning_rate": 0.01,
        "n_estimators": 150,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 7,
    })

    # Random Forest defaults
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Global configuration instance
config = Config()
