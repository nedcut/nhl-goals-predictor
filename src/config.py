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
from typing import Dict, List, Tuple


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
    xg_cache_dir: Path = field(default_factory=lambda: Path("data/xg"))
    active_season_cache_ttl_hours: int = 6
    xg_url_template: str = (
        "https://moneypuck.com/moneypuck/playerData/games/{season}/regular/teams.csv"
    )

    # Request throttling
    request_delay: float = 0.2  # Delay between game data requests
    goalie_request_delay: float = 0.05  # Delay between goalie boxscore requests
    xg_request_delay: float = 0.2
    xg_request_timeout: int = 30

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

    # Rolling window sizes (multiple windows capture different signals)
    rolling_windows: Tuple[int, ...] = (5, 10, 20, 40)  # Short, medium, long-term
    rolling_window: int = 20  # Default/primary window for backwards compatibility
    goalie_window: int = 10  # Window for goalie rolling stats

    # Minimum history requirements
    min_games: int = 3  # Minimum games before features are valid

    # Feature flags
    include_goalies: bool = True
    include_xg: bool = False
    include_multi_window: bool = True  # Use multiple rolling windows
    include_interactions: bool = True  # Add interaction features
    include_temporal: bool = True  # Add month/seasonality features
    xg_windows: Tuple[int, ...] = (5, 10, 20)

    # Season reference date (October 1st typically)
    season_start_month: int = 10
    season_start_day: int = 1


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    # Train/test split
    test_size: float = 0.2

    # Cross-validation
    cv_folds: int = 5

    # Random state for reproducibility
    random_state: int = 42

    # Current champion XGBoost hyperparameters under probabilistic time-series CV.
    xgb_params: Dict[str, float | int] = field(default_factory=lambda: {
        "max_depth": 4,
        "learning_rate": 0.011896873680695898,
        "n_estimators": 65,
        "reg_alpha": 1.7530910973690677,
        "reg_lambda": 2.6740596452335668,
        "subsample": 0.8393574607886646,
        "colsample_bytree": 0.5556469177410432,
        "min_child_weight": 7,
    })

    # Random Forest defaults
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None


@dataclass
class MonitoringConfig:
    """Configuration for prediction logging and drift detection."""

    # Append-only store of served predictions (JSON Lines). Lives under data/
    # (gitignored) since it is a runtime artifact, not source.
    log_path: Path = field(
        default_factory=lambda: Path("data/monitoring/predictions_log.jsonl")
    )

    # Population Stability Index (PSI) binning + interpretation thresholds.
    # The 0.1 / 0.25 cutoffs are the conventional credit-scoring rule of thumb.
    drift_bins: int = 10
    psi_moderate: float = 0.10  # >= this: moderate shift, worth investigating
    psi_significant: float = 0.25  # >= this: significant shift, action needed

    # How many of the most recent reconciled games to score for "recent" MAE.
    recent_window_games: int = 200

    # Over/under thresholds to score with the Brier score during reconciliation.
    brier_thresholds: Tuple[float, ...] = (5.5, 6.5, 7.5)


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


# Global configuration instance
config = Config()
