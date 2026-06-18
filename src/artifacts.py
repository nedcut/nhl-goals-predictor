"""
Model artifact management with metadata.

Provides classes for saving and loading trained models along with
comprehensive metadata about training configuration, performance, and data.

Usage:
    from src.artifacts import ModelArtifact, ModelMetadata

    # Create artifact from training result
    artifact = ModelArtifact.from_training_result(result, seasons=["20232024"])

    # Save to disk
    artifact.save("models/xgboost_v2")

    # Load later
    loaded = ModelArtifact.load("models/xgboost_v2")
    print(loaded.metadata.mae)
    predictions = loaded.model.predict(X)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import joblib

from .config import config
from .logging_config import get_logger

if TYPE_CHECKING:
    from .model import TrainingResult

logger = get_logger(__name__)


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


@dataclass
class ModelMetadata:
    """Metadata for a trained model.

    Contains all information needed to understand and reproduce a model's training.
    """

    # Model info
    model_type: str
    feature_names: List[str]

    # Performance metrics
    mae: float
    rmse: float
    baseline_mae: float
    improvement_pct: float

    # Training info
    training_date: str
    n_training_samples: int
    n_test_samples: int

    # Configuration
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Data info
    data_seasons: List[str] = field(default_factory=list)

    # Version control
    git_commit: Optional[str] = None

    @classmethod
    def from_training_result(
        cls,
        result: "TrainingResult",
        seasons: Optional[List[str]] = None,
        n_training_samples: Optional[int] = None,
    ) -> "ModelMetadata":
        """Create metadata from a TrainingResult.

        Parameters
        ----------
        result : TrainingResult
            Training result from model.py.
        seasons : list of str, optional
            Seasons used for training data.
        n_training_samples : int, optional
            Number of training samples (if not provided, estimated from test size).
        """
        improvement = (1 - result.mae / result.baseline_mae) * 100 if result.baseline_mae > 0 else 0.0

        # Estimate training samples if not provided
        if n_training_samples is None:
            test_size = config.model.test_size
            n_training_samples = int(len(result.y_test) * (1 - test_size) / test_size)

        # Capture config snapshot
        config_dict = {
            "features": {
                "rolling_window": config.features.rolling_window,
                "goalie_window": config.features.goalie_window,
                "min_games": config.features.min_games,
            },
            "model": {
                "test_size": config.model.test_size,
                "cv_folds": config.model.cv_folds,
                "xgb_params": config.model.xgb_params,
            },
        }

        return cls(
            model_type=result.model_type,
            feature_names=result.feature_names,
            mae=result.mae,
            rmse=result.rmse,
            baseline_mae=result.baseline_mae,
            improvement_pct=improvement,
            training_date=datetime.now().isoformat(),
            n_training_samples=n_training_samples,
            n_test_samples=len(result.y_test),
            config_snapshot=config_dict,
            data_seasons=seasons or [],
            git_commit=_get_git_commit(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


class ModelArtifact:
    """Complete model artifact with model and metadata.

    Provides save/load functionality for trained models along with
    all associated metadata.
    """

    def __init__(self, model: Any, metadata: ModelMetadata):
        """Initialize artifact.

        Parameters
        ----------
        model : Any
            Trained model (XGBRegressor, RandomForestRegressor, etc.).
        metadata : ModelMetadata
            Model metadata.
        """
        self.model = model
        self.metadata = metadata

    @classmethod
    def from_training_result(
        cls,
        result: "TrainingResult",
        seasons: Optional[List[str]] = None,
        n_training_samples: Optional[int] = None,
    ) -> "ModelArtifact":
        """Create artifact from a TrainingResult.

        Parameters
        ----------
        result : TrainingResult
            Training result from model.py.
        seasons : list of str, optional
            Seasons used for training data.
        n_training_samples : int, optional
            Number of training samples.

        Returns
        -------
        ModelArtifact
            Complete artifact ready for saving.
        """
        metadata = ModelMetadata.from_training_result(
            result,
            seasons=seasons,
            n_training_samples=n_training_samples,
        )
        return cls(model=result.model, metadata=metadata)

    def save(self, path: Path | str) -> None:
        """Save artifact to disk.

        Creates two files:
        - {path}.joblib: The trained model
        - {path}.json: The metadata

        Parameters
        ----------
        path : Path or str
            Base path for the artifact (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path.with_suffix(".joblib")
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info("Saved model artifact to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "ModelArtifact":
        """Load artifact from disk.

        Parameters
        ----------
        path : Path or str
            Base path for the artifact (without extension).

        Returns
        -------
        ModelArtifact
            Loaded artifact with model and metadata.
        """
        path = Path(path)

        # Load model
        model_path = path.with_suffix(".joblib")
        # Load metadata
        metadata_path = path.with_suffix(".json")
        model = joblib.load(model_path)

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
        else:
            if isinstance(model, dict) and "model" in model:
                model = model["model"]
            metadata = cls._metadata_from_legacy(model)

        logger.info("Loaded model artifact from %s", path)
        return cls(model=model, metadata=metadata)

    @staticmethod
    def _metadata_from_legacy(model_obj: Any) -> ModelMetadata:
        """Build minimal metadata for legacy model files without JSON metadata."""
        feature_names: List[str] = []
        model_type = "Unknown"
        model = model_obj

        if isinstance(model_obj, dict) and "model" in model_obj:
            model = model_obj["model"]
            model_type = model_obj.get("model_type", type(model).__name__)
            feature_names = model_obj.get("feature_names", [])
        else:
            model_type = type(model_obj).__name__
            if hasattr(model_obj, "feature_names_in_"):
                feature_names = list(model_obj.feature_names_in_)

        if not feature_names:
            raise ValueError(
                "Legacy model is missing feature names; retrain and save a new artifact."
            )

        return ModelMetadata(
            model_type=model_type,
            feature_names=feature_names,
            mae=float("nan"),
            rmse=float("nan"),
            baseline_mae=float("nan"),
            improvement_pct=float("nan"),
            training_date="unknown",
            n_training_samples=0,
            n_test_samples=0,
            config_snapshot={},
            data_seasons=[],
            git_commit=_get_git_commit(),
        )

    def predict(self, X) -> Any:
        """Make predictions using the model.

        Parameters
        ----------
        X : array-like
            Features for prediction.

        Returns
        -------
        array-like
            Predictions.
        """
        # Backward compatibility: older Poisson/Ensemble artifacts stored a scaler
        # on the estimator instance instead of using an sklearn Pipeline.
        if hasattr(self.model, "_scaler"):
            scaler = getattr(self.model, "_scaler")
            return self.model.predict(scaler.transform(X))
        return self.model.predict(X)

    def summary(self) -> str:
        """Get a human-readable summary of the artifact."""
        lines = [
            f"Model: {self.metadata.model_type}",
            f"Trained: {self.metadata.training_date}",
            f"MAE: {self.metadata.mae:.4f} (baseline: {self.metadata.baseline_mae:.4f})",
            f"Improvement: {self.metadata.improvement_pct:+.2f}%",
            f"Features: {len(self.metadata.feature_names)}",
            f"Training samples: {self.metadata.n_training_samples}",
            f"Test samples: {self.metadata.n_test_samples}",
        ]
        if self.metadata.data_seasons:
            lines.append(f"Seasons: {', '.join(self.metadata.data_seasons)}")
        if self.metadata.git_commit:
            lines.append(f"Git commit: {self.metadata.git_commit[:8]}")
        return "\n".join(lines)
