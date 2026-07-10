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

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .config import config
from .logging_config import get_logger

if TYPE_CHECKING:
    from .model import TrainingResult

logger = get_logger(__name__)

ARTIFACT_SCHEMA_VERSION = 2


def feature_schema_hash(feature_names: List[str]) -> str:
    """Stable digest of the ordered serving feature schema."""
    payload = json.dumps(list(feature_names), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Plausible range for an XGBoost total-goals regressor's learned intercept
# (base_score ≈ mean total goals per game, ~6). A value outside this range means
# the intercept was lost — the known failure mode when a pickled model crosses
# an XGBoost version boundary and base_score silently resets to 0.5.
_XGB_SANE_BASE_SCORE = (2.0, 12.0)


def _is_xgboost_model(model: Any) -> bool:
    """True if the model is an XGBoost sklearn-wrapper estimator."""
    return HAS_XGBOOST and isinstance(model, xgb.XGBModel)


def _parse_base_score(raw: Any) -> List[float]:
    """Parse save_config()'s base_score, scalar ("5E-1") or vector ("[5.8E0]").

    XGBoost >= 3 serializes base_score as a JSON list to support multi-target
    models; earlier versions emit a bare number string.
    """
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        parsed = float(str(raw).strip("[]"))
    values = parsed if isinstance(parsed, list) else [parsed]
    return [float(v) for v in values]


def _check_xgb_base_score(model: Any, path: Path) -> None:
    """Reject XGBoost regressors whose learned intercept is implausible."""
    if not isinstance(model, xgb.XGBRegressor):
        return
    learner_param = json.loads(model.get_booster().save_config())["learner"][
        "learner_model_param"
    ]
    scores = _parse_base_score(learner_param["base_score"])
    lo, hi = _XGB_SANE_BASE_SCORE
    if not scores or not all(lo <= s <= hi for s in scores):
        raise ValueError(
            f"Model at {path} has base_score={learner_param['base_score']}, outside "
            f"the plausible total-goals range [{lo}, {hi}]. The intercept was likely "
            "lost by unpickling across an XGBoost version boundary; retrain and "
            "re-save the artifact."
        )


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

    # Release-grade provenance and serving contract. Defaults preserve loading
    # of schema-v1 artifacts, but validate_for_serving rejects incomplete ones.
    schema_version: int = 1
    benchmark_release: Optional[str] = None
    data_fingerprint: Optional[str] = None
    feature_schema_hash: Optional[str] = None
    prediction_interface: str = "feature_matrix"
    training_cohort: Dict[str, Any] = field(default_factory=dict)
    holdout_metrics: Dict[str, float] = field(default_factory=dict)
    monitoring_reference: Dict[str, Any] = field(default_factory=dict)
    artifact_id: Optional[str] = None

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
        improvement = (
            (1 - result.mae / result.baseline_mae) * 100 if result.baseline_mae > 0 else 0.0
        )

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
            schema_version=ARTIFACT_SCHEMA_VERSION,
            feature_schema_hash=feature_schema_hash(result.feature_names),
            prediction_interface="feature_matrix",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(**data)

    def serving_validation_errors(self) -> List[str]:
        """Return release-contract violations that make serving unsafe."""
        errors: List[str] = []
        if self.schema_version < ARTIFACT_SCHEMA_VERSION:
            errors.append(
                f"artifact schema {self.schema_version} is older than required "
                f"{ARTIFACT_SCHEMA_VERSION}"
            )
        if not self.benchmark_release:
            errors.append("benchmark_release is missing")
        if not self.data_fingerprint:
            errors.append("data_fingerprint is missing")
        if not self.git_commit:
            errors.append("git_commit is missing")
        if self.training_date == "unknown":
            errors.append("training_date is unknown")
        if not self.data_seasons:
            errors.append("data_seasons is empty")
        for name in ("mae", "rmse", "baseline_mae"):
            if not math.isfinite(float(getattr(self, name))):
                errors.append(f"{name} is not finite")
        if self.prediction_interface not in {"feature_matrix", "game_frame"}:
            errors.append(f"unknown prediction_interface={self.prediction_interface!r}")
        expected_hash = feature_schema_hash(self.feature_names)
        if self.feature_schema_hash != expected_hash:
            errors.append("feature_schema_hash does not match ordered feature_names")
        if self.prediction_interface == "feature_matrix" and not self.feature_names:
            errors.append("feature_matrix artifact has no feature_names")
        return errors


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

        For XGBoost models a third file, {path}.ubj, is written in XGBoost's
        native format. The pickle in .joblib is not stable across XGBoost
        versions (the learned base_score can be silently lost), so load()
        prefers the native file when present.

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
        if _is_xgboost_model(self.model):
            self.model.save_model(path.with_suffix(".ubj"))

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

        # Load model — prefer the version-stable native XGBoost file over the
        # pickle (see save()).
        native_path = path.with_suffix(".ubj")
        metadata_path = path.with_suffix(".json")
        if HAS_XGBOOST and native_path.exists():
            model: Any = xgb.XGBRegressor()
            model.load_model(native_path)
        else:
            model = joblib.load(path.with_suffix(".joblib"))

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
        else:
            if isinstance(model, dict) and "model" in model:
                model = model["model"]
            metadata = cls._metadata_from_legacy(model)

        if _is_xgboost_model(model):
            _check_xgb_base_score(model, path)

        logger.info("Loaded model artifact from %s", path)
        return cls(model=model, metadata=metadata)

    def validate_for_serving(self) -> None:
        """Raise when the artifact lacks the release provenance required by the API."""
        errors = self.metadata.serving_validation_errors()
        if errors:
            raise ValueError("Artifact is not release-grade: " + "; ".join(errors))

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
            schema_version=1,
            feature_schema_hash=feature_schema_hash(feature_names),
            prediction_interface="feature_matrix",
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
        if self.metadata.prediction_interface == "game_frame":
            if not hasattr(self.model, "predict_mu"):
                raise ValueError("game_frame artifact model does not implement predict_mu")
            return self.model.predict_mu(X)

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
