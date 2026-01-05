"""
Model registry for versioned model storage.

Provides a way to register, track, and manage multiple model versions.
Supports promoting models to production and listing model history.

Usage:
    from src.registry import ModelRegistry, get_registry

    registry = get_registry()

    # Register a new model
    version = registry.register(artifact, name="xgboost")

    # Get production model
    prod_model = registry.get_production_model()

    # List all models
    for model in registry.list_models():
        print(model["version"], model["mae"])
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import ModelArtifact
from .logging_config import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manage versioned model storage.

    Provides registration, listing, and production model management.
    """

    def __init__(self, base_path: Path = Path("models")):
        """Initialize the registry.

        Parameters
        ----------
        base_path : Path
            Base directory for model storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_path / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": [],
                "production": None,
                "created_at": datetime.now().isoformat(),
            }

    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry["updated_at"] = datetime.now().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register(
        self,
        artifact: ModelArtifact,
        name: str = "xgboost",
        description: str = "",
        promote_to_production: bool = False,
    ) -> str:
        """Register a new model version.

        Parameters
        ----------
        artifact : ModelArtifact
            The model artifact to register.
        name : str
            Base name for the model (e.g., "xgboost", "rf").
        description : str
            Optional description of this version.
        promote_to_production : bool
            If True, immediately promote this model to production.

        Returns
        -------
        str
            The version string for the registered model.
        """
        # Generate version string
        version_num = len([m for m in self.registry["models"] if m["name"] == name]) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{name}_v{version_num}_{timestamp}"

        # Save artifact
        model_path = self.base_path / version
        artifact.save(model_path)

        # Create registry entry
        entry = {
            "version": version,
            "name": name,
            "path": str(model_path),
            "registered_at": datetime.now().isoformat(),
            "description": description,
            "model_type": artifact.metadata.model_type,
            "mae": artifact.metadata.mae,
            "rmse": artifact.metadata.rmse,
            "baseline_mae": artifact.metadata.baseline_mae,
            "improvement_pct": artifact.metadata.improvement_pct,
            "n_features": len(artifact.metadata.feature_names),
            "data_seasons": artifact.metadata.data_seasons,
            "git_commit": artifact.metadata.git_commit,
        }

        self.registry["models"].append(entry)

        if promote_to_production:
            self.registry["production"] = version
            logger.info("Promoted %s to production", version)

        self._save_registry()
        logger.info("Registered model: %s", version)

        return version

    def get_production_model(self) -> Optional[ModelArtifact]:
        """Load the current production model.

        Returns
        -------
        ModelArtifact or None
            The production model, or None if no production model is set.
        """
        if self.registry["production"] is None:
            logger.warning("No production model set")
            return None

        version = self.registry["production"]
        for entry in self.registry["models"]:
            if entry["version"] == version:
                return ModelArtifact.load(Path(entry["path"]))

        logger.error("Production model %s not found in registry", version)
        return None

    def get_model(self, version: str) -> Optional[ModelArtifact]:
        """Load a specific model version.

        Parameters
        ----------
        version : str
            The version string to load.

        Returns
        -------
        ModelArtifact or None
            The model artifact, or None if not found.
        """
        for entry in self.registry["models"]:
            if entry["version"] == version:
                return ModelArtifact.load(Path(entry["path"]))

        logger.warning("Model version %s not found", version)
        return None

    def promote_to_production(self, version: str) -> None:
        """Promote a model version to production.

        Parameters
        ----------
        version : str
            The version string to promote.

        Raises
        ------
        ValueError
            If the version is not found in the registry.
        """
        if not any(e["version"] == version for e in self.registry["models"]):
            raise ValueError(f"Version {version} not found in registry")

        old_production = self.registry["production"]
        self.registry["production"] = version
        self._save_registry()

        if old_production:
            logger.info("Changed production from %s to %s", old_production, version)
        else:
            logger.info("Set production model to %s", version)

    def list_models(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models.

        Parameters
        ----------
        name : str, optional
            If provided, only list models with this base name.

        Returns
        -------
        list of dict
            List of model entries, sorted by registration date (newest first).
        """
        models = self.registry["models"]

        if name:
            models = [m for m in models if m["name"] == name]

        # Sort by registration date (newest first)
        return sorted(models, key=lambda m: m["registered_at"], reverse=True)

    def get_best_model(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the model with the best (lowest) MAE.

        Parameters
        ----------
        name : str, optional
            If provided, only consider models with this base name.

        Returns
        -------
        dict or None
            The best model entry, or None if no models exist.
        """
        models = self.list_models(name=name)

        if not models:
            return None

        return min(models, key=lambda m: m["mae"])

    def get_production_version(self) -> Optional[str]:
        """Get the current production model version.

        Returns
        -------
        str or None
            The production version string, or None if not set.
        """
        return self.registry["production"]

    def summary(self) -> str:
        """Get a human-readable summary of the registry.

        Returns
        -------
        str
            Summary of registered models.
        """
        lines = [
            f"Model Registry: {self.base_path}",
            f"Total models: {len(self.registry['models'])}",
            f"Production: {self.registry['production'] or 'Not set'}",
            "",
        ]

        if self.registry["models"]:
            lines.append("Recent models:")
            for entry in self.list_models()[:5]:
                prod_marker = " [PROD]" if entry["version"] == self.registry["production"] else ""
                lines.append(
                    f"  {entry['version']}: MAE={entry['mae']:.4f} "
                    f"({entry['improvement_pct']:+.2f}%){prod_marker}"
                )

        return "\n".join(lines)


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry(base_path: Path = Path("models")) -> ModelRegistry:
    """Get the global model registry instance.

    Parameters
    ----------
    base_path : Path
        Base directory for model storage.

    Returns
    -------
    ModelRegistry
        The global registry instance.
    """
    global _registry
    if _registry is None or _registry.base_path != base_path:
        _registry = ModelRegistry(base_path)
    return _registry
