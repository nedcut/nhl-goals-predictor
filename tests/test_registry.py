"""Tests for src/registry.py — the versioned model registry.

The registry handles model versioning and production promotion — logic that was
previously at 0% test coverage. These tests use lightweight artifacts (a tiny
picklable model + hand-built metadata) so registry behavior is exercised
directly without the cost of training a real model.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.artifacts import ModelArtifact, ModelMetadata
from src.registry import ModelRegistry, get_registry


def _make_artifact(mae: float, *, model_type: str = "XGBoost") -> ModelArtifact:
    """Build a minimal, saveable artifact with a controllable MAE."""
    meta = ModelMetadata(
        model_type=model_type,
        feature_names=["home_avg_GF", "away_avg_GF"],
        mae=mae,
        rmse=mae + 0.5,
        baseline_mae=2.0,
        improvement_pct=(1 - mae / 2.0) * 100,
        training_date=datetime.now().isoformat(),
        n_training_samples=100,
        n_test_samples=20,
    )
    # A plain dict is picklable, which is all joblib.dump needs for save/load.
    return ModelArtifact(model={"weights": [1, 2, 3]}, metadata=meta)


# ---------------------------------------------------------------------------
# Registration & versioning
# ---------------------------------------------------------------------------


def test_register_writes_artifact_and_entry(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    version = reg.register(_make_artifact(1.8), name="xgboost", description="first")

    assert version.startswith("xgboost_v1_")
    # Artifact files were written next to the registry.
    assert (tmp_path / f"{version}.joblib").exists()
    assert (tmp_path / f"{version}.json").exists()
    assert (tmp_path / "registry.json").exists()

    entries = reg.list_models()
    assert len(entries) == 1
    assert entries[0]["description"] == "first"
    assert entries[0]["mae"] == 1.8


def test_version_numbers_increment_per_name(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    v1 = reg.register(_make_artifact(1.9), name="xgboost")
    v2 = reg.register(_make_artifact(1.8), name="xgboost")
    rf1 = reg.register(_make_artifact(2.1), name="rf")

    assert "_v1_" in v1
    assert "_v2_" in v2
    assert "_v1_" in rf1  # independent counter per model name


def test_register_can_promote_immediately(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    version = reg.register(_make_artifact(1.7), promote_to_production=True)
    assert reg.get_production_version() == version


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_get_model_roundtrip_and_missing(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    version = reg.register(_make_artifact(1.8))

    loaded = reg.get_model(version)
    assert loaded is not None
    assert loaded.model == {"weights": [1, 2, 3]}
    assert loaded.metadata.mae == 1.8

    assert reg.get_model("does_not_exist") is None


def test_get_production_model_none_when_unset(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    reg.register(_make_artifact(1.8))  # registered but not promoted
    assert reg.get_production_model() is None


def test_get_production_model_loads_promoted(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    version = reg.register(_make_artifact(1.6), promote_to_production=True)
    prod = reg.get_production_model()
    assert prod is not None
    assert prod.metadata.mae == 1.6
    assert reg.get_production_version() == version


def test_get_production_model_handles_dangling_pointer(tmp_path):
    # Defensive path: production points to a version whose entry is gone
    # (e.g. a hand-edited / corrupted registry.json). Should warn and return
    # None rather than raise.
    reg = ModelRegistry(base_path=tmp_path)
    reg.register(_make_artifact(1.6), promote_to_production=True)
    reg.registry["models"] = []  # simulate the entry disappearing
    assert reg.get_production_model() is None


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_promote_to_production_switches_version(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    v1 = reg.register(_make_artifact(1.9))
    v2 = reg.register(_make_artifact(1.8))

    reg.promote_to_production(v1)
    assert reg.get_production_version() == v1
    reg.promote_to_production(v2)  # switch
    assert reg.get_production_version() == v2


def test_promote_unknown_version_raises(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    with pytest.raises(ValueError, match="not found"):
        reg.promote_to_production("ghost_v1_20250101_000000")


# ---------------------------------------------------------------------------
# Listing / querying
# ---------------------------------------------------------------------------


def test_list_models_sorted_newest_first_and_filtered(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    reg.register(_make_artifact(1.9), name="xgboost")
    reg.register(_make_artifact(1.8), name="xgboost")
    reg.register(_make_artifact(2.0), name="rf")

    all_models = reg.list_models()
    assert len(all_models) == 3
    # Newest first by registration timestamp.
    times = [m["registered_at"] for m in all_models]
    assert times == sorted(times, reverse=True)

    only_xgb = reg.list_models(name="xgboost")
    assert len(only_xgb) == 2
    assert all(m["name"] == "xgboost" for m in only_xgb)


def test_get_best_model_picks_lowest_mae(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    reg.register(_make_artifact(1.9), name="xgboost")
    reg.register(_make_artifact(1.7), name="xgboost")  # best
    reg.register(_make_artifact(1.5), name="rf")        # best overall, different name

    assert reg.get_best_model()["mae"] == 1.5
    assert reg.get_best_model(name="xgboost")["mae"] == 1.7
    assert reg.get_best_model(name="missing") is None


def test_get_best_model_none_when_empty(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    assert reg.get_best_model() is None


def test_summary_reports_counts_and_production(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    version = reg.register(_make_artifact(1.8), promote_to_production=True)
    text = reg.summary()
    assert "Total models: 1" in text
    assert version in text
    assert "[PROD]" in text


# ---------------------------------------------------------------------------
# Persistence & singleton
# ---------------------------------------------------------------------------


def test_registry_state_persists_across_instances(tmp_path):
    reg = ModelRegistry(base_path=tmp_path)
    version = reg.register(_make_artifact(1.8), promote_to_production=True)

    # A fresh registry over the same directory reloads the prior state.
    reopened = ModelRegistry(base_path=tmp_path)
    assert reopened.get_production_version() == version
    assert len(reopened.list_models()) == 1


def test_get_registry_caches_by_path(tmp_path):
    a = get_registry(base_path=tmp_path)
    b = get_registry(base_path=tmp_path)
    assert a is b  # same path -> cached singleton

    other = get_registry(base_path=tmp_path / "other")
    assert other is not a  # different path -> new instance
