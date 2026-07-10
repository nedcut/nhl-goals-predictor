"""Tests for paired-bootstrap significance and per-game scoring integration."""

from __future__ import annotations

import numpy as np
import pytest

from src.champion import (
    compare_models_significance,
    per_game_weighted_scores,
    weighted_score,
)
from src.evaluation import time_series_cv_forecast
from src.features import add_features
from src.model import FEATURE_COLUMNS_ATTR, get_feature_columns
from src.significance import holm_adjusted_p_values, paired_bootstrap


def test_paired_bootstrap_detects_clear_winner():
    rng = np.random.default_rng(0)
    # A is consistently lower (better) than B by ~0.5 per game.
    b = rng.uniform(1.0, 2.0, size=400)
    a = b - 0.5
    result = paired_bootstrap(a, b, name_a="A", name_b="B")
    assert result.significant
    assert result.better_model == "A"
    assert result.mean_diff < 0
    assert result.ci_high < 0  # entire CI below zero


def test_paired_bootstrap_calls_noise_insignificant():
    rng = np.random.default_rng(1)
    base = rng.uniform(1.0, 2.0, size=400)
    # Two models with identical scores plus tiny symmetric jitter => no real edge.
    a = base + rng.normal(0, 1e-6, size=400)
    b = base + rng.normal(0, 1e-6, size=400)
    result = paired_bootstrap(a, b, name_a="A", name_b="B")
    assert not result.significant
    assert result.better_model is None


def test_paired_bootstrap_requires_aligned_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_paired_bootstrap_resamples_whole_blocks():
    a = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    b = a + 0.2
    groups = ["w1", "w1", "w2", "w2", "w3", "w3"]
    result = paired_bootstrap(a, b, groups=groups, n_boot=500)
    assert result.resampling_unit == "block"
    assert result.n_blocks == 3
    assert result.n_games == 6


def test_holm_adjustment_is_monotone_and_restores_original_order():
    adjusted = holm_adjusted_p_values([0.03, 0.001, 0.02, 0.8])
    assert adjusted == pytest.approx([0.06, 0.004, 0.06, 0.8])


def test_per_game_weighted_scores_reconstruct_aggregate():
    # Mean of per-game weighted scores must equal the aggregate weighted_score.
    n = 50
    rng = np.random.default_rng(7)
    per_game = {
        "game_key": np.array([f"g{i}" for i in range(n)], dtype=object),
        "abs_error": rng.uniform(0.5, 3.0, n),
        "crps": rng.uniform(0.8, 2.0, n),
        "dist_nll": rng.uniform(1.8, 2.8, n),
        "over_brier": rng.uniform(0.1, 0.4, n),
    }
    baseline = {"mae": 1.9, "crps": 1.3, "dist_nll": 2.25, "over_brier": 0.25}
    agg = {
        "mae": float(per_game["abs_error"].mean()),
        "crps": float(per_game["crps"].mean()),
        "dist_nll": float(per_game["dist_nll"].mean()),
        "over_brier": float(per_game["over_brier"].mean()),
    }
    scores = per_game_weighted_scores(per_game, baseline)
    assert np.mean(list(scores.values())) == pytest.approx(weighted_score(agg, baseline), rel=1e-12)


def test_compare_models_returns_none_without_shared_games():
    pg_a = {
        "game_key": np.array(["x"], dtype=object),
        "abs_error": np.array([1.0]),
        "crps": np.array([1.0]),
        "dist_nll": np.array([2.0]),
        "over_brier": np.array([0.2]),
    }
    pg_b = {
        "game_key": np.array(["y"], dtype=object),
        "abs_error": np.array([1.0]),
        "crps": np.array([1.0]),
        "dist_nll": np.array([2.0]),
        "over_brier": np.array([0.2]),
    }
    baseline = {"mae": 1.9, "crps": 1.3, "dist_nll": 2.25, "over_brier": 0.25}
    assert compare_models_significance(pg_a, pg_b, baseline, name_a="a", name_b="b") is None


def test_cv_forecast_exposes_per_game_and_std(sample_game_data):
    df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
    result = time_series_cv_forecast(df, point_model="xgb", dist_model="nb2", n_splits=3)

    # Per-game arrays present and internally consistent in length.
    pg = result.per_game
    assert pg is not None
    n = len(pg["game_key"])
    assert n > 0
    for key in ("abs_error", "crps", "dist_nll", "over_brier"):
        assert len(pg[key]) == n

    # metrics_std exposes across-fold spread for the headline metrics.
    std = result.metrics_std
    assert set(["mae", "crps", "dist_nll", "over_brier"]).issubset(std)
    assert all(v >= 0 for v in std.values())


def test_feature_registry_is_stamped_and_used(sample_game_data):
    df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
    assert FEATURE_COLUMNS_ATTR in df.attrs
    registered = df.attrs[FEATURE_COLUMNS_ATTR]
    assert registered, "registry should be non-empty"
    # The target must never be considered a feature.
    assert "totalGoals" not in registered
    # Registry path and detection path agree on this freshly built frame.
    assert get_feature_columns(df) == registered
