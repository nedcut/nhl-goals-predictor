"""Tests for multi-threshold calibration reporting in time_series_cv_forecast."""

from __future__ import annotations

import pytest


def test_multi_threshold_metrics_populated(sample_features_df):
    from src.evaluation import time_series_cv_forecast

    result = time_series_cv_forecast(
        sample_features_df,
        point_model="team_strength",
        dist_model="poisson",
        threshold=6.5,
        thresholds=[5.5, 6.5, 7.5],
        n_splits=3,
    )

    assert result.threshold_metrics is not None
    assert set(result.threshold_metrics.keys()) == {"5.5", "6.5", "7.5"}
    for metrics in result.threshold_metrics.values():
        assert set(metrics.keys()) == {"brier", "log_loss"}
        assert 0.0 <= metrics["brier"] <= 1.0
        assert metrics["log_loss"] >= 0.0

    assert result.reliability_by_threshold is not None
    assert set(result.reliability_by_threshold.keys()) == {"5.5", "6.5", "7.5"}
    for bins in result.reliability_by_threshold.values():
        assert len(bins) >= 1
        assert sum(b.count for b in bins) > 0


def test_primary_over_brier_matches_threshold_entry(sample_features_df):
    from src.evaluation import time_series_cv_forecast

    result = time_series_cv_forecast(
        sample_features_df,
        point_model="team_strength",
        dist_model="poisson",
        threshold=6.5,
        thresholds=[5.5, 6.5, 7.5],
        n_splits=3,
    )

    assert result.threshold_metrics is not None
    primary = result.threshold_metrics["6.5"]
    assert result.metrics_mean["over_brier"] == pytest.approx(primary["brier"], rel=1e-12)
    assert result.metrics_mean["over_log_loss"] == pytest.approx(primary["log_loss"], rel=1e-12)


def test_backcompat_without_thresholds_kwarg(sample_features_df):
    from src.evaluation import time_series_cv_forecast

    result = time_series_cv_forecast(
        sample_features_df,
        point_model="team_strength",
        dist_model="poisson",
        threshold=6.5,
        n_splits=3,
    )

    # Primary fold metrics still present
    assert "over_brier" in result.metrics_mean
    assert len(result.folds) == 3
    assert result.per_game is not None
    assert "over_brier" in result.per_game

    # Default: only the primary threshold is reported under threshold_metrics
    assert result.threshold_metrics is not None
    assert set(result.threshold_metrics.keys()) == {"6.5"}
    assert result.metrics_mean["over_brier"] == pytest.approx(
        result.threshold_metrics["6.5"]["brier"],
        rel=1e-12,
    )
    assert result.reliability_by_threshold is not None
    assert set(result.reliability_by_threshold.keys()) == {"6.5"}
