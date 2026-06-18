"""Tests for src/monitoring.py — logging, reconciliation, and drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import monitoring


# ---------------------------------------------------------------------------
# Prediction log roundtrip
# ---------------------------------------------------------------------------


def test_log_and_load_roundtrip(tmp_path):
    log_path = tmp_path / "preds.jsonl"
    preds = pd.DataFrame(
        {
            "gamePk": [1, 2],
            "date": ["2025-01-01", "2025-01-01"],
            "homeTeam": ["Boston Bruins", "Toronto Maple Leafs"],
            "awayTeam": ["Montreal Canadiens", "Ottawa Senators"],
            "predicted_total_goals": [6.1, 5.4],
        }
    )

    written = monitoring.log_predictions(preds, path=log_path, model_version="v1")
    assert written == 2

    # Appends rather than overwrites.
    monitoring.log_predictions(
        preds.head(1), path=log_path, model_version="v1"
    )

    loaded = monitoring.load_prediction_log(log_path)
    assert len(loaded) == 3
    assert set(loaded.columns) >= {
        "gamePk", "predicted_total_goals", "model_version", "logged_at"
    }
    assert loaded["model_version"].unique().tolist() == ["v1"]


def test_log_requires_prediction_column(tmp_path):
    with pytest.raises(ValueError, match="predicted_total_goals"):
        monitoring.log_predictions(pd.DataFrame({"gamePk": [1]}), path=tmp_path / "x.jsonl")


def test_load_missing_log_returns_empty(tmp_path):
    assert monitoring.load_prediction_log(tmp_path / "nope.jsonl").empty


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_on_gamepk():
    log = pd.DataFrame(
        {"gamePk": [1, 2, 3], "predicted_total_goals": [6.0, 5.0, 7.0]}
    )
    results = pd.DataFrame(
        {"gamePk": [1, 2], "totalGoals": [4, 8]}  # game 3 not yet played
    )
    out = monitoring.reconcile_outcomes(log, results)
    assert out.loc[out["gamePk"] == 1, "actual_total_goals"].iloc[0] == 4
    assert out.loc[out["gamePk"] == 2, "actual_total_goals"].iloc[0] == 8
    assert pd.isna(out.loc[out["gamePk"] == 3, "actual_total_goals"].iloc[0])


def test_reconcile_falls_back_to_date_team_triple():
    log = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "homeTeam": ["Boston Bruins"],
            "awayTeam": ["Montreal Canadiens"],
            "predicted_total_goals": [6.0],
        }
    )
    results = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "homeTeam": ["Boston Bruins"],
            "awayTeam": ["Montreal Canadiens"],
            "totalGoals": [5],
        }
    )
    out = monitoring.reconcile_outcomes(log, results)
    assert out["actual_total_goals"].iloc[0] == 5


# ---------------------------------------------------------------------------
# Realized metrics
# ---------------------------------------------------------------------------


def test_realized_metrics_known_values():
    reconciled = pd.DataFrame(
        {
            "predicted_total_goals": [6.0, 5.0, 7.0, 4.0],
            "actual_total_goals": [5.0, 5.0, 6.0, 6.0],
        }
    )
    m = monitoring.realized_metrics(reconciled, thresholds=[5.5])
    # errors: +1, 0, +1, -2 -> |.| mean = 1.0; bias mean = 0.0
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(0.0)
    # direction at 5.5: pred>5.5 = [T,F,T,F], actual>5.5 = [F,F,T,T] -> matches [F,T,T,F] = 0.5
    assert m["thresholds"]["5.5"]["point_accuracy"] == pytest.approx(0.5)


def test_realized_metrics_brier_uses_logged_prob():
    reconciled = pd.DataFrame(
        {
            "predicted_total_goals": [6.0, 6.0],
            "actual_total_goals": [7.0, 4.0],  # over, under at 6.5
            "prob_over": [0.8, 0.3],
            "threshold": [6.5, 6.5],
        }
    )
    m = monitoring.realized_metrics(reconciled, thresholds=[6.5])
    # outcomes over 6.5: [1, 0]; brier = mean((0.8-1)^2, (0.3-0)^2) = mean(0.04, 0.09) = 0.065
    assert m["thresholds"]["6.5"]["brier"] == pytest.approx(0.065)


def test_realized_metrics_empty_when_no_outcomes():
    reconciled = pd.DataFrame(
        {"predicted_total_goals": [6.0], "actual_total_goals": [np.nan]}
    )
    assert monitoring.realized_metrics(reconciled)["n"] == 0


# ---------------------------------------------------------------------------
# PSI / drift
# ---------------------------------------------------------------------------


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    sample = rng.normal(6, 1.5, size=2000)
    psi = monitoring.population_stability_index(sample, sample.copy())
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(6, 1.5, size=4000)
    small = rng.normal(6.2, 1.5, size=4000)
    large = rng.normal(9.0, 1.5, size=4000)
    psi_small = monitoring.population_stability_index(reference, small)
    psi_large = monitoring.population_stability_index(reference, large)
    assert 0 <= psi_small < psi_large
    assert monitoring.drift_status(psi_large) == "significant"


def test_psi_nan_on_empty_sample():
    assert np.isnan(monitoring.population_stability_index([], [1, 2, 3]))
    assert np.isnan(monitoring.population_stability_index([1, 2, 3], []))


def test_drift_status_thresholds():
    assert monitoring.drift_status(0.05) == "stable"
    assert monitoring.drift_status(0.15) == "moderate"
    assert monitoring.drift_status(0.30) == "significant"
    assert monitoring.drift_status(float("nan")) == "unknown"


def test_feature_drift_selects_numeric_and_ranks():
    rng = np.random.default_rng(2)
    reference = pd.DataFrame(
        {
            "home_avg_GF": rng.normal(3, 0.5, 1000),
            "away_avg_GA": rng.normal(3, 0.5, 1000),
            "homeTeam": ["X"] * 1000,  # non-numeric, must be ignored
            "gamePk": np.arange(1000),  # identity, must be ignored
        }
    )
    recent = pd.DataFrame(
        {
            "home_avg_GF": rng.normal(4, 0.5, 500),  # shifted -> higher PSI
            "away_avg_GA": rng.normal(3, 0.5, 500),  # stable
            "homeTeam": ["X"] * 500,
            "gamePk": np.arange(500),
        }
    )
    drift = monitoring.feature_drift(reference, recent)
    assert set(drift["feature"]) == {"home_avg_GF", "away_avg_GA"}
    # The shifted feature should rank first (sorted by PSI desc).
    assert drift.iloc[0]["feature"] == "home_avg_GF"
    assert drift.iloc[0]["psi"] > drift.iloc[1]["psi"]


def test_assess_overall_drift_any_significant_alerts():
    # A single significant feature flags the whole model (high-severity tier is
    # maximally sensitive).
    drift = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "psi": [0.02, 0.05, 0.40],
            "status": ["stable", "stable", "significant"],
        }
    )
    verdict = monitoring.assess_overall_drift(drift)
    assert verdict["status"] == "significant"
    assert verdict["max_psi"] == pytest.approx(0.40)
    assert verdict["drifted_features"] == ["c"]


def test_assess_overall_drift_single_moderate_stays_stable():
    # One feature in the noisy 0.10-0.25 band is not enough to alert.
    drift = pd.DataFrame(
        {"feature": ["a", "b"], "psi": [0.02, 0.18], "status": ["stable", "moderate"]}
    )
    verdict = monitoring.assess_overall_drift(drift)
    assert verdict["status"] == "stable"
    assert verdict["drifted_features"] == ["b"]  # still reported for triage


def test_assess_overall_drift_two_moderate_escalates():
    drift = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "psi": [0.02, 0.15, 0.20],
            "status": ["stable", "moderate", "moderate"],
        }
    )
    verdict = monitoring.assess_overall_drift(drift)
    assert verdict["status"] == "moderate"
    assert set(verdict["drifted_features"]) == {"b", "c"}


def test_assess_overall_drift_stable_when_all_stable():
    drift = pd.DataFrame(
        {"feature": ["a", "b"], "psi": [0.01, 0.05], "status": ["stable", "stable"]}
    )
    verdict = monitoring.assess_overall_drift(drift)
    assert verdict["status"] == "stable"
    assert verdict["drifted_features"] == []


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------


def test_monitoring_summary_end_to_end():
    log = pd.DataFrame(
        {
            "gamePk": [1, 2, 3],
            "predicted_total_goals": [6.0, 5.0, 7.0],
            "logged_at": ["2025-01-01T00:00", "2025-01-02T00:00", "2025-01-03T00:00"],
        }
    )
    results = pd.DataFrame({"gamePk": [1, 2], "totalGoals": [5, 6]})
    summary = monitoring.monitoring_summary(log, results)
    assert summary["n_logged"] == 3
    assert summary["n_reconciled"] == 2
    assert summary["realized"]["n"] == 2
    assert "status" in summary["prediction_drift"]


def test_monitoring_summary_empty_log():
    summary = monitoring.monitoring_summary(pd.DataFrame(), pd.DataFrame())
    assert summary["n_logged"] == 0
