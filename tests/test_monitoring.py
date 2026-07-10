from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ModelArtifact,
    ModelMetadata,
    feature_schema_hash,
)
from src.monitoring import (
    PredictionLedger,
    histogram_reference,
    monitoring_summary,
    realized_metrics,
    reconcile_outcomes,
)


class _TeamModel:
    def predict_mu(self, frame):
        return np.full(len(frame), 6.0)


def _artifact() -> ModelArtifact:
    metadata = ModelMetadata(
        model_type="team_strength",
        feature_names=[],
        mae=1.9,
        rmse=2.3,
        baseline_mae=1.9,
        improvement_pct=0.0,
        training_date=datetime.now().isoformat(),
        n_training_samples=100,
        n_test_samples=20,
        data_seasons=["20242025"],
        git_commit="abc123",
        schema_version=ARTIFACT_SCHEMA_VERSION,
        benchmark_release="benchmark-v1",
        data_fingerprint="fingerprint",
        feature_schema_hash=feature_schema_hash([]),
        prediction_interface="game_frame",
        artifact_id="artifact-1",
        monitoring_reference={
            "predicted_total_goals": histogram_reference([5, 5.5, 6, 6.5, 7]),
            "features": {},
        },
    )
    return ModelArtifact(_TeamModel(), metadata)


def _prediction(mu=6.0):
    pmf = np.array([0.01, 0.03, 0.08, 0.12, 0.18, 0.22, 0.18, 0.1, 0.05, 0.02, 0.01])
    pmf = (pmf / pmf.sum()).tolist()
    return pd.DataFrame(
        [
            {
                "gamePk": 1,
                "date": "2025-01-01",
                "homeTeam": "A",
                "awayTeam": "B",
                "mu": mu,
                "pmf": pmf,
                "over_probs": {">6.5": sum(pmf[7:])},
                "feature_values": None,
            }
        ]
    )


def test_ledger_upserts_duplicate_game_artifact(tmp_path):
    ledger = PredictionLedger(tmp_path / "predictions.sqlite3")
    artifact = _artifact()
    ledger.upsert_predictions(_prediction(6.0), artifact)
    ledger.upsert_predictions(_prediction(6.2), artifact)
    loaded = ledger.load()
    assert len(loaded) == 1
    assert loaded.iloc[0]["mu"] == 6.2
    assert loaded.iloc[0]["pmf"] is not None


def test_reconcile_falls_back_per_row_when_some_ids_do_not_match():
    predictions = pd.DataFrame(
        [
            {"game_pk": 1, "game_date": "2025-01-01", "home_team": "A", "away_team": "B"},
            {"game_pk": 999, "game_date": "2025-01-02", "home_team": "C", "away_team": "D"},
        ]
    )
    results = pd.DataFrame(
        [
            {"gamePk": 1, "date": "2025-01-01", "homeTeam": "A", "awayTeam": "B", "totalGoals": 5},
            {"gamePk": 2, "date": "2025-01-02", "homeTeam": "C", "awayTeam": "D", "totalGoals": 7},
        ]
    )
    reconciled = reconcile_outcomes(predictions, results)
    assert reconciled["actual_total_goals"].tolist() == [5.0, 7.0]


def test_realized_metrics_include_full_distribution_scores(tmp_path):
    ledger = PredictionLedger(tmp_path / "predictions.sqlite3")
    artifact = _artifact()
    ledger.upsert_predictions(_prediction(), artifact)
    results = pd.DataFrame(
        [{"gamePk": 1, "date": "2025-01-01", "homeTeam": "A", "awayTeam": "B", "totalGoals": 6}]
    )
    reconciled = reconcile_outcomes(ledger.load(), results)
    metrics = realized_metrics(reconciled)
    assert metrics["n"] == 1
    assert metrics["probabilistic"]["n"] == 1
    assert np.isfinite(metrics["probabilistic"]["crps"])


def test_summary_is_honest_when_feature_drift_unavailable(tmp_path):
    ledger = PredictionLedger(tmp_path / "predictions.sqlite3")
    artifact = _artifact()
    ledger.upsert_predictions(_prediction(), artifact)
    summary = monitoring_summary(ledger, pd.DataFrame(), artifact)
    assert summary["drift"]["features"]["available"] is False
