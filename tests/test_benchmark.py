from __future__ import annotations

import pandas as pd
import pytest

from src.benchmark import (
    BenchmarkProtocol,
    assess_benchmark_data,
    build_release_artifact,
    prepare_benchmark_frame,
)
from src.evaluation import evaluate_holdout_forecast
from src.registry import ModelRegistry


def _four_season_frame(sample_game_data: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for index, season in enumerate(("20222023", "20232024", "20242025", "20252026")):
        chunk = sample_game_data.copy()
        dates = pd.to_datetime(chunk["date"]) + pd.DateOffset(years=index)
        chunk["date"] = dates.dt.strftime("%Y-%m-%d")
        chunk["season"] = season
        chunk["gamePk"] = chunk["gamePk"] + index * 10_000
        chunk["gameType"] = "R"
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def test_data_quality_and_primary_cohort_are_deterministic(sample_game_data):
    raw = _four_season_frame(sample_game_data)
    protocol = BenchmarkProtocol()
    report = assess_benchmark_data(raw, protocol)
    assert report.ready is True
    assert report.primary_row_count == len(raw)
    assert report.duplicate_game_ids == 0
    assert len(report.data_fingerprint) == 64

    frame, features = prepare_benchmark_frame(raw, protocol)
    assert not frame.empty
    assert features
    assert set(frame["gameType"]) == {"R"}


def test_data_quality_rejects_score_mismatch(sample_game_data):
    raw = _four_season_frame(sample_game_data)
    raw.loc[0, "totalGoals"] += 1
    report = assess_benchmark_data(raw, BenchmarkProtocol())
    assert report.ready is False
    assert report.score_mismatches == 1


def test_holdout_rejects_temporal_overlap(sample_features_df):
    with pytest.raises(ValueError, match="strictly after"):
        evaluate_holdout_forecast(
            sample_features_df.iloc[:120],
            sample_features_df.iloc[100:],
            point_model="team_strength",
        )


def test_holdout_scores_only_future_season(sample_game_data):
    raw = _four_season_frame(sample_game_data)
    frame, features = prepare_benchmark_frame(raw, BenchmarkProtocol())
    train = frame[frame["season"] != "20252026"]
    holdout = frame[frame["season"] == "20252026"]
    result = evaluate_holdout_forecast(
        train,
        holdout,
        point_model="team_strength",
        thresholds=[5.5, 6.5, 7.5],
    )
    assert result.folds[0].n_test == len(holdout)
    assert set(result.per_game["season"]) == {"20252026"}
    assert set(result.threshold_metrics) == {"5.5", "6.5", "7.5"}


def test_release_artifact_is_promoted_and_release_grade(sample_game_data, tmp_path):
    raw = _four_season_frame(sample_game_data)
    protocol = BenchmarkProtocol()
    quality = assess_benchmark_data(raw, protocol)
    frame, features = prepare_benchmark_frame(raw, protocol)
    manifest = {
        "selection": {"champion": {"model": "team_strength"}},
        "models": {
            "team_strength": {
                "metrics": {
                    "mae": 1.9,
                    "rmse": 2.3,
                    "crps": 1.3,
                    "dist_nll": 2.2,
                    "over_brier": 0.24,
                }
            }
        },
        "tuning": {"best_params": {}},
        "cohort": {"holdout_rows": 200},
        "data_quality": quality.to_dict(),
    }
    info = build_release_artifact(
        manifest=manifest,
        frame=frame,
        feature_columns=features,
        protocol=protocol,
        models_dir=tmp_path / "models",
    )
    assert info["prediction_interface"] == "game_frame"
    artifact = ModelRegistry(tmp_path / "models").get_production_model(require_release_grade=True)
    assert artifact is not None
    assert artifact.metadata.benchmark_release == "benchmark-v1"
    assert len(artifact.predict(frame.head(3))) == 3
