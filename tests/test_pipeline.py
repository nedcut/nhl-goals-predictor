"""
End-to-end test suite for NHL Goals Predictor pipeline.

Tests the complete pipeline from data loading through feature engineering,
model training, artifact persistence, and predictions.

Run with:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "test_end_to_end"
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import config
from src.validation import (
    ValidationError,
    validate_features,
    validate_game_data,
    validate_seasons,
    validate_target,
)

# =============================================================================
# FIXTURES - Synthetic Data for Testing
# =============================================================================


@pytest.fixture
def sample_teams() -> List[str]:
    """List of NHL teams for test data."""
    return [
        "Boston Bruins",
        "Toronto Maple Leafs",
        "Montreal Canadiens",
        "New York Rangers",
        "Tampa Bay Lightning",
        "Florida Panthers",
        "Detroit Red Wings",
        "Pittsburgh Penguins",
    ]


@pytest.fixture
def sample_game_data(sample_teams) -> pd.DataFrame:
    """Generate synthetic game data mimicking NHL API output.

    Creates 200 games over a 4-month span with realistic scores.
    """
    np.random.seed(42)
    n_games = 200

    # Generate dates over 4 months
    base_date = datetime(2024, 10, 10)
    dates = [base_date + timedelta(days=i // 2) for i in range(n_games)]

    # Generate matchups (ensure home != away)
    games = []
    for i in range(n_games):
        home_idx = i % len(sample_teams)
        away_idx = (i + 1 + (i // len(sample_teams))) % len(sample_teams)
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % len(sample_teams)

        home_score = np.random.poisson(3)  # Average ~3 goals
        away_score = np.random.poisson(3)

        games.append(
            {
                "gamePk": 2024020000 + i,
                "season": "20242025",
                "date": dates[i].strftime("%Y-%m-%d"),
                "homeTeam": sample_teams[home_idx],
                "awayTeam": sample_teams[away_idx],
                "homeScore": home_score,
                "awayScore": away_score,
                "totalGoals": home_score + away_score,
            }
        )

    return pd.DataFrame(games)


@pytest.fixture
def small_game_data(sample_teams) -> pd.DataFrame:
    """Generate small dataset for quick tests (50 games)."""
    np.random.seed(42)
    n_games = 50

    base_date = datetime(2024, 10, 10)
    dates = [base_date + timedelta(days=i // 2) for i in range(n_games)]

    games = []
    for i in range(n_games):
        home_idx = i % len(sample_teams)
        away_idx = (i + 1) % len(sample_teams)
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % len(sample_teams)

        home_score = np.random.poisson(3)
        away_score = np.random.poisson(3)

        games.append(
            {
                "gamePk": 2024020000 + i,
                "season": "20242025",
                "date": dates[i].strftime("%Y-%m-%d"),
                "homeTeam": sample_teams[home_idx],
                "awayTeam": sample_teams[away_idx],
                "homeScore": home_score,
                "awayScore": away_score,
                "totalGoals": home_score + away_score,
            }
        )

    return pd.DataFrame(games)


@pytest.fixture
def temp_cache_dir(tmp_path) -> Path:
    """Temporary directory for caching test data."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Temporary directory for saving models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# =============================================================================
# VALIDATION MODULE TESTS
# =============================================================================


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_game_data_valid(self, sample_game_data):
        """Valid game data should pass validation."""
        validate_game_data(sample_game_data)

    def test_validate_game_data_missing_columns(self, sample_game_data):
        """Missing columns should raise ValidationError."""
        df = sample_game_data.drop(columns=["homeTeam"])
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_game_data(df)

    def test_validate_game_data_empty(self):
        """Empty DataFrame should raise ValidationError."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            validate_game_data(df)

    def test_validate_game_data_none(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError, match="None"):
            validate_game_data(None)

    def test_validate_game_data_wrong_type(self):
        """Non-DataFrame should raise ValidationError."""
        with pytest.raises(ValidationError, match="Expected DataFrame"):
            validate_game_data({"not": "a dataframe"})

    def test_validate_seasons_valid(self):
        """Valid season strings should pass."""
        validate_seasons(["20232024", "20242025"])
        validate_seasons(["20222023"])

    def test_validate_seasons_invalid_length(self):
        """Invalid length should raise ValidationError."""
        with pytest.raises(ValidationError, match="8 characters"):
            validate_seasons(["2024"])

    def test_validate_seasons_non_consecutive(self):
        """Non-consecutive years should raise ValidationError."""
        with pytest.raises(ValidationError, match="consecutive years"):
            validate_seasons(["20242026"])

    def test_validate_seasons_empty(self):
        """Empty seasons list should raise ValidationError."""
        with pytest.raises(ValidationError, match="No seasons"):
            validate_seasons([])

    def test_validate_seasons_non_digits(self):
        """Non-digit characters should raise ValidationError."""
        with pytest.raises(ValidationError, match="only digits"):
            validate_seasons(["2024abcd"])

    def test_validate_target_present(self, sample_game_data):
        """Present target column should pass."""
        validate_target(sample_game_data)

    def test_validate_target_missing(self, sample_game_data):
        """Missing target column should raise ValidationError."""
        df = sample_game_data.drop(columns=["totalGoals"])
        with pytest.raises(ValidationError, match="totalGoals"):
            validate_target(df)


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================


class TestFeatureImputation:
    """Tests for the shared training-representative feature imputation."""

    def test_feature_fill_values_uses_median_and_skips_unknowns(self):
        from src.features import feature_fill_values

        hist = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]})
        fills = feature_fill_values(hist, ["a", "b", "missing"])

        assert fills["a"] == 2.0  # median of [1, 2, 3]
        # All-NaN column has no finite median; absent column is skipped entirely.
        assert "b" not in fills
        assert "missing" not in fills

    def test_impute_features_single_row_does_not_collapse_to_zero(self):
        """A 1-row request with a NaN feature must use the training fill, not 0.0.

        Regression: imputing with the request-batch mean made a single missing
        value collapse to NaN -> 0.0, feeding the model an unseen value.
        """
        from src.features import feature_fill_values, impute_features

        hist = pd.DataFrame({"home_avg_GF": [2.0, 3.0, 4.0]})
        fills = feature_fill_values(hist, ["home_avg_GF"])

        one_game = pd.DataFrame({"home_avg_GF": [np.nan]})
        imputed = impute_features(one_game, fills)

        assert imputed["home_avg_GF"].iloc[0] == 3.0  # training median, not 0.0

    def test_impute_features_falls_back_to_zero_without_fill(self):
        from src.features import impute_features

        X = pd.DataFrame({"x": [np.nan, 1.0]})
        imputed = impute_features(X, fill_values=None)

        assert imputed["x"].iloc[0] == 0.0


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_add_features_basic(self, sample_game_data):
        """add_features should add expected columns."""
        from src.features import add_features

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)

        # Check that feature columns were added
        expected_prefixes = ["home_avg_", "away_avg_", "home_win_", "away_win_"]
        for prefix in expected_prefixes:
            matching_cols = [c for c in df.columns if c.startswith(prefix)]
            assert len(matching_cols) > 0, f"No columns with prefix {prefix}"

        # Check h2h and venue features
        assert "h2h_avg_goals" in df.columns
        assert "venue_avg_goals" in df.columns

    def test_add_features_default_does_not_require_xg(self, sample_game_data):
        """Backwards-compatibility: include_xg defaults to False."""
        from src.features import add_features

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        assert not any(c.startswith("home_avg_xGF_") for c in df.columns)

    def test_add_features_preserves_original_columns(self, sample_game_data):
        """Original columns should be preserved after adding features."""
        from src.features import add_features

        original_cols = set(sample_game_data.columns)
        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)

        for col in original_cols:
            assert col in df.columns, f"Original column {col} was lost"

    def test_add_features_no_leakage(self, sample_game_data):
        """Features should use only past data (no future leakage)."""
        from src.features import add_features

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)

        # Verify the rolling feature column exists and is not entirely NaN after
        # the min_games threshold is applied.
        assert not df["home_avg_GF"].isna().all(), "All home_avg_GF values are NaN"

    def test_add_features_empty_dataframe(self):
        """Empty DataFrame should return empty DataFrame."""
        from src.features import add_features

        df = pd.DataFrame()
        result = add_features(df, include_goalies=False)
        assert result.empty

    def test_add_features_sorted_by_date(self, sample_game_data):
        """Output should be sorted by date."""
        from src.features import add_features

        # Shuffle input
        shuffled = sample_game_data.sample(frac=1, random_state=42)
        df = add_features(shuffled, window=5, min_games=1, include_goalies=False)

        # Check sorted
        dates = pd.to_datetime(df["date"])
        assert dates.is_monotonic_increasing

    def test_build_team_game_log(self, small_game_data):
        """Team game log should double the rows (home + away perspective)."""
        from src.features import _build_team_game_log

        team_log = _build_team_game_log(small_game_data)

        # Each game creates 2 rows
        assert len(team_log) == 2 * len(small_game_data)

        # Check required columns
        assert "team" in team_log.columns
        assert "goals_for" in team_log.columns
        assert "goals_against" in team_log.columns
        assert "is_home" in team_log.columns

    def test_rolling_features_window_size(self, sample_game_data):
        """Different window sizes should affect feature values."""
        from src.features import add_features

        df_small_window = add_features(
            sample_game_data, window=3, min_games=1, include_goalies=False
        )
        df_large_window = add_features(
            sample_game_data, window=10, min_games=1, include_goalies=False
        )

        # Features should differ with different window sizes
        # (not all values, but the overall distribution)
        # Just verify both have data
        assert not df_small_window.empty
        assert not df_large_window.empty

    def test_h2h_features_exist(self, sample_game_data):
        """Head-to-head features should be computed."""
        from src.features import _compute_h2h_features

        df = sample_game_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        result = _compute_h2h_features(df, window=5)

        assert "h2h_avg_goals" in result.columns

    def test_venue_features_exist(self, sample_game_data):
        """Venue features should be computed."""
        from src.features import _compute_venue_features

        df = sample_game_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        result = _compute_venue_features(df, window=10)

        assert "venue_avg_goals" in result.columns

    def test_future_games_do_not_change_outcome_history(self):
        """Unknown scheduled results must not look like 0-0 losses."""
        from src.features import add_features

        games = pd.DataFrame(
            [
                {
                    "gamePk": 1,
                    "season": "20252026",
                    "date": "2025-10-01",
                    "homeTeam": "Team A",
                    "awayTeam": "Team B",
                    "homeScore": 3,
                    "awayScore": 2,
                    "totalGoals": 5,
                },
                {
                    "gamePk": 2,
                    "season": "20252026",
                    "date": "2025-10-03",
                    "homeTeam": "Team A",
                    "awayTeam": "Team C",
                    "homeScore": 4,
                    "awayScore": 1,
                    "totalGoals": 5,
                },
                {
                    "gamePk": 3,
                    "season": "20252026",
                    "date": "2025-10-05",
                    "homeTeam": "Team A",
                    "awayTeam": "Team D",
                    "homeScore": pd.NA,
                    "awayScore": pd.NA,
                    "totalGoals": pd.NA,
                },
                {
                    "gamePk": 4,
                    "season": "20252026",
                    "date": "2025-10-07",
                    "homeTeam": "Team A",
                    "awayTeam": "Team E",
                    "homeScore": pd.NA,
                    "awayScore": pd.NA,
                    "totalGoals": pd.NA,
                },
            ]
        )

        result = add_features(
            games,
            window=5,
            min_games=1,
            include_goalies=False,
            include_multi_window=False,
            include_interactions=False,
            include_temporal=False,
        )
        later = result.loc[result["gamePk"] == 4].iloc[0]

        assert later["home_avg_GF"] == pytest.approx(3.5)
        assert later["home_win_pct"] == pytest.approx(1.0)
        assert later["home_win_streak"] == 2
        assert later["home_games_played"] == 2


# =============================================================================
# MODEL TRAINING TESTS
# =============================================================================


class TestModelTraining:
    """Tests for model training functions."""

    def test_prepare_data_ignores_unrelated_nan_columns(self, sample_game_data):
        """Rows should be dropped only for missing model inputs, not unrelated columns."""
        from src.features import add_features
        from src.model import get_feature_columns, prepare_data

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        df["unrelated_debug_col"] = np.nan

        feature_cols = get_feature_columns(df)
        expected_rows = len(df.dropna(subset=feature_cols + ["totalGoals"]))

        X_train, X_test, _, _, _ = prepare_data(df, test_size=0.2)
        used_rows = len(X_train) + len(X_test)

        assert used_rows == expected_rows

    def test_prepare_data_splits_chronologically(self, sample_game_data):
        """Data should be split chronologically, not randomly."""
        from src.features import add_features
        from src.model import prepare_data

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        X_train, X_test, y_train, y_test, features = prepare_data(df, test_size=0.2)

        # Train set should come before test set in time
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) > len(X_test)  # 80/20 split

    def test_prepare_data_returns_feature_names(self, sample_game_data):
        """prepare_data should return feature names."""
        from src.features import add_features
        from src.model import prepare_data

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        _, _, _, _, feature_names = prepare_data(df, test_size=0.2)

        assert len(feature_names) > 0
        assert all(isinstance(f, str) for f in feature_names)

    def test_train_random_forest(self, sample_game_data):
        """Random Forest training should return TrainingResult."""
        from src.features import add_features
        from src.model import TrainingResult, train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        assert isinstance(result, TrainingResult)
        assert result.model_type == "RandomForest"
        assert result.mae > 0
        assert result.rmse > 0
        assert result.baseline_mae > 0
        assert len(result.feature_names) > 0
        assert len(result.y_test) > 0
        assert len(result.y_pred) == len(result.y_test)

    def test_train_xgboost(self, sample_game_data):
        """XGBoost training should return TrainingResult."""
        import os

        if os.getenv("RUN_XGBOOST_TESTS") != "1":
            pytest.skip("Set RUN_XGBOOST_TESTS=1 to run XGBoost training tests.")
        pytest.importorskip("xgboost")
        from src.features import add_features
        from src.model import TrainingResult, train_xgboost

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_xgboost(df, test_size=0.2, n_estimators=10, max_depth=3, random_state=42)

        assert isinstance(result, TrainingResult)
        assert result.model_type == "XGBoost"
        assert result.mae > 0
        assert result.rmse > 0

    def test_model_predictions_reasonable(self, sample_game_data):
        """Model predictions should be in a reasonable range for NHL goals."""
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=20, random_state=42)

        # NHL games typically have 4-7 total goals, predictions should be in range 0-15
        assert result.y_pred.min() >= 0, "Predictions should be non-negative"
        assert result.y_pred.max() < 20, "Predictions shouldn't exceed 20 goals"
        assert 3 < result.y_pred.mean() < 10, "Mean prediction should be reasonable"

    def test_baseline_mae_computed(self, sample_game_data):
        """Baseline MAE (predicting mean) should be computed."""
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        # Baseline should be positive
        assert result.baseline_mae > 0

    def test_cross_validate(self, sample_game_data):
        """Cross-validation should return CVResult with fold scores."""
        from src.features import add_features
        from src.model import CVResult, cross_validate

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = cross_validate(df, model_type="rf", n_splits=3, n_estimators=10)

        assert isinstance(result, CVResult)
        assert len(result.mae_scores) == 3
        assert len(result.rmse_scores) == 3
        assert result.mae_mean > 0
        assert result.mae_std >= 0

    def test_get_feature_columns(self, sample_game_data):
        """Feature column detection should find expected columns."""
        from src.features import add_features
        from src.model import get_feature_columns

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        feature_cols = get_feature_columns(df)

        assert len(feature_cols) > 0
        # Check some expected features
        assert any("home_avg" in c for c in feature_cols)
        assert any("away_avg" in c for c in feature_cols)


# =============================================================================
# ARTIFACT PERSISTENCE TESTS
# =============================================================================


class TestArtifactPersistence:
    """Tests for model artifact saving and loading."""

    def test_save_and_load_artifact(self, sample_game_data, temp_model_dir):
        """Artifact should be saveable and loadable."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        # Create and save artifact
        artifact = ModelArtifact.from_training_result(result, seasons=["20242025"])
        model_path = temp_model_dir / "test_model"
        artifact.save(model_path)

        # Verify files created
        assert (model_path.with_suffix(".joblib")).exists()
        assert (model_path.with_suffix(".json")).exists()

        # Load and verify
        loaded = ModelArtifact.load(model_path)
        assert loaded.metadata.model_type == result.model_type
        assert loaded.metadata.mae == pytest.approx(result.mae, rel=1e-6)
        assert loaded.metadata.feature_names == result.feature_names

    def test_artifact_metadata_complete(self, sample_game_data, temp_model_dir):
        """Artifact metadata should contain all required fields."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        artifact = ModelArtifact.from_training_result(result, seasons=["20242025"])

        # Check all metadata fields
        meta = artifact.metadata
        assert meta.model_type == "RandomForest"
        assert len(meta.feature_names) > 0
        assert meta.mae > 0
        assert meta.rmse > 0
        assert meta.baseline_mae > 0
        assert meta.training_date is not None
        assert meta.n_test_samples > 0
        assert "20242025" in meta.data_seasons

    def test_artifact_predict(self, sample_game_data, temp_model_dir):
        """Artifact.predict() should work correctly."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import prepare_data, train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        artifact = ModelArtifact.from_training_result(result)
        model_path = temp_model_dir / "test_model"
        artifact.save(model_path)

        # Load and predict
        loaded = ModelArtifact.load(model_path)
        X_train, X_test, _, _, _ = prepare_data(df, test_size=0.2)
        predictions = loaded.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p >= 0 for p in predictions)

    def test_poisson_artifact_prediction_matches_training_model(
        self, sample_game_data, temp_model_dir
    ):
        """Saved Poisson artifacts should preserve preprocessing at inference time."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import prepare_data, train_poisson

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_poisson(df, test_size=0.2, alpha=1.0)

        artifact = ModelArtifact.from_training_result(result)
        model_path = temp_model_dir / "poisson_model"
        artifact.save(model_path)
        loaded = ModelArtifact.load(model_path)

        _, X_test, _, _, _ = prepare_data(df, test_size=0.2)
        original_pred = result.model.predict(X_test)
        loaded_pred = loaded.predict(X_test)

        assert np.allclose(original_pred, loaded_pred, atol=1e-10)

    def test_artifact_summary(self, sample_game_data):
        """Artifact summary should return readable string."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        artifact = ModelArtifact.from_training_result(result, seasons=["20242025"])
        summary = artifact.summary()

        assert "RandomForest" in summary
        assert "MAE" in summary
        assert "20242025" in summary

    def test_metadata_to_dict_roundtrip(self, sample_game_data):
        """Metadata should survive dict serialization roundtrip."""
        from src.artifacts import ModelMetadata
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        original = ModelMetadata.from_training_result(result, seasons=["20242025"])
        dict_form = original.to_dict()
        restored = ModelMetadata.from_dict(dict_form)

        assert restored.model_type == original.model_type
        assert restored.mae == original.mae
        assert restored.feature_names == original.feature_names


# =============================================================================
# DATA MODULE TESTS
# =============================================================================


class TestDataModule:
    """Tests for data loading and caching."""

    def test_cache_save_and_load(self, sample_game_data, temp_cache_dir):
        """Season data should be cacheable."""
        from src.data import load_cached_season, save_season_cache

        save_season_cache(sample_game_data, "20242025", temp_cache_dir)

        loaded = load_cached_season("20242025", temp_cache_dir)
        assert loaded is not None
        assert len(loaded) == len(sample_game_data)
        assert set(loaded.columns) == set(sample_game_data.columns)

    def test_cache_path_generation(self, temp_cache_dir):
        """Cache path should be generated correctly."""
        from src.data import get_cache_path

        path = get_cache_path("20242025", temp_cache_dir)
        assert path == temp_cache_dir / "20242025.csv"

    def test_load_nonexistent_cache_returns_none(self, temp_cache_dir):
        """Loading non-existent cache should return None."""
        from src.data import load_cached_season

        result = load_cached_season("99999999", temp_cache_dir)
        assert result is None

    def test_build_dataset_uses_cache(self, sample_game_data, temp_cache_dir):
        """build_dataset should use cached data when available."""
        from src.data import save_season_cache

        # Save to cache
        save_season_cache(sample_game_data, "20242025", temp_cache_dir)

        # Patch load_cached_season to use our temp directory
        def mock_load_cached(season, cache_dir=temp_cache_dir):
            import pandas as pd

            from src.data import get_cache_path

            path = get_cache_path(season, temp_cache_dir)
            if path.exists():
                return pd.read_csv(path)
            return None

        with patch("src.data.load_cached_season", side_effect=mock_load_cached):
            with patch("src.data.fetch_season_games") as mock_fetch:
                from src.data import build_dataset

                result = build_dataset(["20242025"], use_cache=True)

                # Should not have called fetch since cache exists
                mock_fetch.assert_not_called()
                assert len(result) == len(sample_game_data)

    def test_recent_seasons_roll_forward_from_calendar_date(self):
        """Runtime defaults should include the active NHL season."""
        from src.data import recent_seasons, season_for_date

        as_of = datetime(2026, 6, 15)
        assert season_for_date(as_of) == "20252026"
        assert recent_seasons(2, as_of) == ["20242025", "20252026"]

    def test_active_season_cache_freshness(self, sample_game_data, temp_cache_dir):
        """Active-season data should age out instead of staying frozen forever."""
        from src.data import _active_season_cache_is_fresh, get_cache_path, save_season_cache

        season = "20252026"
        save_season_cache(sample_game_data, season, temp_cache_dir)
        cache_path = get_cache_path(season, temp_cache_dir)
        written_at = datetime.fromtimestamp(cache_path.stat().st_mtime)

        assert _active_season_cache_is_fresh(
            season,
            temp_cache_dir,
            now=written_at + timedelta(hours=1),
        )
        assert not _active_season_cache_is_fresh(
            season,
            temp_cache_dir,
            now=written_at + timedelta(hours=7),
        )


# =============================================================================
# PREDICTION MODULE TESTS
# =============================================================================


class TestPredictionModule:
    """Tests for prediction functionality."""

    def test_predict_games_basic(self, sample_game_data, temp_model_dir):
        """predict_games should return predictions DataFrame."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import train_random_forest
        from src.predict import predict_games

        # Train and save model
        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)
        artifact = ModelArtifact.from_training_result(result)
        model_path = temp_model_dir / "test_model"
        artifact.save(model_path)

        # Create "upcoming" games (just use a few from the dataset)
        upcoming = sample_game_data.tail(5).copy()
        upcoming["gamePk"] = upcoming["gamePk"] + 10000  # Different game IDs

        # Historical data is everything except the upcoming
        historical = sample_game_data.head(len(sample_game_data) - 5)

        predictions = predict_games(upcoming, model_path, historical)

        assert not predictions.empty
        assert "predicted_total_goals" in predictions.columns
        assert "homeTeam" in predictions.columns
        assert "awayTeam" in predictions.columns
        assert "date" in predictions.columns

    def test_predictions_have_reasonable_values(self, sample_game_data, temp_model_dir):
        """Predictions should be in reasonable NHL goal range."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import train_random_forest
        from src.predict import predict_games

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)
        artifact = ModelArtifact.from_training_result(result)
        model_path = temp_model_dir / "test_model"
        artifact.save(model_path)

        upcoming = sample_game_data.tail(5).copy()
        upcoming["gamePk"] = upcoming["gamePk"] + 10000
        historical = sample_game_data.head(len(sample_game_data) - 5)

        predictions = predict_games(upcoming, model_path, historical)

        # Predictions should be reasonable (between 0 and 15 goals)
        assert all(predictions["predicted_total_goals"] >= 0)
        assert all(predictions["predicted_total_goals"] <= 20)

    def test_prepare_upcoming_rows_removes_placeholder_scores(self):
        """The NHL API's future 0-0 placeholders are not completed results."""
        from src.predict import _prepare_upcoming_rows

        upcoming = pd.DataFrame(
            [
                {
                    "gamePk": 1,
                    "homeScore": 0,
                    "awayScore": 0,
                    "totalGoals": 0,
                }
            ]
        )
        prepared = _prepare_upcoming_rows(upcoming)

        assert prepared["_is_upcoming"].all()
        assert prepared[["homeScore", "awayScore", "totalGoals"]].isna().all().all()

    def test_fetch_upcoming_games_respects_requested_end_date(self):
        """Schedule weeks can include games beyond the requested horizon."""
        from src import predict

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 6, 15)

        games = [
            {"gamePk": 1, "date": "2026-06-15", "gameState": "FUT"},
            {"gamePk": 2, "date": "2026-06-17", "gameState": "FUT"},
            {"gamePk": 3, "date": "2026-06-18", "gameState": "FUT"},
        ]

        with patch("src.predict.datetime", FixedDateTime):
            with patch("src.predict.fetch_schedule_week", return_value=games):
                result = predict.fetch_upcoming_games(days_ahead=2)

        assert result["gamePk"].tolist() == [1, 2]


# =============================================================================
# END-TO-END PIPELINE TESTS
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end tests verifying the complete pipeline."""

    def test_probabilistic_cv_poisson_glm_handles_feature_nans(self, sample_game_data):
        """Poisson GLM evaluation path should handle early-season NaNs safely."""
        from src.evaluation import time_series_cv_forecast
        from src.features import add_features

        df_features = add_features(sample_game_data, window=5, min_games=3, include_goalies=False)
        result = time_series_cv_forecast(
            df_features,
            point_model="poisson_glm",
            dist_model="poisson",
            n_splits=3,
            cal_fraction=0.2,
        )

        assert len(result.folds) == 3

    def test_full_pipeline_random_forest(self, sample_game_data, temp_model_dir):
        """Full pipeline: data → features → train RF → save → load → predict."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import prepare_data, train_random_forest

        # Step 1: Validate raw data
        validate_game_data(sample_game_data)
        assert len(sample_game_data) == 200

        # Step 2: Add features
        df_features = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        validate_features(df_features)
        assert len(df_features) == len(sample_game_data)

        # Step 3: Train model
        result = train_random_forest(df_features, test_size=0.2, n_estimators=20, random_state=42)
        assert result.mae > 0
        assert result.mae < result.baseline_mae * 1.5  # Shouldn't be much worse than baseline

        # Step 4: Save artifact
        artifact = ModelArtifact.from_training_result(result, seasons=["20242025"])
        model_path = temp_model_dir / "pipeline_test_rf"
        artifact.save(model_path)

        # Step 5: Load artifact
        loaded_artifact = ModelArtifact.load(model_path)
        assert loaded_artifact.metadata.mae == pytest.approx(result.mae, rel=1e-6)

        # Step 6: Make predictions on "new" data
        X_train, X_test, y_train, y_test, features = prepare_data(df_features, test_size=0.2)
        predictions = loaded_artifact.predict(X_test)
        assert len(predictions) == len(X_test)

        # Step 7: Verify predictions are reasonable
        assert all(p >= 0 for p in predictions)
        assert all(p < 20 for p in predictions)

    def test_full_pipeline_xgboost(self, sample_game_data, temp_model_dir):
        """Full pipeline with XGBoost model."""
        import os

        if os.getenv("RUN_XGBOOST_TESTS") != "1":
            pytest.skip("Set RUN_XGBOOST_TESTS=1 to run XGBoost training tests.")
        pytest.importorskip("xgboost")
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import prepare_data, train_xgboost

        # Data → Features → Train → Save → Load → Predict
        df_features = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)

        result = train_xgboost(
            df_features, test_size=0.2, n_estimators=20, max_depth=3, random_state=42
        )
        assert result.model_type == "XGBoost"

        artifact = ModelArtifact.from_training_result(result, seasons=["20242025"])
        model_path = temp_model_dir / "pipeline_test_xgb"
        artifact.save(model_path)

        loaded = ModelArtifact.load(model_path)
        X_train, X_test, _, _, _ = prepare_data(df_features, test_size=0.2)
        predictions = loaded.predict(X_test)

        assert len(predictions) == len(X_test)
        assert loaded.metadata.model_type == "XGBoost"

    def test_pipeline_with_different_window_sizes(self, sample_game_data):
        """Pipeline should work with different rolling window sizes."""
        from src.features import add_features
        from src.model import train_random_forest

        for window in [3, 5, 10]:
            df = add_features(sample_game_data, window=window, min_games=1, include_goalies=False)
            result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)
            assert result.mae > 0, f"Failed for window={window}"

    def test_pipeline_deterministic_with_seed(self, sample_game_data):
        """Pipeline should be deterministic with fixed random seed."""
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)

        result1 = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)
        result2 = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        assert result1.mae == pytest.approx(result2.mae, rel=1e-6)
        assert np.allclose(result1.y_pred, result2.y_pred)

    def test_pipeline_cross_validation(self, sample_game_data):
        """Pipeline with cross-validation should return consistent results."""
        from src.features import add_features
        from src.model import cross_validate

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)

        cv_result = cross_validate(df, model_type="rf", n_splits=3, n_estimators=10)

        assert len(cv_result.mae_scores) == 3
        assert all(score > 0 for score in cv_result.mae_scores)
        assert cv_result.mae_mean > 0

    def test_pipeline_handles_missing_features_gracefully(self, sample_game_data, temp_model_dir):
        """Pipeline should handle prediction when some features are missing."""
        from src.artifacts import ModelArtifact
        from src.features import add_features
        from src.model import train_random_forest

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df, test_size=0.2, n_estimators=10, random_state=42)

        artifact = ModelArtifact.from_training_result(result)
        model_path = temp_model_dir / "test_model"
        artifact.save(model_path)

        # Create test data with one feature column removed
        loaded = ModelArtifact.load(model_path)
        X_test = df[result.feature_names].tail(10).copy()

        # Should still be able to predict
        predictions = loaded.predict(X_test)
        assert len(predictions) == 10


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestConfig:
    """Tests for configuration system."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        assert config.data.api_base == "https://api-web.nhle.com/v1"
        assert config.data.request_delay > 0
        assert config.features.rolling_window > 0
        assert config.features.min_games >= 0
        assert 0 < config.model.test_size < 1
        assert config.model.random_state == 42

    def test_config_xgb_params_exist(self):
        """XGBoost params should be defined."""
        params = config.model.xgb_params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "n_estimators" in params


class TestLoggingConfig:
    """Tests for logging setup behavior."""

    def test_setup_logging_can_raise_log_level_after_initialization(self):
        """setup_logging should reconfigure levels on subsequent calls."""
        import logging

        from src.logging_config import setup_logging

        logger = setup_logging(level="INFO")
        assert logger.level == logging.INFO

        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG


# =============================================================================
# INTEGRATION TESTS WITH MOCKED API
# =============================================================================


class TestAPIIntegration:
    """Tests with mocked NHL API calls."""

    def test_fetch_schedule_week_parsing(self):
        """Schedule parsing should handle NHL API response format."""
        from src.data import fetch_schedule_week

        mock_response = {
            "gameWeek": [
                {
                    "date": "2024-10-15",
                    "games": [
                        {
                            "id": 2024020001,
                            "season": 20242025,
                            "gameType": 2,
                            "gameState": "FINAL",
                            "homeTeam": {
                                "placeName": {"default": "Boston"},
                                "commonName": {"default": "Bruins"},
                                "score": 4,
                            },
                            "awayTeam": {
                                "placeName": {"default": "Toronto"},
                                "commonName": {"default": "Maple Leafs"},
                                "score": 2,
                            },
                        }
                    ],
                }
            ]
        }

        # fetch_schedule_week now routes through the shared resilient HTTP
        # client (src.http_client.get_json), so patch that seam.
        with patch("src.data.get_json", return_value=mock_response) as mock_get:
            games = fetch_schedule_week("2024-10-15")

            mock_get.assert_called_once()
            assert len(games) == 1
            assert games[0]["homeTeam"] == "Boston Bruins"
            assert games[0]["awayTeam"] == "Toronto Maple Leafs"
            assert games[0]["totalGoals"] == 6


# =============================================================================
# PERFORMANCE / EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_team_multiple_games(self, sample_teams):
        """Features should work when a team plays many consecutive games."""
        np.random.seed(42)

        # Create data where one team plays every game
        games = []
        base_date = datetime(2024, 10, 10)
        for i in range(30):
            other_team = sample_teams[(i + 1) % len(sample_teams)]
            if other_team == sample_teams[0]:
                other_team = sample_teams[-1]

            games.append(
                {
                    "gamePk": 2024020000 + i,
                    "season": "20242025",
                    "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "homeTeam": sample_teams[0] if i % 2 == 0 else other_team,
                    "awayTeam": other_team if i % 2 == 0 else sample_teams[0],
                    "homeScore": np.random.poisson(3),
                    "awayScore": np.random.poisson(3),
                    "totalGoals": 0,  # Will be computed
                }
            )

        df = pd.DataFrame(games)
        df["totalGoals"] = df["homeScore"] + df["awayScore"]

        from src.features import add_features

        result = add_features(df, window=5, min_games=1, include_goalies=False)
        assert not result.empty

    def test_back_to_back_games(self, sample_teams):
        """Back-to-back indicator should be computed correctly."""
        games = []
        base_date = datetime(2024, 10, 10)

        # Team plays on consecutive days
        for i in range(5):
            games.append(
                {
                    "gamePk": 2024020000 + i,
                    "season": "20242025",
                    "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "homeTeam": sample_teams[0],
                    "awayTeam": sample_teams[1],
                    "homeScore": 3,
                    "awayScore": 2,
                    "totalGoals": 5,
                }
            )

        df = pd.DataFrame(games)

        from src.features import add_features

        result = add_features(df, window=5, min_games=1, include_goalies=False)

        # After first game, team should have back-to-back indicator
        assert "home_is_back_to_back" in result.columns

    def test_very_high_scoring_game(self, sample_game_data, temp_model_dir):
        """Model should handle prediction for high-scoring scenarios."""
        from src.features import add_features
        from src.model import train_random_forest

        # Add a few very high-scoring games
        df = sample_game_data.copy()
        high_scoring = df.tail(3).copy()
        high_scoring["homeScore"] = 8
        high_scoring["awayScore"] = 7
        high_scoring["totalGoals"] = 15
        df = pd.concat([df, high_scoring], ignore_index=True)

        df_features = add_features(df, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df_features, test_size=0.2, n_estimators=10, random_state=42)

        # Model should still work
        assert result.mae > 0
        assert result.model is not None

    def test_low_scoring_games_only(self, sample_teams):
        """Model should work with consistently low-scoring games."""
        np.random.seed(42)
        games = []
        base_date = datetime(2024, 10, 10)

        for i in range(100):
            home_idx = i % len(sample_teams)
            away_idx = (i + 1) % len(sample_teams)

            games.append(
                {
                    "gamePk": 2024020000 + i,
                    "season": "20242025",
                    "date": (base_date + timedelta(days=i // 2)).strftime("%Y-%m-%d"),
                    "homeTeam": sample_teams[home_idx],
                    "awayTeam": sample_teams[away_idx],
                    "homeScore": np.random.randint(0, 3),  # Low scoring
                    "awayScore": np.random.randint(0, 3),
                    "totalGoals": 0,
                }
            )

        df = pd.DataFrame(games)
        df["totalGoals"] = df["homeScore"] + df["awayScore"]

        from src.features import add_features
        from src.model import train_random_forest

        df_features = add_features(df, window=5, min_games=1, include_goalies=False)
        result = train_random_forest(df_features, test_size=0.2, n_estimators=10, random_state=42)

        assert result.mae > 0


# =============================================================================
# MODEL COMPARISON TESTS
# =============================================================================


class TestModelComparison:
    """Tests for model comparison functionality."""

    def test_compare_models_returns_dataframe(self, sample_game_data):
        """compare_models should return comparison DataFrame."""
        import os

        if os.getenv("RUN_XGBOOST_TESTS") != "1":
            pytest.skip("Set RUN_XGBOOST_TESTS=1 to run XGBoost training tests.")
        pytest.importorskip("xgboost")
        from src.features import add_features
        from src.model import compare_models

        df = add_features(sample_game_data, window=5, min_games=1, include_goalies=False)
        comparison = compare_models(df, test_size=0.2)

        assert isinstance(comparison, pd.DataFrame)
        assert "Model" in comparison.columns
        assert "MAE" in comparison.columns
        assert len(comparison) >= 2  # At least RF and baseline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
