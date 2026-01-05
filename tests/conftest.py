"""
Pytest configuration and shared fixtures for NHL Goals Predictor tests.

Fixtures defined here are automatically available to all test files in the tests/ directory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest


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

    base_date = datetime(2024, 10, 10)
    dates = [base_date + timedelta(days=i // 2) for i in range(n_games)]

    games = []
    for i in range(n_games):
        home_idx = i % len(sample_teams)
        away_idx = (i + 1 + (i // len(sample_teams))) % len(sample_teams)
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % len(sample_teams)

        home_score = np.random.poisson(3)
        away_score = np.random.poisson(3)

        games.append({
            "gamePk": 2024020000 + i,
            "season": "20242025",
            "date": dates[i].strftime("%Y-%m-%d"),
            "homeTeam": sample_teams[home_idx],
            "awayTeam": sample_teams[away_idx],
            "homeScore": home_score,
            "awayScore": away_score,
            "totalGoals": home_score + away_score,
        })

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

        games.append({
            "gamePk": 2024020000 + i,
            "season": "20242025",
            "date": dates[i].strftime("%Y-%m-%d"),
            "homeTeam": sample_teams[home_idx],
            "awayTeam": sample_teams[away_idx],
            "homeScore": home_score,
            "awayScore": away_score,
            "totalGoals": home_score + away_score,
        })

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


@pytest.fixture
def sample_features_df(sample_game_data):
    """Game data with features already added."""
    from src.features import add_features
    return add_features(sample_game_data, window=5, min_games=1, include_goalies=False)


@pytest.fixture
def trained_model_artifact(sample_features_df):
    """A trained model artifact for testing predictions."""
    from src.artifacts import ModelArtifact
    from src.model import train_random_forest

    result = train_random_forest(
        sample_features_df, test_size=0.2, n_estimators=10, random_state=42
    )
    return ModelArtifact.from_training_result(result, seasons=["20242025"])
