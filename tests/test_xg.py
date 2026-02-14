from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import config
from src.features import add_features
from src.xg import build_xg_team_log, load_xg_games, normalize_team_name


def test_team_normalization_map_handles_st_louis():
    assert normalize_team_name("St Louis Blues") == "St. Louis Blues"


def test_xg_cache_and_schema_parse(tmp_path: Path):
    cache_dir = tmp_path / "xg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    season = "20242025"
    csv_path = cache_dir / f"{season}.csv"
    pd.DataFrame(
        {
            "season": [season],
            "date": ["2024-10-10"],
            "homeTeam": ["Boston Bruins"],
            "awayTeam": ["Toronto Maple Leafs"],
            "home_xG": [3.2],
            "away_xG": [2.8],
        }
    ).to_csv(csv_path, index=False)

    old = config.data.xg_cache_dir
    try:
        config.data.xg_cache_dir = cache_dir
        games = load_xg_games([season], use_cache=True)
    finally:
        config.data.xg_cache_dir = old

    assert not games.empty
    assert {"season", "date", "homeTeam", "awayTeam", "home_xG", "away_xG"} <= set(games.columns)


def test_build_team_log_uses_home_and_away_rows(tmp_path: Path):
    cache_dir = tmp_path / "xg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    season = "20242025"
    pd.DataFrame(
        {
            "season": [season],
            "date": ["2024-10-10"],
            "homeTeam": ["Boston Bruins"],
            "awayTeam": ["Toronto Maple Leafs"],
            "home_xG": [3.2],
            "away_xG": [2.8],
        }
    ).to_csv(cache_dir / f"{season}.csv", index=False)

    old = config.data.xg_cache_dir
    try:
        config.data.xg_cache_dir = cache_dir
        team_log = build_xg_team_log([season], use_cache=True)
    finally:
        config.data.xg_cache_dir = old

    assert len(team_log) == 2
    assert {"team", "opponent", "xGF", "xGA"} <= set(team_log.columns)


def test_add_features_default_include_xg_is_safe(sample_game_data):
    df = add_features(sample_game_data, include_goalies=False)
    assert not any(c.startswith("home_avg_xGF_") for c in df.columns)


def test_add_features_include_xg_no_leakage(sample_game_data, tmp_path: Path):
    season = str(sample_game_data["season"].iloc[0])
    cache_dir = tmp_path / "xg"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in sample_game_data.iterrows():
        rows.append(
            {
                "season": str(row["season"]),
                "date": str(row["date"]),
                "homeTeam": str(row["homeTeam"]),
                "awayTeam": str(row["awayTeam"]),
                "home_xG": float(row["homeScore"]) + 0.5,
                "away_xG": float(row["awayScore"]) + 0.5,
            }
        )
    pd.DataFrame(rows).to_csv(cache_dir / f"{season}.csv", index=False)

    old = config.data.xg_cache_dir
    try:
        config.data.xg_cache_dir = cache_dir
        feat = add_features(sample_game_data, include_goalies=False, include_xg=True, min_games=1)
    finally:
        config.data.xg_cache_dir = old

    xg_cols = [c for c in feat.columns if c.startswith("home_avg_xGF_")]
    assert xg_cols, "xG rolling columns were not added"

    # No leakage: first game for a team should not use current-game xG history.
    first_row = feat.sort_values("date").iloc[0]
    assert pd.isna(first_row["home_avg_xGF_5g"]) or first_row["home_avg_xGF_5g"] != first_row["homeScore"] + 0.5
