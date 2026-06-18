"""
Goaltender data fetching and feature engineering.

This module handles fetching starting goalie statistics from NHL game boxscores
and computing rolling goalie performance features for prediction.

Features generated:
- home_goalie_sv_pct / away_goalie_sv_pct: Rolling save percentage (10-game window)
- home_goalie_gaa / away_goalie_gaa: Rolling goals against average (10-game window)

Data is cached to data/goalies/goalie_stats.csv to avoid re-fetching.

Usage:
    from src.goalies import add_goalie_features

    df_with_goalies = add_goalie_features(df, fetch_missing=True)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import config
from .http_client import get_json
from .logging_config import get_logger

logger = get_logger(__name__)

API_BASE = config.data.api_base
GOALIE_CACHE_DIR = config.data.goalie_cache_dir


def _parse_toi_to_seconds(toi: str) -> int:
    """Parse TOI string like 'MM:SS' or 'H:MM:SS' to total seconds."""
    if not toi:
        return 0
    parts = toi.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return 0


def fetch_game_goalies(game_id: int) -> Dict:
    """Fetch starting goalie info from a game boxscore.

    Returns
    -------
    dict with keys:
        - gamePk: game ID
        - home_goalie_id: starting goalie player ID
        - home_goalie_name: goalie name
        - away_goalie_id: starting goalie player ID
        - away_goalie_name: goalie name
        - home_saves: saves by home goalie
        - home_shots_against: shots against home goalie
        - away_saves: saves by away goalie
        - away_shots_against: shots against away goalie
    """
    url = f"{API_BASE}/gamecenter/{game_id}/boxscore"
    data = get_json(url)

    result = {"gamePk": game_id}

    for side in ["home", "away"]:
        team_key = f"{side}Team"
        stats = data.get("playerByGameStats", {}).get(team_key, {})
        goalies = stats.get("goalies", [])

        # Find the starting goalie
        starter = None
        for g in goalies:
            if g.get("starter", False):
                starter = g
                break

        # If no starter flag, use goalie with most TOI (parse to seconds for proper comparison)
        if starter is None and goalies:
            starter = max(goalies, key=lambda x: _parse_toi_to_seconds(x.get("toi", "00:00")))

        if starter:
            result[f"{side}_goalie_id"] = starter.get("playerId")
            result[f"{side}_goalie_name"] = starter.get("name", {}).get("default", "Unknown")
            result[f"{side}_saves"] = starter.get("saves", 0)
            result[f"{side}_shots_against"] = starter.get("shotsAgainst", 0)
            result[f"{side}_goals_against"] = starter.get("goalsAgainst", 0)
            result[f"{side}_save_pct"] = starter.get("savePctg", 0.0)
        else:
            result[f"{side}_goalie_id"] = None
            result[f"{side}_goalie_name"] = None
            result[f"{side}_saves"] = 0
            result[f"{side}_shots_against"] = 0
            result[f"{side}_goals_against"] = 0
            result[f"{side}_save_pct"] = 0.0

    return result


def fetch_goalies_for_games(
    game_ids: List[int],
    delay: float | None = None,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch goalie data for multiple games.

    Parameters
    ----------
    game_ids : list of int
        Game IDs to fetch.
    delay : float
        Delay between requests.
    cache_path : Path, optional
        If provided, save/load from this cache file.

    Returns
    -------
    pd.DataFrame
        Goalie data for each game.
    """
    if delay is None:
        delay = config.data.goalie_request_delay

    # Check cache
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached_ids = set(cached["gamePk"].tolist())
        game_ids = [g for g in game_ids if g not in cached_ids]
        if not game_ids:
            return cached
        logger.info("Found %d cached, fetching %d new games", len(cached), len(game_ids))
    else:
        cached = pd.DataFrame()

    results = []
    failed = []

    for game_id in tqdm(game_ids, desc="Fetching goalie data"):
        try:
            data = fetch_game_goalies(game_id)
            results.append(data)
        except Exception as e:
            logger.warning("Failed to fetch goalie data for game %d: %s", game_id, e)
            failed.append(game_id)
        time.sleep(delay)

    if failed:
        logger.warning("Failed to fetch %d games total", len(failed))

    if results:
        new_df = pd.DataFrame(results)
        combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df

        # Save cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(cache_path, index=False)
            logger.info("Cached %d games to %s", len(combined), cache_path)

        return combined

    return cached if not cached.empty else pd.DataFrame()


def _compute_goalie_rolling_for_side(
    df: pd.DataFrame,
    side: str,
    window: int,
) -> pd.DataFrame:
    """Compute rolling goalie stats for one side (home or away).

    This is a vectorized helper that uses groupby/transform instead of iterrows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with goalie data merged in.
    side : str
        "home" or "away".
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling stats columns added.
    """
    goalie_col = f"{side}_goalie_id"
    saves_col = f"{side}_saves"
    shots_col = f"{side}_shots_against"
    goals_col = f"{side}_goals_against"

    # Create a goalie-game dataframe for this side
    goalie_games = df[["gamePk", "date", goalie_col, saves_col, shots_col, goals_col]].copy()
    goalie_games = goalie_games.rename(columns={
        goalie_col: "goalie_id",
        saves_col: "saves",
        shots_col: "shots",
        goals_col: "goals_against",
    })

    # Remove rows without goalie data
    goalie_games = goalie_games.dropna(subset=["goalie_id"])
    goalie_games["goalie_id"] = goalie_games["goalie_id"].astype(int)

    # Sort by date for proper rolling calculations
    goalie_games = goalie_games.sort_values("date").reset_index(drop=True)

    # Compute rolling stats per goalie using shift(1) to avoid data leakage
    grouped = goalie_games.groupby("goalie_id")

    # Rolling sums for save percentage calculation
    goalie_games["rolling_saves"] = grouped["saves"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )
    goalie_games["rolling_shots"] = grouped["shots"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )

    # Rolling sum for GAA calculation
    goalie_games["rolling_goals"] = grouped["goals_against"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )
    # Use rolling count within window, not cumulative count
    goalie_games["rolling_games"] = grouped["goals_against"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).count()
    )

    # Compute save percentage: total_saves / total_shots
    goalie_games[f"{side}_goalie_sv_pct"] = np.where(
        goalie_games["rolling_shots"] > 0,
        goalie_games["rolling_saves"] / goalie_games["rolling_shots"],
        np.nan
    )

    # Compute GAA: total_goals / games_played (within rolling window)
    goalie_games[f"{side}_goalie_gaa"] = np.where(
        goalie_games["rolling_games"] > 0,
        goalie_games["rolling_goals"] / goalie_games["rolling_games"],
        np.nan
    )

    # Return only the columns we need for merging
    return goalie_games[["gamePk", f"{side}_goalie_sv_pct", f"{side}_goalie_gaa"]]


def compute_goalie_rolling_stats(
    df: pd.DataFrame,
    goalie_df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Add rolling goalie statistics to the game DataFrame.

    For each game, computes the starting goalie's rolling save percentage
    and goals against average based on their previous games.

    This implementation uses vectorized pandas operations (groupby/transform)
    instead of row-by-row iteration for significantly better performance.

    Parameters
    ----------
    df : pd.DataFrame
        Game data with gamePk, date, homeTeam, awayTeam columns.
    goalie_df : pd.DataFrame
        Goalie data from fetch_goalies_for_games().
    window : int
        Number of previous games to use for rolling stats.

    Returns
    -------
    pd.DataFrame
        Original df with added goalie feature columns.
    """
    if window is None:
        window = config.features.goalie_window

    # Merge goalie data
    df = df.merge(goalie_df, on="gamePk", how="left")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Compute rolling stats for each side using vectorized operations
    home_stats = _compute_goalie_rolling_for_side(df, "home", window)
    away_stats = _compute_goalie_rolling_for_side(df, "away", window)

    # Merge rolling stats back to main dataframe
    df = df.merge(home_stats, on="gamePk", how="left")
    df = df.merge(away_stats, on="gamePk", how="left")

    return df


def add_goalie_features(
    df: pd.DataFrame,
    goalie_cache_path: Path | None = None,
    fetch_missing: bool = True,
    delay: float | None = None,
) -> pd.DataFrame:
    """Main entry point: add goalie features to game data.

    Parameters
    ----------
    df : pd.DataFrame
        Game data with gamePk column.
    goalie_cache_path : Path, optional
        Path to cache goalie data. Defaults to config path.
    fetch_missing : bool
        If True, fetch missing goalie data from API.
    delay : float, optional
        Delay between API requests. Defaults to config value.

    Returns
    -------
    pd.DataFrame
        Game data with goalie features added.
    """
    if goalie_cache_path is None:
        goalie_cache_path = config.data.goalie_cache_path

    game_ids = df["gamePk"].tolist()

    if fetch_missing:
        goalie_df = fetch_goalies_for_games(
            game_ids,
            delay=delay,
            cache_path=goalie_cache_path,
        )
    else:
        if goalie_cache_path.exists():
            goalie_df = pd.read_csv(goalie_cache_path)
        else:
            logger.warning("No goalie cache found and fetch_missing=False")
            return df

    if goalie_df.empty:
        logger.warning("No goalie data available")
        return df

    return compute_goalie_rolling_stats(df, goalie_df)
