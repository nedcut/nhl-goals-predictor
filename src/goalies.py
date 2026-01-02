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
import requests
from tqdm import tqdm

API_BASE = "https://api-web.nhle.com/v1"
GOALIE_CACHE_DIR = Path("data/goalies")


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
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

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

        # If no starter flag, use goalie with most TOI
        if starter is None and goalies:
            starter = max(goalies, key=lambda x: x.get("toi", "00:00"))

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
    delay: float = 0.05,
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
    # Check cache
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached_ids = set(cached["gamePk"].tolist())
        game_ids = [g for g in game_ids if g not in cached_ids]
        if not game_ids:
            return cached
        print(f"Found {len(cached)} cached, fetching {len(game_ids)} new games")
    else:
        cached = pd.DataFrame()

    results = []
    failed = []

    for game_id in tqdm(game_ids, desc="Fetching goalie data"):
        try:
            data = fetch_game_goalies(game_id)
            results.append(data)
        except Exception as e:
            failed.append(game_id)
        time.sleep(delay)

    if failed:
        print(f"Failed to fetch {len(failed)} games")

    if results:
        new_df = pd.DataFrame(results)
        combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df

        # Save cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(cache_path, index=False)
            print(f"Cached {len(combined)} games to {cache_path}")

        return combined

    return cached if not cached.empty else pd.DataFrame()


def compute_goalie_rolling_stats(
    df: pd.DataFrame,
    goalie_df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Add rolling goalie statistics to the game DataFrame.

    For each game, computes the starting goalie's rolling save percentage
    and goals against average based on their previous games.

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
    # Merge goalie data
    df = df.merge(goalie_df, on="gamePk", how="left")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Track each goalie's cumulative stats
    goalie_history: Dict[int, List[Dict]] = {}

    # New columns to add
    home_rolling_sv = []
    away_rolling_sv = []
    home_rolling_gaa = []
    away_rolling_gaa = []

    for _, row in df.iterrows():
        for side, sv_list, gaa_list in [
            ("home", home_rolling_sv, home_rolling_gaa),
            ("away", away_rolling_sv, away_rolling_gaa),
        ]:
            goalie_id = row.get(f"{side}_goalie_id")

            if pd.isna(goalie_id) or goalie_id is None:
                sv_list.append(np.nan)
                gaa_list.append(np.nan)
                continue

            goalie_id = int(goalie_id)
            history = goalie_history.get(goalie_id, [])

            # Compute rolling stats from history
            if len(history) >= 1:
                recent = history[-window:]
                total_saves = sum(g["saves"] for g in recent)
                total_shots = sum(g["shots"] for g in recent)
                total_goals = sum(g["goals_against"] for g in recent)
                games_played = len(recent)

                sv_pct = total_saves / total_shots if total_shots > 0 else 0.9
                gaa = (total_goals / games_played) if games_played > 0 else 3.0

                sv_list.append(sv_pct)
                gaa_list.append(gaa)
            else:
                # No history yet - use league average
                sv_list.append(np.nan)
                gaa_list.append(np.nan)

            # Update history with this game
            saves = row.get(f"{side}_saves", 0)
            shots = row.get(f"{side}_shots_against", 0)
            goals = row.get(f"{side}_goals_against", 0)

            if not pd.isna(saves):
                goalie_history.setdefault(goalie_id, []).append({
                    "saves": int(saves),
                    "shots": int(shots),
                    "goals_against": int(goals),
                })

    df["home_goalie_sv_pct"] = home_rolling_sv
    df["away_goalie_sv_pct"] = away_rolling_sv
    df["home_goalie_gaa"] = home_rolling_gaa
    df["away_goalie_gaa"] = away_rolling_gaa

    return df


def add_goalie_features(
    df: pd.DataFrame,
    goalie_cache_path: Path = GOALIE_CACHE_DIR / "goalie_stats.csv",
    fetch_missing: bool = True,
    delay: float = 0.05,
) -> pd.DataFrame:
    """Main entry point: add goalie features to game data.

    Parameters
    ----------
    df : pd.DataFrame
        Game data with gamePk column.
    goalie_cache_path : Path
        Path to cache goalie data.
    fetch_missing : bool
        If True, fetch missing goalie data from API.
    delay : float
        Delay between API requests.

    Returns
    -------
    pd.DataFrame
        Game data with goalie features added.
    """
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
            print("No goalie cache found and fetch_missing=False")
            return df

    if goalie_df.empty:
        print("No goalie data available")
        return df

    return compute_goalie_rolling_stats(df, goalie_df, window=10)
