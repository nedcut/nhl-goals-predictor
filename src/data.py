"""
Utilities for downloading NHL game logs using the NHL Web API.

This module fetches schedules and final scores for NHL games from the public
API at https://api-web.nhle.com. It includes regular season and playoff games.

Data is cached per-season to data/raw/{season}.csv for fast subsequent loads.
First download takes ~20 seconds per season; cached loads are instant.

Usage:
    from src.data import build_dataset

    # Load 4 recent seasons (recommended for best model performance)
    df = build_dataset(['20212022', '20222023', '20232024', '20242025'])

    # Force re-download (skip cache)
    df = build_dataset(['20232024'], use_cache=False)

Columns returned:
    gamePk, season, date, homeTeam, awayTeam, homeScore, awayScore, totalGoals
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from .config import config
from .logging_config import get_logger

logger = get_logger(__name__)

# Use config for API and cache settings
API_BASE = config.data.api_base
CACHE_DIR = config.data.cache_dir

# Season start dates (approximate - October of first year)
SEASON_START_MONTH = config.data.season_start_month
SEASON_END_MONTH = config.data.season_end_month


def season_for_date(as_of: date | datetime | None = None) -> str:
    """Return the NHL season code containing the given date.

    Dates before October belong to the season that started in the prior
    calendar year. For example, June 2026 maps to ``20252026``.
    """
    current = as_of or datetime.now()
    year = current.year
    month = current.month
    start_year = year if month >= SEASON_START_MONTH else year - 1
    return f"{start_year}{start_year + 1}"


def recent_seasons(count: int = 2, as_of: date | datetime | None = None) -> List[str]:
    """Return consecutive recent season codes ending with the active season."""
    if count < 1:
        raise ValueError("count must be at least 1")

    active = season_for_date(as_of)
    active_start = int(active[:4])
    first_start = active_start - count + 1
    return [f"{year}{year + 1}" for year in range(first_start, active_start + 1)]


def _season_to_years(season: str) -> tuple[int, int]:
    """Convert season string like '20232024' to (2023, 2024)."""
    return int(season[:4]), int(season[4:])


def _get_season_date_range(season: str) -> tuple[datetime, datetime]:
    """Get approximate date range for a season."""
    start_year, end_year = _season_to_years(season)
    start_date = datetime(start_year, SEASON_START_MONTH, 1)
    end_date = datetime(end_year, SEASON_END_MONTH, 30)
    return start_date, end_date


def fetch_schedule_week(date: str, include_upcoming: bool = False) -> List[dict]:
    """Fetch schedule for the week containing the given date.

    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format.
    include_upcoming : bool
        If True, include scheduled/upcoming games (not just completed).

    Returns
    -------
    list of dict
        List of game dictionaries for the week.
    """
    url = f"{API_BASE}/schedule/{date}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Game states: OFF/FINAL = completed, FUT = future, PRE = pre-game, LIVE = in progress
    completed_states = ("OFF", "FINAL")
    upcoming_states = ("FUT", "PRE", "LIVE")

    games = []
    for day in data.get("gameWeek", []):
        for game in day.get("games", []):
            game_state = game.get("gameState")
            is_completed = game_state in completed_states
            is_upcoming = game_state in upcoming_states

            if is_completed or (include_upcoming and is_upcoming):
                games.append({
                    "gamePk": game["id"],
                    "season": str(game["season"]),
                    "gameType": "R" if game["gameType"] == 2 else "P" if game["gameType"] == 3 else "O",
                    "date": day["date"],
                    "homeTeam": game["homeTeam"]["placeName"]["default"] + " " + game["homeTeam"]["commonName"]["default"],
                    "awayTeam": game["awayTeam"]["placeName"]["default"] + " " + game["awayTeam"]["commonName"]["default"],
                    "homeScore": game["homeTeam"].get("score", 0),
                    "awayScore": game["awayTeam"].get("score", 0),
                    "totalGoals": game["homeTeam"].get("score", 0) + game["awayTeam"].get("score", 0),
                    "gameState": game_state,
                })
    return games


def fetch_season_games(season: str, delay: float | None = None) -> pd.DataFrame:
    """Fetch all games for a single NHL season.

    Parameters
    ----------
    season : str
        Season in format "YYYYYYYY", e.g. "20232024".
    delay : float
        Delay between API requests in seconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per game.
    """
    if delay is None:
        delay = config.data.request_delay

    start_date, end_date = _get_season_date_range(season)

    all_games = []
    seen_ids: Set[int] = set()

    current_date = start_date
    pbar = tqdm(desc=f"Fetching {season}", unit="week")

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            games = fetch_schedule_week(date_str)
            for game in games:
                # Only include games from this season, avoid duplicates
                if game["season"] == season and game["gamePk"] not in seen_ids:
                    all_games.append(game)
                    seen_ids.add(game["gamePk"])
            pbar.update(1)
        except requests.RequestException as e:
            # Log the error but continue (some weeks may fail during lockouts)
            logger.warning("Failed to fetch schedule for %s: %s", date_str, e)

        # Move forward 7 days (schedule endpoint returns a week)
        current_date += timedelta(days=7)
        time.sleep(delay)

    pbar.close()

    if not all_games:
        return pd.DataFrame()

    df = pd.DataFrame(all_games)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_cache_path(season: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Get the cache file path for a season."""
    return cache_dir / f"{season}.csv"


def load_cached_season(season: str, cache_dir: Path = CACHE_DIR) -> Optional[pd.DataFrame]:
    """Load a cached season if it exists."""
    path = get_cache_path(season, cache_dir)
    if path.exists():
        return pd.read_csv(path)
    return None


def _active_season_cache_is_fresh(
    season: str,
    cache_dir: Path = CACHE_DIR,
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether an active-season cache is recent enough for inference."""
    path = get_cache_path(season, cache_dir)
    if not path.exists():
        return False

    current = now or datetime.now()
    age_seconds = current.timestamp() - path.stat().st_mtime
    max_age_seconds = config.data.active_season_cache_ttl_hours * 60 * 60
    return age_seconds <= max_age_seconds


def save_season_cache(df: pd.DataFrame, season: str, cache_dir: Path = CACHE_DIR) -> None:
    """Save a season's data to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(season, cache_dir)
    df.to_csv(path, index=False)
    logger.info("Cached %d games to %s", len(df), path)


def build_dataset(
    seasons: Iterable[str],
    delay: float | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download game data for one or more seasons.

    Parameters
    ----------
    seasons : iterable of str
        Season codes (e.g. ["20222023", "20232024"]).
    delay : float
        Delay between API requests in seconds.
    use_cache : bool
        If True, load cached data when available.

    Returns
    -------
    pd.DataFrame
        Combined schedule and results for all seasons.
    """
    if delay is None:
        delay = config.data.request_delay

    all_results: List[pd.DataFrame] = []

    for season in seasons:
        # Check cache first
        if use_cache:
            cached = load_cached_season(season)
            if cached is not None:
                is_active = season == season_for_date()
                if not is_active or _active_season_cache_is_fresh(season):
                    logger.info("Loaded %d games from cache for %s", len(cached), season)
                    all_results.append(cached)
                    continue
                logger.info("Refreshing stale active-season cache for %s", season)

        # Fetch from API
        df = fetch_season_games(season, delay=delay)

        if df.empty:
            logger.warning("No games found for %s", season)
            continue

        logger.info("Fetched %d games for %s", len(df), season)

        # Cache for future use
        if use_cache:
            save_season_cache(df, season)

        all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download NHL game logs via the NHL Web API."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="List of seasons to download, e.g. 20222023 20232024",
    )
    parser.add_argument(
        "--out",
        default="data/raw/games.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between requests in seconds.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point when running as a script."""
    args = _parse_args(argv)
    df = build_dataset(args.seasons, delay=args.delay, use_cache=not args.no_cache)

    if df.empty:
        print("No games found.")
        return

    # Save combined output
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} games to {args.out}")


if __name__ == "__main__":
    main()
