"""
Utilities for downloading NHL game logs using the NHL Web API.

This module defines functions to fetch schedules and final scores for NHL games
from the public API at https://api-web.nhle.com. It includes regular season
and playoff games.

If run as a script (`python -m src.data`), it will download all games for the
specified seasons and save the combined dataset as a CSV file.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

# API base URL (new NHL API as of 2023)
API_BASE = "https://api-web.nhle.com/v1"

# Default cache directory
CACHE_DIR = Path("data/raw")

# Season start dates (approximate - October of first year)
SEASON_START_MONTH = 10
SEASON_END_MONTH = 6  # June of second year (playoffs end)


def _season_to_years(season: str) -> tuple[int, int]:
    """Convert season string like '20232024' to (2023, 2024)."""
    return int(season[:4]), int(season[4:])


def _get_season_date_range(season: str) -> tuple[datetime, datetime]:
    """Get approximate date range for a season."""
    start_year, end_year = _season_to_years(season)
    start_date = datetime(start_year, SEASON_START_MONTH, 1)
    end_date = datetime(end_year, SEASON_END_MONTH, 30)
    return start_date, end_date


def fetch_schedule_week(date: str) -> List[dict]:
    """Fetch schedule for the week containing the given date.

    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format.

    Returns
    -------
    list of dict
        List of game dictionaries for the week.
    """
    url = f"{API_BASE}/schedule/{date}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for day in data.get("gameWeek", []):
        for game in day.get("games", []):
            # Only include completed games
            if game.get("gameState") in ("OFF", "FINAL"):
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
                })
    return games


def fetch_season_games(season: str, delay: float = 0.2) -> pd.DataFrame:
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
            # Skip weeks with errors (e.g., lockout periods)
            pass

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


def save_season_cache(df: pd.DataFrame, season: str, cache_dir: Path = CACHE_DIR) -> None:
    """Save a season's data to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(season, cache_dir)
    df.to_csv(path, index=False)
    print(f"Cached {len(df)} games to {path}")


def build_dataset(
    seasons: Iterable[str],
    delay: float = 0.2,
    use_cache: bool = True
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
    all_results: List[pd.DataFrame] = []

    for season in seasons:
        # Check cache first
        if use_cache:
            cached = load_cached_season(season)
            if cached is not None:
                print(f"Loaded {len(cached)} games from cache for {season}")
                all_results.append(cached)
                continue

        # Fetch from API
        df = fetch_season_games(season, delay=delay)

        if df.empty:
            print(f"No games found for {season}")
            continue

        print(f"Fetched {len(df)} games for {season}")

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
