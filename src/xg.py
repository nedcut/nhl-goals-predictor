"""
MoneyPuck xG ingestion helpers with local caching.

This module standardizes historical expected-goals data to a compact schema:
    season, date, homeTeam, awayTeam, home_xG, away_xG
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import config
from .logging_config import get_logger

logger = get_logger(__name__)


TEAM_NAME_NORMALIZATION = {
    "anaheim ducks": "Anaheim Ducks",
    "arizona coyotes": "Arizona Coyotes",
    "boston bruins": "Boston Bruins",
    "buffalo sabres": "Buffalo Sabres",
    "calgary flames": "Calgary Flames",
    "carolina hurricanes": "Carolina Hurricanes",
    "chicago blackhawks": "Chicago Blackhawks",
    "colorado avalanche": "Colorado Avalanche",
    "columbus blue jackets": "Columbus Blue Jackets",
    "dallas stars": "Dallas Stars",
    "detroit red wings": "Detroit Red Wings",
    "edmonton oilers": "Edmonton Oilers",
    "florida panthers": "Florida Panthers",
    "los angeles kings": "Los Angeles Kings",
    "la kings": "Los Angeles Kings",
    "minnesota wild": "Minnesota Wild",
    "montreal canadiens": "Montreal Canadiens",
    "montréal canadiens": "Montreal Canadiens",
    "nashville predators": "Nashville Predators",
    "new jersey devils": "New Jersey Devils",
    "new york islanders": "New York Islanders",
    "new york rangers": "New York Rangers",
    "ottawa senators": "Ottawa Senators",
    "philadelphia flyers": "Philadelphia Flyers",
    "pittsburgh penguins": "Pittsburgh Penguins",
    "san jose sharks": "San Jose Sharks",
    "seattle kraken": "Seattle Kraken",
    "st louis blues": "St. Louis Blues",
    "st. louis blues": "St. Louis Blues",
    "tampa bay lightning": "Tampa Bay Lightning",
    "toronto maple leafs": "Toronto Maple Leafs",
    "utah hockey club": "Utah Hockey Club",
    "vancouver canucks": "Vancouver Canucks",
    "vegas golden knights": "Vegas Golden Knights",
    "washington capitals": "Washington Capitals",
    "winnipeg jets": "Winnipeg Jets",
}


def normalize_team_name(name: str) -> str:
    """Normalize team names from external feeds to NHL API names."""
    cleaned = " ".join(str(name).replace(".", ". ").split()).strip()
    key = cleaned.lower().replace("  ", " ")
    key = key.replace(". ", ".").replace("st louis", "st. louis")
    return TEAM_NAME_NORMALIZATION.get(key, cleaned)


def _first_existing(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"MoneyPuck schema missing {label}. Tried: {candidates}")


def _standardize_xg_schema(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Validate and map MoneyPuck columns to canonical schema."""
    if df.empty:
        return pd.DataFrame(columns=["season", "date", "homeTeam", "awayTeam", "home_xG", "away_xG"])

    season_col = "season" if "season" in df.columns else None
    date_col = _first_existing(df, ["date", "gameDate", "game_date"], "date")
    home_team_col = _first_existing(
        df,
        ["homeTeam", "home_team", "homeTeamName", "home_team_name", "home"],
        "home team",
    )
    away_team_col = _first_existing(
        df,
        ["awayTeam", "away_team", "awayTeamName", "away_team_name", "away"],
        "away team",
    )
    home_xg_col = _first_existing(
        df,
        ["home_xG", "home_xg", "homeExpectedGoals", "home_xGoals", "xGoalsForHome"],
        "home xG",
    )
    away_xg_col = _first_existing(
        df,
        ["away_xG", "away_xg", "awayExpectedGoals", "away_xGoals", "xGoalsForAway"],
        "away xG",
    )
    season_values = (
        df[season_col].astype(str).fillna(season)
        if season_col is not None
        else pd.Series([season] * len(df), index=df.index)
    )

    out = pd.DataFrame(
        {
            "season": season_values,
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "homeTeam": df[home_team_col].astype(str).map(normalize_team_name),
            "awayTeam": df[away_team_col].astype(str).map(normalize_team_name),
            "home_xG": pd.to_numeric(df[home_xg_col], errors="coerce"),
            "away_xG": pd.to_numeric(df[away_xg_col], errors="coerce"),
        }
    )
    out["season"] = out["season"].replace("nan", season).fillna(season)
    out = out.dropna(subset=["date", "homeTeam", "awayTeam", "home_xG", "away_xG"]).copy()
    out["season"] = out["season"].astype(str)
    return out


def _xg_cache_path(season: str, cache_dir: Path | None = None) -> Path:
    cache_root = cache_dir or config.data.xg_cache_dir
    return cache_root / f"{season}.csv"


def load_cached_xg_season(season: str, cache_dir: Path | None = None) -> pd.DataFrame | None:
    path = _xg_cache_path(season, cache_dir)
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    return _standardize_xg_schema(cached, season=season)


def fetch_xg_season(season: str) -> pd.DataFrame:
    """Fetch one season from MoneyPuck and standardize schema."""
    url = config.data.xg_url_template.format(season=season)
    logger.info("Fetching MoneyPuck xG for %s from %s", season, url)
    response = requests.get(url, timeout=config.data.xg_request_timeout)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text))
    return _standardize_xg_schema(raw, season=season)


def save_xg_cache(df: pd.DataFrame, season: str, cache_dir: Path | None = None) -> None:
    cache_root = cache_dir or config.data.xg_cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    path = _xg_cache_path(season, cache_root)
    df.to_csv(path, index=False)


def load_xg_games(seasons: Iterable[str], *, use_cache: bool = True) -> pd.DataFrame:
    """Load standardized game-level xG rows for one or more seasons."""
    season_list = [str(s) for s in seasons]
    rows: list[pd.DataFrame] = []
    for i, season in enumerate(season_list):
        season_df: pd.DataFrame | None = None
        if use_cache:
            season_df = load_cached_xg_season(season)
        if season_df is None:
            season_df = fetch_xg_season(season)
            if use_cache:
                save_xg_cache(season_df, season)
            if i < len(season_list) - 1:
                time.sleep(config.data.xg_request_delay)
        rows.append(season_df)
    if not rows:
        return pd.DataFrame(columns=["season", "date", "homeTeam", "awayTeam", "home_xG", "away_xG"])
    return pd.concat(rows, ignore_index=True)


def build_xg_team_log(seasons: Iterable[str], *, use_cache: bool = True) -> pd.DataFrame:
    """Return team-game xG rows used for rolling feature generation."""
    xg_games = load_xg_games(seasons, use_cache=use_cache)
    if xg_games.empty:
        return pd.DataFrame(columns=["season", "date", "team", "opponent", "xGF", "xGA"])

    home = xg_games[["season", "date", "homeTeam", "awayTeam", "home_xG", "away_xG"]].copy()
    home.columns = ["season", "date", "team", "opponent", "xGF", "xGA"]

    away = xg_games[["season", "date", "awayTeam", "homeTeam", "away_xG", "home_xG"]].copy()
    away.columns = ["season", "date", "team", "opponent", "xGF", "xGA"]

    team_log = pd.concat([home, away], ignore_index=True)
    team_log["date"] = pd.to_datetime(team_log["date"]).dt.strftime("%Y-%m-%d")
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)
    return team_log
