"""
MoneyPuck xG ingestion helpers with local caching.

MoneyPuck's public game-by-game team feed is a single bulk CSV
(``careers/gameByGame/all_teams.csv``) with one row per team per game per
situation. This module filters to ``situation == "all"``, maps abbreviations
to NHL Web API team names, converts MoneyPuck's start-year season ids to the
pipeline's 8-digit season format, and caches a compact schema:

    season, date, homeTeam, awayTeam, home_xG, away_xG
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import config
from .http_client import get_text
from .logging_config import get_logger

logger = get_logger(__name__)


# Canonical full names used for joins with NHL Web API game logs.
TEAM_NAME_NORMALIZATION = {
    "anaheim ducks": "Anaheim Ducks",
    "arizona coyotes": "Arizona Coyotes",
    "atlanta thrashers": "Atlanta Thrashers",
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
    "montreal canadiens": "Montréal Canadiens",
    "montréal canadiens": "Montréal Canadiens",
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
    # Utah franchise renames / NHL API naming quirks → one join key.
    "utah hockey club": "Utah Hockey Club",
    "utah utah hockey club": "Utah Hockey Club",
    "utah mammoth": "Utah Hockey Club",
    "vancouver canucks": "Vancouver Canucks",
    "vegas golden knights": "Vegas Golden Knights",
    "washington capitals": "Washington Capitals",
    "winnipeg jets": "Winnipeg Jets",
}

# MoneyPuck uses short codes (and a few legacy dotted codes).
TEAM_ABBREV_TO_NAME = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "ATL": "Atlanta Thrashers",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "L.A": "Los Angeles Kings",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montréal Canadiens",
    "N.J": "New Jersey Devils",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "S.J": "San Jose Sharks",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "T.B": "Tampa Bay Lightning",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}

_EMPTY_GAME_COLS = ["season", "date", "homeTeam", "awayTeam", "home_xG", "away_xG"]
_EMPTY_TEAM_COLS = ["season", "date", "team", "opponent", "xGF", "xGA"]


def normalize_team_name(name: str) -> str:
    """Normalize team names from external feeds / NHL API to a join key."""
    cleaned = " ".join(str(name).replace(".", ". ").split()).strip()
    key = cleaned.lower().replace("  ", " ")
    key = key.replace(". ", ".").replace("st louis", "st. louis")
    if key in TEAM_NAME_NORMALIZATION:
        return TEAM_NAME_NORMALIZATION[key]
    # MoneyPuck abbreviations (and any dotted variants).
    abbrev = str(name).strip().upper()
    if abbrev in TEAM_ABBREV_TO_NAME:
        return TEAM_ABBREV_TO_NAME[abbrev]
    return cleaned


def nhl_season_to_moneypuck_year(season: str) -> int:
    """Map pipeline season id ``20232024`` → MoneyPuck start year ``2023``."""
    s = str(season).strip()
    if len(s) == 8 and s.isdigit():
        return int(s[:4])
    return int(s)


def moneypuck_year_to_nhl_season(year: int | str) -> str:
    """Map MoneyPuck start year ``2023`` → pipeline season id ``20232024``."""
    y = int(year)
    return f"{y}{y + 1}"


def _parse_moneypuck_date(series: pd.Series) -> pd.Series:
    """Parse MoneyPuck ``gameDate`` (int YYYYMMDD or ISO strings) to YYYY-MM-DD."""
    if pd.api.types.is_numeric_dtype(series):
        as_str = series.astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
        return pd.to_datetime(as_str, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _first_existing(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"MoneyPuck schema missing {label}. Tried: {candidates}")


def _standardize_game_level_schema(df: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Validate and map already game-level columns to the canonical schema.

    Used for per-season caches written by this module and for unit-test fixtures
    that supply home/away xG directly.
    """
    if df.empty:
        return pd.DataFrame(columns=_EMPTY_GAME_COLS)

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
    default_season = season or ""
    season_values = (
        df[season_col].astype(str).fillna(default_season)
        if season_col is not None
        else pd.Series([default_season] * len(df), index=df.index)
    )
    # MoneyPuck bulk rows may carry a 4-digit start year; normalize to 8-digit.
    season_values = season_values.map(
        lambda s: (
            moneypuck_year_to_nhl_season(s) if str(s).isdigit() and len(str(s)) <= 4 else str(s)
        )
    )

    dates = df[date_col]
    if date_col == "gameDate" or pd.api.types.is_numeric_dtype(dates):
        date_str = _parse_moneypuck_date(dates)
    else:
        date_str = pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m-%d")

    out = pd.DataFrame(
        {
            "season": season_values,
            "date": date_str,
            "homeTeam": df[home_team_col].astype(str).map(normalize_team_name),
            "awayTeam": df[away_team_col].astype(str).map(normalize_team_name),
            "home_xG": pd.to_numeric(df[home_xg_col], errors="coerce"),
            "away_xG": pd.to_numeric(df[away_xg_col], errors="coerce"),
        }
    )
    if season:
        out["season"] = out["season"].replace("nan", season).fillna(season)
    out = out.dropna(subset=["date", "homeTeam", "awayTeam", "home_xG", "away_xG"]).copy()
    out["season"] = out["season"].astype(str)
    return out


def _is_moneypuck_team_game_feed(df: pd.DataFrame) -> bool:
    """True when the frame looks like MoneyPuck's all_teams game-by-game CSV."""
    cols = set(df.columns)
    return {"xGoalsFor", "home_or_away", "gameDate"}.issubset(cols) or {
        "xGoalsFor",
        "home_or_away",
        "gameId",
    }.issubset(cols)


def _standardize_team_game_feed(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MoneyPuck team-game rows into canonical game-level xG rows."""
    if df.empty:
        return pd.DataFrame(columns=_EMPTY_GAME_COLS)

    work = df.copy()
    if "situation" in work.columns:
        work = work[work["situation"].astype(str).str.lower() == "all"].copy()
    if work.empty:
        return pd.DataFrame(columns=_EMPTY_GAME_COLS)

    team_col = _first_existing(work, ["team", "playerTeam", "name"], "team")
    opp_col = _first_existing(work, ["opposingTeam", "opponent", "opp"], "opponent")
    xgf_col = _first_existing(work, ["xGoalsFor", "xGoals_for", "xGF"], "xGoalsFor")
    side_col = _first_existing(work, ["home_or_away", "homeOrAway", "HoA"], "home_or_away")
    date_col = _first_existing(work, ["gameDate", "date", "game_date"], "date")
    game_id_col = "gameId" if "gameId" in work.columns else None

    work["_team"] = work[team_col].astype(str).map(normalize_team_name)
    work["_opp"] = work[opp_col].astype(str).map(normalize_team_name)
    work["_xGF"] = pd.to_numeric(work[xgf_col], errors="coerce")
    work["_side"] = work[side_col].astype(str).str.upper().str.strip()
    work["_date"] = (
        _parse_moneypuck_date(work[date_col])
        if date_col == "gameDate" or pd.api.types.is_numeric_dtype(work[date_col])
        else pd.to_datetime(work[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    )

    if "season" in work.columns:
        work["_season"] = work["season"].map(moneypuck_year_to_nhl_season)
    else:
        # Infer season from date (Oct–Dec → start year, Jan–Jun → start year - 1).
        dt = pd.to_datetime(work["_date"], errors="coerce")
        start_year = dt.dt.year.where(dt.dt.month >= 8, dt.dt.year - 1)
        work["_season"] = start_year.map(
            lambda y: moneypuck_year_to_nhl_season(int(y)) if pd.notna(y) else None
        )

    home = work[work["_side"] == "HOME"].copy()
    away = work[work["_side"] == "AWAY"].copy()
    if home.empty or away.empty:
        return pd.DataFrame(columns=_EMPTY_GAME_COLS)

    if game_id_col is not None:
        merged = home.merge(
            away[[game_id_col, "_team", "_xGF"]],
            on=game_id_col,
            how="inner",
            suffixes=("_home", "_away"),
        )
        out = pd.DataFrame(
            {
                "season": merged["_season"].astype(str),
                "date": merged["_date"],
                "homeTeam": merged["_team_home"],
                "awayTeam": merged["_team_away"],
                "home_xG": merged["_xGF_home"],
                "away_xG": merged["_xGF_away"],
            }
        )
    else:
        # Fallback: match home/away on date + mutual opponent pairing.
        home = home.rename(columns={"_team": "homeTeam", "_opp": "awayTeam", "_xGF": "home_xG"})
        away = away.rename(columns={"_team": "awayTeam", "_opp": "homeTeam", "_xGF": "away_xG"})
        merged = home.merge(
            away[["_season", "_date", "homeTeam", "awayTeam", "away_xG"]],
            on=["_season", "_date", "homeTeam", "awayTeam"],
            how="inner",
        )
        out = pd.DataFrame(
            {
                "season": merged["_season"].astype(str),
                "date": merged["_date"],
                "homeTeam": merged["homeTeam"],
                "awayTeam": merged["awayTeam"],
                "home_xG": merged["home_xG"],
                "away_xG": merged["away_xG"],
            }
        )

    out = out.dropna(subset=["date", "homeTeam", "awayTeam", "home_xG", "away_xG"]).copy()
    out = out.drop_duplicates(subset=["season", "date", "homeTeam", "awayTeam"], keep="last")
    return out.reset_index(drop=True)


def _standardize_xg_schema(df: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Route MoneyPuck team-game feeds and cached game-level frames to one schema."""
    if df.empty:
        return pd.DataFrame(columns=_EMPTY_GAME_COLS)
    if _is_moneypuck_team_game_feed(df):
        out = _standardize_team_game_feed(df)
        if season is not None:
            out = out[out["season"].astype(str) == str(season)].copy()
        return out
    return _standardize_game_level_schema(df, season=season)


def _xg_cache_path(season: str, cache_dir: Path | None = None) -> Path:
    cache_root = cache_dir or config.data.xg_cache_dir
    return cache_root / f"{season}.csv"


def _bulk_cache_path(cache_dir: Path | None = None) -> Path:
    cache_root = cache_dir or config.data.xg_cache_dir
    return cache_root / "all_teams_game_by_game.csv"


def load_cached_xg_season(season: str, cache_dir: Path | None = None) -> pd.DataFrame | None:
    path = _xg_cache_path(season, cache_dir)
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    return _standardize_xg_schema(cached, season=season)


def fetch_xg_bulk() -> pd.DataFrame:
    """Fetch MoneyPuck's full game-by-game team CSV and standardize to game-level xG."""
    url = config.data.xg_url
    logger.info("Fetching MoneyPuck xG bulk feed from %s", url)
    text = get_text(url, timeout=config.data.xg_request_timeout)
    raw = pd.read_csv(io.StringIO(text))
    return _standardize_xg_schema(raw)


def fetch_xg_season(season: str) -> pd.DataFrame:
    """Fetch xG for one NHL season (loads bulk feed, filters to ``season``)."""
    bulk = fetch_xg_bulk()
    season = str(season)
    out = bulk[bulk["season"].astype(str) == season].copy()
    if out.empty:
        # Allow callers that pass a MoneyPuck start year.
        alt = moneypuck_year_to_nhl_season(nhl_season_to_moneypuck_year(season))
        out = bulk[bulk["season"].astype(str) == alt].copy()
    return out.reset_index(drop=True)


def save_xg_cache(df: pd.DataFrame, season: str, cache_dir: Path | None = None) -> None:
    cache_root = cache_dir or config.data.xg_cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    path = _xg_cache_path(season, cache_root)
    df.to_csv(path, index=False)


def _load_or_fetch_bulk(cache_dir: Path | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    """Return standardized game-level xG for all seasons, using a local bulk cache."""
    path = _bulk_cache_path(cache_dir)
    if use_cache and path.exists():
        cached = pd.read_csv(path)
        # Bulk cache is already game-level; re-standardize for safety.
        return _standardize_xg_schema(cached)

    bulk = fetch_xg_bulk()
    if use_cache and not bulk.empty:
        cache_root = cache_dir or config.data.xg_cache_dir
        cache_root.mkdir(parents=True, exist_ok=True)
        bulk.to_csv(path, index=False)
        logger.info("Cached MoneyPuck bulk xG to %s (%d games)", path, len(bulk))
    return bulk


def load_xg_games(seasons: Iterable[str], *, use_cache: bool = True) -> pd.DataFrame:
    """Load standardized game-level xG rows for one or more seasons."""
    season_list = [str(s) for s in seasons]
    rows: list[pd.DataFrame] = []
    missing: list[str] = []

    for season in season_list:
        season_df: pd.DataFrame | None = None
        if use_cache:
            season_df = load_cached_xg_season(season)
        if season_df is None or season_df.empty:
            missing.append(season)
        else:
            rows.append(season_df)

    if missing:
        # One bulk download covers every missing season.
        bulk = _load_or_fetch_bulk(use_cache=use_cache)
        for season in missing:
            season_df = bulk[bulk["season"].astype(str) == season].copy()
            if season_df.empty:
                # Try converting 8-digit → already 8-digit from bulk; also try start year form.
                mp_year = nhl_season_to_moneypuck_year(season)
                alt = moneypuck_year_to_nhl_season(mp_year)
                season_df = bulk[bulk["season"].astype(str) == alt].copy()
            if use_cache and not season_df.empty:
                save_xg_cache(season_df, season)
            rows.append(season_df)

    if not rows:
        return pd.DataFrame(columns=_EMPTY_GAME_COLS)
    return pd.concat(rows, ignore_index=True)


def build_xg_team_log(seasons: Iterable[str], *, use_cache: bool = True) -> pd.DataFrame:
    """Return team-game xG rows used for rolling feature generation."""
    xg_games = load_xg_games(seasons, use_cache=use_cache)
    if xg_games.empty:
        return pd.DataFrame(columns=_EMPTY_TEAM_COLS)

    home = xg_games[["season", "date", "homeTeam", "awayTeam", "home_xG", "away_xG"]].copy()
    home.columns = ["season", "date", "team", "opponent", "xGF", "xGA"]

    away = xg_games[["season", "date", "awayTeam", "homeTeam", "away_xG", "home_xG"]].copy()
    away.columns = ["season", "date", "team", "opponent", "xGF", "xGA"]

    team_log = pd.concat([home, away], ignore_index=True)
    team_log["date"] = pd.to_datetime(team_log["date"]).dt.strftime("%Y-%m-%d")
    team_log["team"] = team_log["team"].map(normalize_team_name)
    team_log["opponent"] = team_log["opponent"].map(normalize_team_name)
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)
    return team_log
