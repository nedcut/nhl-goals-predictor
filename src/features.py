"""
Feature engineering utilities for NHL total goals prediction.

The functions in this module operate on the raw game logs produced by
`src.data.build_dataset`. They generate rolling averages and other features
that capture team performance, fatigue, and momentum.

Features generated (40+ total with multi-window enabled):
- Rolling goals for/against averages at multiple windows (5, 10, 20, 40 games)
- Rest days and back-to-back indicators (home_rest_days, home_is_back_to_back)
- Win streaks (home_win_streak, away_win_streak)
- Win percentages (home_win_pct, away_win_pct)
- Games played in season (home_games_played, away_games_played)
- Goalie rolling stats (home_goalie_sv_pct, home_goalie_gaa, etc.)
- Interaction features (matchup dynamics like scoring_opportunity, opponent_threat)
- Temporal features (month, days_into_season)

Usage:
    from src.data import build_dataset
    from src.features import add_features

    df = build_dataset(['20232024', '20242025'])
    df_features = add_features(df, include_goalies=True)
"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from .config import config
from .logging_config import get_logger
from .validation import validate_game_data

logger = get_logger(__name__)


def _build_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Convert game-level data to team-game-level data.

    Each game becomes two rows: one for the home team, one for the away team.
    This makes it easier to compute rolling stats per team.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Home team perspective
    home = df[["gamePk", "season", "date", "homeTeam", "homeScore", "awayScore"]].copy()
    home.columns = ["gamePk", "season", "date", "team", "goals_for", "goals_against"]
    home["is_home"] = True
    home["opponent"] = df["awayTeam"]

    # Away team perspective
    away = df[["gamePk", "season", "date", "awayTeam", "awayScore", "homeScore"]].copy()
    away.columns = ["gamePk", "season", "date", "team", "goals_for", "goals_against"]
    away["is_home"] = False
    away["opponent"] = df["homeTeam"]

    # Combine and sort
    team_log = pd.concat([home, away], ignore_index=True)
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)

    # Add derived columns
    team_log["win"] = (team_log["goals_for"] > team_log["goals_against"]).astype(int)
    team_log["total_goals"] = team_log["goals_for"] + team_log["goals_against"]

    return team_log


def _compute_rolling_features(team_log: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling features for each team.

    Uses vectorized groupby operations for performance.
    All features use shift(1) to ensure we only use past data (no leakage).
    """
    team_log = team_log.copy()

    # Group by team for rolling calculations
    grouped = team_log.groupby("team")

    # Rolling averages (shifted to prevent leakage)
    team_log["avg_GF"] = grouped["goals_for"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    team_log["avg_GA"] = grouped["goals_against"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    team_log["avg_total"] = grouped["total_goals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Win percentage (rolling)
    team_log["win_pct"] = grouped["win"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Games played this season
    team_log["games_played"] = grouped.cumcount()

    # Rest days (days since last game)
    team_log["prev_game_date"] = grouped["date"].shift(1)
    team_log["rest_days"] = (team_log["date"] - team_log["prev_game_date"]).dt.days
    team_log["is_back_to_back"] = (team_log["rest_days"] == 1).astype(int)

    # Win streak (positive = winning streak, negative = losing streak)
    def calc_streak(wins: pd.Series) -> pd.Series:
        """Calculate current streak based on previous games."""
        shifted = wins.shift(1)
        streak = pd.Series(0, index=wins.index)
        current_streak = 0
        for i, val in enumerate(shifted):
            if pd.isna(val):
                current_streak = 0
            elif val == 1:
                current_streak = max(1, current_streak + 1)
            else:
                current_streak = min(-1, current_streak - 1)
            streak.iloc[i] = current_streak
        return streak

    team_log["win_streak"] = grouped["win"].transform(calc_streak)

    # Home/away specific performance (rolling window)
    # For home games, track home win pct; for away games, track away win pct
    team_log["home_win"] = team_log["win"] * team_log["is_home"].astype(int)
    team_log["away_win"] = team_log["win"] * (~team_log["is_home"]).astype(int)
    team_log["home_game"] = team_log["is_home"].astype(int)
    team_log["away_game"] = (~team_log["is_home"]).astype(int)

    # Rolling home/away performance
    team_log["home_wins_roll"] = grouped["home_win"].transform(
        lambda x: x.shift(1).rolling(window * 2, min_periods=1).sum()
    )
    team_log["home_games_roll"] = grouped["home_game"].transform(
        lambda x: x.shift(1).rolling(window * 2, min_periods=1).sum()
    )
    team_log["away_wins_roll"] = grouped["away_win"].transform(
        lambda x: x.shift(1).rolling(window * 2, min_periods=1).sum()
    )
    team_log["away_games_roll"] = grouped["away_game"].transform(
        lambda x: x.shift(1).rolling(window * 2, min_periods=1).sum()
    )

    # Compute percentages (avoid division by zero)
    team_log["home_win_pct"] = np.where(
        team_log["home_games_roll"] > 0,
        team_log["home_wins_roll"] / team_log["home_games_roll"],
        0.5
    )
    team_log["away_win_pct"] = np.where(
        team_log["away_games_roll"] > 0,
        team_log["away_wins_roll"] / team_log["away_games_roll"],
        0.5
    )

    # Clean up intermediate columns
    drop_cols = ["home_win", "away_win", "home_game", "away_game",
                 "home_wins_roll", "home_games_roll", "away_wins_roll",
                 "away_games_roll", "prev_game_date"]
    team_log = team_log.drop(columns=drop_cols)

    return team_log


def _compute_h2h_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute head-to-head historical features.

    For each game, computes the rolling average total goals in previous
    matchups between the same two teams.

    Parameters
    ----------
    df : pd.DataFrame
        Game data with homeTeam, awayTeam, date, totalGoals columns.
    window : int
        Number of previous matchups to use for rolling average.

    Returns
    -------
    pd.DataFrame
        Original df with h2h_avg_goals column added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Create a consistent matchup key (alphabetical order)
    df["matchup"] = df.apply(
        lambda r: tuple(sorted([r["homeTeam"], r["awayTeam"]])),
        axis=1,
    )

    # Compute rolling average goals for each matchup
    # Use shift(1) to avoid data leakage
    df["h2h_avg_goals"] = df.groupby("matchup")["totalGoals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Drop intermediate column
    df = df.drop(columns=["matchup"])

    return df


def _compute_venue_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute venue-specific features.

    For each game, computes the rolling average total goals at the home team's
    venue (arena) based on previous games there.

    Parameters
    ----------
    df : pd.DataFrame
        Game data with homeTeam, date, totalGoals columns.
    window : int
        Number of previous games at venue to use for rolling average.

    Returns
    -------
    pd.DataFrame
        Original df with venue_avg_goals column added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Use homeTeam as proxy for venue (each team has one home arena)
    df["venue_avg_goals"] = df.groupby("homeTeam")["totalGoals"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    return df


def _compute_multi_window_rolling(
    team_log: pd.DataFrame,
    windows: Tuple[int, ...] = (5, 10, 20, 40),
) -> pd.DataFrame:
    """Compute rolling features for multiple time windows.

    Different windows capture different signals:
    - 5 games: Recent form/hot streaks
    - 10 games: Short-term trends
    - 20 games: Medium-term baseline
    - 40 games: Season-level ability

    Parameters
    ----------
    team_log : pd.DataFrame
        Team-game level data from _build_team_game_log().
    windows : tuple of int
        Window sizes to compute features for.

    Returns
    -------
    pd.DataFrame
        Team log with multi-window rolling features added.
    """
    team_log = team_log.copy()
    grouped = team_log.groupby("team")

    for w in windows:
        suffix = f"_{w}g"

        # Goals for/against rolling averages
        team_log[f"avg_GF{suffix}"] = grouped["goals_for"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        team_log[f"avg_GA{suffix}"] = grouped["goals_against"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        team_log[f"avg_total{suffix}"] = grouped["total_goals"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

        # Win percentage
        team_log[f"win_pct{suffix}"] = grouped["win"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

        # Goals standard deviation (scoring volatility)
        team_log[f"std_GF{suffix}"] = grouped["goals_for"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std()
        )

    return team_log


def _add_xg_rolling_features(
    team_log: pd.DataFrame,
    *,
    seasons: Tuple[str, ...],
    xg_windows: Tuple[int, ...],
) -> pd.DataFrame:
    """Join team-level xG rows and compute no-leakage rolling xG features."""
    from .xg import build_xg_team_log

    team_log = team_log.copy()
    xg_log = build_xg_team_log(seasons, use_cache=True)
    if xg_log.empty:
        return team_log

    xg_log = xg_log.copy()
    xg_log["season"] = xg_log["season"].astype(str)
    xg_log["date"] = pd.to_datetime(xg_log["date"]).dt.strftime("%Y-%m-%d")

    team_log["season"] = team_log["season"].astype(str)
    team_log["date"] = pd.to_datetime(team_log["date"]).dt.strftime("%Y-%m-%d")

    team_log = team_log.merge(
        xg_log,
        on=["season", "date", "team", "opponent"],
        how="left",
    )

    grouped = team_log.groupby("team")
    for w in xg_windows:
        suffix = f"_{w}g"
        team_log[f"avg_xGF{suffix}"] = grouped["xGF"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        team_log[f"avg_xGA{suffix}"] = grouped["xGA"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    return team_log


def _compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction features capturing matchup dynamics.

    These features capture how team strengths/weaknesses interact:
    - scoring_opportunity: High offense vs weak defense
    - opponent_threat: Opponent offense vs our defense
    - rest_advantage: Relative rest between teams
    - form_diff: Difference in recent win percentage

    Parameters
    ----------
    df : pd.DataFrame
        Game data with rolling features already added.

    Returns
    -------
    pd.DataFrame
        Data with interaction features added.
    """
    df = df.copy()

    # Scoring opportunity: home offense vs away defense
    if "home_avg_GF" in df.columns and "away_avg_GA" in df.columns:
        df["scoring_opportunity"] = df["home_avg_GF"] * df["away_avg_GA"]
        df["opponent_threat"] = df["away_avg_GF"] * df["home_avg_GA"]

    # Rest advantage (positive = home team more rested)
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
        # Capped rest days difference (extreme values less meaningful)
        df["rest_advantage_capped"] = df["rest_advantage"].clip(-5, 5)

    # Form difference (recent performance gap)
    if "home_win_pct" in df.columns and "away_win_pct" in df.columns:
        df["form_diff"] = df["home_win_pct"] - df["away_win_pct"]

    # Combined team totals (expected game total based on both teams)
    if "home_avg_total" in df.columns and "away_avg_total" in df.columns:
        df["combined_avg_total"] = (df["home_avg_total"] + df["away_avg_total"]) / 2

    # Multi-window interactions (if available)
    for w in [5, 10, 20, 40]:
        suffix = f"_{w}g"
        home_gf = f"home_avg_GF{suffix}"
        away_ga = f"away_avg_GA{suffix}"
        away_gf = f"away_avg_GF{suffix}"
        home_ga = f"home_avg_GA{suffix}"

        if home_gf in df.columns and away_ga in df.columns:
            df[f"scoring_opp{suffix}"] = df[home_gf] * df[away_ga]
            df[f"opp_threat{suffix}"] = df[away_gf] * df[home_ga]

    # Goalie interaction features (if goalie data present)
    if "home_goalie_sv_pct" in df.columns and "away_avg_GF" in df.columns:
        # Goalie quality vs opponent offense
        df["home_goalie_vs_offense"] = df["home_goalie_sv_pct"] * df["away_avg_GF"]
        df["away_goalie_vs_offense"] = df["away_goalie_sv_pct"] * df["home_avg_GF"]

    return df


def _compute_xg_derived_features(df: pd.DataFrame, xg_windows: Tuple[int, ...]) -> pd.DataFrame:
    """Add xG-vs-goals deltas and matchup interactions for each xG window."""
    df = df.copy()
    for w in xg_windows:
        suffix = f"_{w}g"
        home_xgf = f"home_avg_xGF{suffix}"
        away_xgf = f"away_avg_xGF{suffix}"
        home_xga = f"home_avg_xGA{suffix}"
        away_xga = f"away_avg_xGA{suffix}"
        home_gf = f"home_avg_GF{suffix}"
        away_gf = f"away_avg_GF{suffix}"

        if all(c in df.columns for c in [home_xgf, home_gf]):
            df[f"home_xg_diff_{w}g"] = df[home_xgf] - df[home_gf]
        if all(c in df.columns for c in [away_xgf, away_gf]):
            df[f"away_xg_diff_{w}g"] = df[away_xgf] - df[away_gf]
        if all(c in df.columns for c in [home_xgf, away_xga]):
            df[f"xg_scoring_opp_{w}g"] = df[home_xgf] * df[away_xga]
        if all(c in df.columns for c in [away_xgf, home_xga]):
            df[f"xg_opp_threat_{w}g"] = df[away_xgf] * df[home_xga]
    return df


def _compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal and seasonal features.

    NHL scoring patterns vary throughout the season:
    - Early season: More goals (teams finding form)
    - Late season: Tighter games (playoff implications)
    - Day of week: Possible scheduling effects

    Parameters
    ----------
    df : pd.DataFrame
        Game data with date column.

    Returns
    -------
    pd.DataFrame
        Data with temporal features added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Month (captures seasonal patterns)
    df["month"] = df["date"].dt.month

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Weekend indicator (Saturday=5, Sunday=6)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Days into season (from Oct 1 of season year)
    def days_into_season(row):
        game_date = row["date"]
        # Determine season start year (games before July belong to previous year's season)
        if game_date.month >= config.features.season_start_month:
            season_start = datetime(
                game_date.year,
                config.features.season_start_month,
                config.features.season_start_day
            )
        else:
            season_start = datetime(
                game_date.year - 1,
                config.features.season_start_month,
                config.features.season_start_day
            )
        return (game_date - season_start).days

    df["days_into_season"] = df.apply(days_into_season, axis=1)

    # Normalized season progress (0 = start, 1 = end of regular season ~180 days)
    df["season_progress"] = (df["days_into_season"] / 180).clip(0, 1)

    # Late season indicator (last 2 months - playoff race)
    df["is_late_season"] = (df["days_into_season"] > 140).astype(int)

    # Early season indicator (first month - teams finding form)
    df["is_early_season"] = (df["days_into_season"] < 30).astype(int)

    return df


def add_features(
    df: pd.DataFrame,
    *,
    window: int | None = None,
    min_games: int | None = None,
    include_goalies: bool | None = None,
    include_xg: bool | None = None,
    include_multi_window: bool | None = None,
    include_interactions: bool | None = None,
    include_temporal: bool | None = None,
) -> pd.DataFrame:
    """Add all features to the game dataset.

    This is the main entry point for feature engineering. It computes rolling
    statistics for both home and away teams and joins them back to the original
    game-level data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw game logs with columns: gamePk, date, homeTeam, awayTeam,
        homeScore, awayScore, totalGoals.
    window : int, optional
        Number of prior games for rolling averages. Defaults to config value.
    min_games : int, optional
        Minimum games a team must have played before features are valid.
        Games before this threshold will have NaN features. Defaults to config value.
    include_goalies : bool, optional
        If True, add goalie features (requires goalie data to be fetched).
        Defaults to config value.
    include_xg : bool, optional
        If True, add MoneyPuck rolling xG features when available.
        If unavailable, logs a warning and continues without xG columns.
    include_multi_window : bool, optional
        If True, compute rolling stats for multiple windows (5, 10, 20, 40 games).
        Defaults to config value.
    include_interactions : bool, optional
        If True, add interaction features (matchup dynamics).
        Defaults to config value.
    include_temporal : bool, optional
        If True, add temporal/seasonal features (month, days_into_season).
        Defaults to config value.

    Returns
    -------
    pd.DataFrame
        Original data with added feature columns for both home and away teams.
    """
    # Apply config defaults
    if window is None:
        window = config.features.rolling_window
    if min_games is None:
        min_games = config.features.min_games
    if include_goalies is None:
        include_goalies = config.features.include_goalies
    if include_xg is None:
        include_xg = config.features.include_xg
    if include_multi_window is None:
        include_multi_window = config.features.include_multi_window
    if include_interactions is None:
        include_interactions = config.features.include_interactions
    if include_temporal is None:
        include_temporal = config.features.include_temporal

    if df.empty:
        return df.copy()

    # Validate input data
    validate_game_data(df)

    # Build team-level game log
    team_log = _build_team_game_log(df)

    # Compute rolling features (primary window)
    team_log = _compute_rolling_features(team_log, window=window)

    # Compute multi-window rolling features if enabled
    if include_multi_window:
        team_log = _compute_multi_window_rolling(
            team_log, windows=config.features.rolling_windows
        )
        logger.debug("Added multi-window features for windows: %s", config.features.rolling_windows)

    # Add optional rolling xG features
    if include_xg:
        try:
            seasons = tuple(sorted(df["season"].astype(str).unique()))
            team_log = _add_xg_rolling_features(
                team_log,
                seasons=seasons,
                xg_windows=config.features.xg_windows,
            )
            logger.debug("Added xG rolling features for windows: %s", config.features.xg_windows)
        except Exception as e:
            logger.warning("Could not add xG features; proceeding without xG columns: %s", e)

    # Base feature columns (from primary window)
    feature_cols = ["avg_GF", "avg_GA", "avg_total", "win_pct", "rest_days",
                    "is_back_to_back", "win_streak", "home_win_pct", "away_win_pct"]

    # Add multi-window feature columns
    multi_window_cols = []
    if include_multi_window:
        for w in config.features.rolling_windows:
            suffix = f"_{w}g"
            multi_window_cols.extend([
                f"avg_GF{suffix}", f"avg_GA{suffix}", f"avg_total{suffix}",
                f"win_pct{suffix}", f"std_GF{suffix}"
            ])

    xg_cols = []
    if include_xg:
        for w in config.features.xg_windows:
            suffix = f"_{w}g"
            xg_cols.extend([f"avg_xGF{suffix}", f"avg_xGA{suffix}"])

    all_feature_cols = feature_cols + multi_window_cols + xg_cols

    # Set features to NaN for teams without enough history
    mask = team_log["games_played"] < min_games
    # Only set NaN for columns that exist
    cols_to_nan = [c for c in all_feature_cols if c in team_log.columns]
    team_log.loc[mask, cols_to_nan] = np.nan

    # Split back into home and away
    home_feats = team_log[team_log["is_home"]].copy()
    away_feats = team_log[~team_log["is_home"]].copy()

    # Rename columns for home team
    home_rename = {col: f"home_{col}" for col in all_feature_cols if col in home_feats.columns}
    home_rename["games_played"] = "home_games_played"
    home_feats = home_feats.rename(columns=home_rename)

    # Rename columns for away team
    away_rename = {col: f"away_{col}" for col in all_feature_cols if col in away_feats.columns}
    away_rename["games_played"] = "away_games_played"
    away_feats = away_feats.rename(columns=away_rename)

    # Select columns to merge
    home_cols = ["gamePk"] + list(home_rename.values())
    away_cols = ["gamePk"] + list(away_rename.values())

    # Merge features back to original game data
    df_out = df.copy()
    df_out = df_out.merge(home_feats[home_cols], on="gamePk", how="left")
    df_out = df_out.merge(away_feats[away_cols], on="gamePk", how="left")

    # Add goalie features if requested
    if include_goalies:
        try:
            from .goalies import add_goalie_features
            df_out = add_goalie_features(df_out, fetch_missing=False)
        except Exception as e:
            logger.warning("Could not add goalie features: %s", e)

    # Add head-to-head features
    df_out = _compute_h2h_features(df_out, window=window // 2)  # Use half window for H2H

    # Add venue-specific features
    df_out = _compute_venue_features(df_out, window=window)

    # Add temporal features if enabled
    if include_temporal:
        df_out = _compute_temporal_features(df_out)
        logger.debug("Added temporal features")

    # Add interaction features if enabled (must be after other features)
    if include_interactions:
        df_out = _compute_interaction_features(df_out)
        logger.debug("Added interaction features")

    if include_xg:
        df_out = _compute_xg_derived_features(df_out, config.features.xg_windows)

    # Sort by date
    df_out = df_out.sort_values("date").reset_index(drop=True)

    return df_out


# Backwards compatibility alias
def add_rolling_team_features(
    df: pd.DataFrame,
    *,
    window: int = 5,
    min_games: int = 1,
) -> pd.DataFrame:
    """Legacy function name for backwards compatibility.

    Use `add_features()` instead for the full feature set.
    """
    return add_features(df, window=window, min_games=min_games)
