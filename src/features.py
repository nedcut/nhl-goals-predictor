"""
Feature engineering utilities for NHL total goals prediction.

The functions in this module operate on the raw game logs produced by
`src.data.build_dataset`. They generate rolling averages and other features
that capture team performance, fatigue, and momentum.

Features generated (20 total):
- Rolling goals for/against averages (home_avg_GF, away_avg_GF, etc.)
- Rest days and back-to-back indicators (home_rest_days, home_is_back_to_back)
- Win streaks (home_win_streak, away_win_streak)
- Win percentages (home_win_pct, away_win_pct)
- Games played in season (home_games_played, away_games_played)
- Goalie rolling stats (home_goalie_sv_pct, home_goalie_gaa, etc.)

Usage:
    from src.data import build_dataset
    from src.features import add_features

    df = build_dataset(['20232024', '20242025'])
    df_features = add_features(df, window=20, include_goalies=True)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


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


def add_features(
    df: pd.DataFrame,
    *,
    window: int = 5,
    min_games: int = 3,
    include_goalies: bool = False,
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
        Number of prior games for rolling averages. Default is 5.
    min_games : int, optional
        Minimum games a team must have played before features are valid.
        Games before this threshold will have NaN features. Default is 3.
    include_goalies : bool, optional
        If True, add goalie features (requires goalie data to be fetched).
        Default is False.

    Returns
    -------
    pd.DataFrame
        Original data with added feature columns for both home and away teams.
    """
    if df.empty:
        return df.copy()

    # Build team-level game log
    team_log = _build_team_game_log(df)

    # Compute rolling features
    team_log = _compute_rolling_features(team_log, window=window)

    # Set features to NaN for teams without enough history
    feature_cols = ["avg_GF", "avg_GA", "avg_total", "win_pct", "rest_days",
                    "is_back_to_back", "win_streak", "home_win_pct", "away_win_pct"]
    mask = team_log["games_played"] < min_games
    team_log.loc[mask, feature_cols] = np.nan

    # Split back into home and away
    home_feats = team_log[team_log["is_home"]].copy()
    away_feats = team_log[~team_log["is_home"]].copy()

    # Rename columns for home team
    home_rename = {col: f"home_{col}" for col in feature_cols}
    home_rename["games_played"] = "home_games_played"
    home_feats = home_feats.rename(columns=home_rename)

    # Rename columns for away team
    away_rename = {col: f"away_{col}" for col in feature_cols}
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
            print(f"Warning: Could not add goalie features: {e}")

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
