"""
Input validation utilities for NHL goals prediction pipeline.

Provides validation functions and decorators to ensure data integrity
and provide helpful error messages when inputs are malformed.

Usage:
    from src.validation import validate_game_data, validate_features

    validate_game_data(df)  # Raises ValidationError if invalid
    validate_features(df)   # Raises ValidationError if no features found
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, Iterable, Set

import pandas as pd


class ValidationError(ValueError):
    """Raised when input validation fails.

    Provides helpful error messages indicating what's wrong with the input.
    """

    pass


# Required columns for raw game data
REQUIRED_GAME_COLUMNS: Set[str] = {
    "gamePk",
    "date",
    "homeTeam",
    "awayTeam",
    "homeScore",
    "awayScore",
    "totalGoals",
}

# Feature column prefixes (from add_features)
FEATURE_PREFIXES = (
    "home_avg_",
    "away_avg_",
    "home_win_",
    "away_win_",
    "home_rest_",
    "away_rest_",
    "home_is_",
    "away_is_",
    "home_games_",
    "away_games_",
    "home_goalie_",
    "away_goalie_",
    "h2h_",
    "venue_",
)


def validate_game_data(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required columns for game data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValidationError
        If required columns are missing or DataFrame is empty.
    """
    if df is None:
        raise ValidationError("DataFrame is None")

    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValidationError("DataFrame is empty")

    missing = REQUIRED_GAME_COLUMNS - set(df.columns)
    if missing:
        raise ValidationError(
            f"Missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_GAME_COLUMNS)}"
        )


def validate_features(df: pd.DataFrame) -> None:
    """Validate that DataFrame has feature columns from add_features().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValidationError
        If no feature columns are found.
    """
    if df is None:
        raise ValidationError("DataFrame is None")

    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValidationError("DataFrame is empty")

    feature_cols = [
        col for col in df.columns
        if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES)
    ]

    if not feature_cols:
        raise ValidationError(
            "No feature columns found. Run add_features() first. "
            f"Expected columns starting with: {FEATURE_PREFIXES[:4]}..."
        )


def validate_target(df: pd.DataFrame, target_col: str = "totalGoals") -> None:
    """Validate that DataFrame has the target column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    target_col : str
        Name of the target column.

    Raises
    ------
    ValidationError
        If target column is missing.
    """
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in DataFrame")


def validate_seasons(seasons: Iterable[str]) -> None:
    """Validate season format strings.

    Parameters
    ----------
    seasons : iterable of str
        Season codes to validate (e.g., ["20222023", "20232024"]).

    Raises
    ------
    ValidationError
        If any season string is malformed.
    """
    seasons_list = list(seasons)

    if not seasons_list:
        raise ValidationError("No seasons provided")

    for season in seasons_list:
        if not isinstance(season, str):
            raise ValidationError(f"Season must be a string, got {type(season).__name__}")

        if len(season) != 8:
            raise ValidationError(
                f"Season '{season}' must be 8 characters (e.g., '20232024')"
            )

        if not season.isdigit():
            raise ValidationError(f"Season '{season}' must contain only digits")

        start_year = int(season[:4])
        end_year = int(season[4:])

        if end_year != start_year + 1:
            raise ValidationError(
                f"Season '{season}' must span consecutive years "
                f"(got {start_year} to {end_year})"
            )


def validate_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that DataFrame is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    name : str
        Name to use in error message.

    Raises
    ------
    ValidationError
        If DataFrame is empty.
    """
    if df.empty:
        raise ValidationError(f"{name} is empty")


def requires_game_data(func: Callable) -> Callable:
    """Decorator that validates game data before function execution.

    Usage:
        @requires_game_data
        def process_games(df: pd.DataFrame) -> pd.DataFrame:
            ...
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        validate_game_data(df)
        return func(df, *args, **kwargs)

    return wrapper


def requires_features(func: Callable) -> Callable:
    """Decorator that validates feature columns before function execution.

    Usage:
        @requires_features
        def train_model(df: pd.DataFrame) -> Model:
            ...
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        validate_features(df)
        return func(df, *args, **kwargs)

    return wrapper
