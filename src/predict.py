"""
CLI for making predictions on upcoming NHL games.

Fetches upcoming games from the NHL API, applies the trained model,
and outputs predicted total goals for each game.

Usage:
    python -m src.predict --model models/xgboost_v1 --days 7
    python -m src.predict --output predictions.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .artifacts import ModelArtifact
from .config import config
from .data import build_dataset, fetch_schedule_week
from .features import add_features
from .logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def fetch_upcoming_games(days_ahead: int = 7) -> pd.DataFrame:
    """Fetch upcoming scheduled games from the NHL API.

    Parameters
    ----------
    days_ahead : int
        Number of days ahead to look for games.

    Returns
    -------
    pd.DataFrame
        DataFrame with upcoming games (gamePk, date, homeTeam, awayTeam).
    """
    today = datetime.now().date()
    all_games = []

    # Fetch schedule for the coming weeks
    current_date = datetime.now()
    end_date = current_date + timedelta(days=days_ahead)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            # Use include_upcoming=True to get scheduled games, not just completed
            games = fetch_schedule_week(date_str, include_upcoming=True)
            for game in games:
                # Only include games that haven't been played yet
                game_state = game.get("gameState", "")
                if game_state in ("FUT", "PRE", "LIVE"):
                    game_date = datetime.strptime(game["date"], "%Y-%m-%d").date()
                    if game_date >= today:
                        all_games.append(game)
        except Exception as e:
            logger.warning("Failed to fetch schedule for %s: %s", date_str, e)

        current_date += timedelta(days=7)

    if not all_games:
        return pd.DataFrame()

    df = pd.DataFrame(all_games)

    # Remove duplicates and sort by date
    df = df.drop_duplicates(subset=["gamePk"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def predict_games(
    upcoming_df: pd.DataFrame,
    model_path: Path,
    historical_df: pd.DataFrame,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Make predictions for upcoming games.

    Parameters
    ----------
    upcoming_df : pd.DataFrame
        DataFrame with upcoming games.
    model_path : Path
        Path to the model artifact.
    historical_df : pd.DataFrame
        Historical game data for computing rolling features.
    seasons : list of str, optional
        Seasons to use for historical data.

    Returns
    -------
    pd.DataFrame
        Predictions with date, homeTeam, awayTeam, predicted_total_goals.
    """
    # Load model artifact
    artifact = ModelArtifact.load(model_path)
    logger.info("Loaded model: %s", artifact.metadata.model_type)

    # For upcoming games, we need to compute features based on historical data
    # Mark upcoming games so they don't pollute rolling features
    upcoming_df = upcoming_df.copy()
    upcoming_df["_is_upcoming"] = True
    if "homeScore" not in upcoming_df.columns:
        upcoming_df["homeScore"] = pd.NA
        upcoming_df["awayScore"] = pd.NA
        upcoming_df["totalGoals"] = pd.NA

    historical_df = historical_df.copy()
    historical_df["_is_upcoming"] = False

    # Combine historical and upcoming data
    combined = pd.concat([historical_df, upcoming_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["gamePk"], keep="first")

    # Add features (rolling stats will use historical data only due to NA scores)
    combined = add_features(combined, include_goalies=False)

    # Extract just the upcoming games with features
    upcoming_game_ids = set(upcoming_df["gamePk"])
    upcoming_with_features = combined[combined["gamePk"].isin(upcoming_game_ids)].copy()

    if upcoming_with_features.empty:
        logger.warning("No upcoming games with computable features")
        return pd.DataFrame()

    # Get expected features from model in exact order
    expected_features = artifact.metadata.feature_names
    available_features = [f for f in expected_features if f in upcoming_with_features.columns]
    missing_features = [f for f in expected_features if f not in upcoming_with_features.columns]

    if missing_features:
        logger.warning("Missing %d features: %s", len(missing_features), missing_features[:5])
        # Add missing features with NaN values to maintain feature alignment
        for f in missing_features:
            upcoming_with_features[f] = pd.NA

    # Prepare features for prediction in exact model order
    X = upcoming_with_features[expected_features].copy()

    # Fill NaN values with column means from training data or zeros
    for col in X.columns:
        if X[col].isna().any():
            col_mean = X[col].mean()
            X[col] = X[col].fillna(col_mean if pd.notna(col_mean) else 0)

    # Make predictions
    predictions = artifact.predict(X)

    # Create results DataFrame
    results = upcoming_with_features[["date", "homeTeam", "awayTeam"]].copy()
    results["predicted_total_goals"] = predictions.round(1)

    return results


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the prediction CLI."""
    parser = argparse.ArgumentParser(
        description="Predict total goals for upcoming NHL games"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/xgboost_v1"),
        help="Path to model artifact (without extension)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: print to stdout)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days ahead to fetch games",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["20232024", "20242025"],
        help="Seasons to use for historical data",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Load historical data for feature computation
    logger.info("Loading historical data for seasons: %s", args.seasons)
    historical = build_dataset(args.seasons, use_cache=True)

    if historical.empty:
        logger.error("No historical data available")
        return

    logger.info("Loaded %d historical games", len(historical))

    # Fetch upcoming games
    logger.info("Fetching upcoming games for next %d days", args.days)
    upcoming = fetch_upcoming_games(args.days)

    if upcoming.empty:
        print("No upcoming games found.")
        return

    logger.info("Found %d upcoming games", len(upcoming))

    # Make predictions
    predictions = predict_games(
        upcoming,
        args.model,
        historical,
        seasons=args.seasons,
    )

    if predictions.empty:
        print("Could not generate predictions.")
        return

    # Output results
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Saved {len(predictions)} predictions to {args.output}")
    else:
        print("\nUpcoming Game Predictions:")
        print("=" * 60)
        for _, row in predictions.iterrows():
            print(f"{row['date']}: {row['awayTeam']} @ {row['homeTeam']}")
            print(f"  Predicted Total Goals: {row['predicted_total_goals']:.1f}")
        print("=" * 60)
        print(f"\nTotal: {len(predictions)} games")


if __name__ == "__main__":
    main()
