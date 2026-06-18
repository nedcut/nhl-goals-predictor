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

import numpy as np
import pandas as pd

from .artifacts import ModelArtifact
from .conformal import split_conformal_interval
from .config import config
from .data import build_dataset, fetch_schedule_week, recent_seasons
from .features import add_features, feature_fill_values, impute_features
from .live import apply_live_residual_update, fetch_live_states
from .logging_config import get_logger, setup_logging
from .probabilistic import (
    discrete_quantile_from_pmf,
    fit_nb2_alpha,
    fit_poisson_mixture,
    nb2_pmf_matrix,
    poisson_mixture_pmf_matrix,
    poisson_pmf_matrix,
    prob_over_from_pmf,
)

logger = get_logger(__name__)


def _model_requires_xg(feature_names: List[str]) -> bool:
    return any("xg" in name.lower() for name in feature_names)


def _prepare_upcoming_rows(upcoming_df: pd.DataFrame) -> pd.DataFrame:
    """Mark scheduled games and remove placeholder score values."""
    upcoming = upcoming_df.copy()
    upcoming["_is_upcoming"] = True
    for col in ("homeScore", "awayScore", "totalGoals"):
        upcoming[col] = pd.NA
    return upcoming


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
    end_date = today + timedelta(days=days_ahead)
    all_games = []

    # Fetch schedule for the coming weeks
    current_date = today

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
                    if today <= game_date <= end_date:
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
    include_xg = _model_requires_xg(artifact.metadata.feature_names)

    # For upcoming games, we need to compute features based on historical data
    # Mark upcoming games so they don't pollute rolling features
    upcoming_df = _prepare_upcoming_rows(upcoming_df)

    historical_df = historical_df.copy()
    historical_df["_is_upcoming"] = False

    # Combine historical and upcoming data
    combined = pd.concat([historical_df, upcoming_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["gamePk"], keep="first")

    # Add features (rolling stats will use historical data only due to NA scores)
    combined = add_features(combined, include_goalies=False, include_xg=include_xg)

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

    # Training-representative fill values from the historical games (not the
    # handful of upcoming rows), so a missing feature is imputed with a value the
    # model actually saw during training rather than the request-batch mean.
    historical_features = combined[~combined["gamePk"].isin(upcoming_game_ids)]
    fill_values = feature_fill_values(historical_features, expected_features)

    # Prepare features for prediction in exact model order
    X = impute_features(upcoming_with_features[expected_features].copy(), fill_values)

    # Make predictions
    predictions = artifact.predict(X)

    # Create results DataFrame
    results = upcoming_with_features[["date", "homeTeam", "awayTeam"]].copy()
    results["predicted_total_goals"] = predictions.round(1)

    return results


def predict_games_probabilistic(
    upcoming_df: pd.DataFrame,
    model_path: Path,
    historical_df: pd.DataFrame,
    *,
    dist_model: str = "nb2",
    thresholds: Optional[List[float]] = None,
    max_goals: int = 20,
    cal_fraction: float = 0.2,
) -> pd.DataFrame:
    """Probabilistic predictions for upcoming games.

    Returns columns:
      - mu (expected total goals)
      - p_over_{threshold}
      - pi80_low/pi80_high (80% predictive interval from the chosen distribution)
      - conformal90_low/conformal90_high (split-conformal interval from historical residuals)
      - pmf (list[float] over {0..max_goals})
    """
    if thresholds is None:
        thresholds = [5.5, 6.5, 7.5]

    artifact = ModelArtifact.load(model_path)
    expected_features = artifact.metadata.feature_names
    include_xg = _model_requires_xg(expected_features)

    upcoming_df = _prepare_upcoming_rows(upcoming_df)

    historical_df = historical_df.copy()
    historical_df["_is_upcoming"] = False

    combined = pd.concat([historical_df, upcoming_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["gamePk"], keep="first")
    combined = add_features(combined, include_goalies=False, include_xg=include_xg)

    # Calibrate dispersion and conformal radius using historical games only
    hist = combined[combined["_is_upcoming"] == False].copy()  # noqa: E712
    hist = hist.dropna(subset=["totalGoals"]).sort_values("date").reset_index(drop=True)
    hist = hist.dropna(subset=expected_features)
    if hist.empty:
        raise ValueError("No historical rows with complete features for calibration.")

    # Training-representative fill values from the complete historical frame, so
    # calibration and upcoming-game inference impute identically (not from the
    # request batch mean).
    fill_values = feature_fill_values(hist, expected_features)

    def build_X(frame: pd.DataFrame) -> pd.DataFrame:
        return impute_features(frame.reindex(columns=expected_features).copy(), fill_values)

    n_hist = len(hist)
    cal_size = max(1, int(cal_fraction * n_hist))
    cal_df = hist.iloc[n_hist - cal_size :].copy()
    X_cal = build_X(cal_df)
    y_cal = cal_df["totalGoals"].to_numpy(dtype=float)
    mu_cal = artifact.predict(X_cal)

    nb2_alpha = None
    mix_w = None
    mix_m = None
    if dist_model == "poisson":
        pass
    elif dist_model == "nb2":
        nb2_alpha = fit_nb2_alpha(y_cal, mu_cal)
    elif dist_model == "poisson_mixture":
        mix_w, mix_m = fit_poisson_mixture(y_cal, mu_cal, max_goals=max_goals)
    else:
        raise ValueError(f"Unknown dist_model: {dist_model}")

    # Conformal 90% interval radius (simple split-conformal on calibration slice)
    _, _, q90 = split_conformal_interval(y_cal, mu_cal, mu_cal, alpha=0.1, clip_lower=0.0)

    # Forecast upcoming
    upcoming_ids = set(upcoming_df["gamePk"])
    up = combined[combined["gamePk"].isin(upcoming_ids)].copy()
    if up.empty:
        return pd.DataFrame()

    X_up = build_X(up)
    mu_up = artifact.predict(X_up)

    if dist_model == "poisson":
        pmf = poisson_pmf_matrix(mu_up, max_goals=max_goals)
    elif dist_model == "nb2":
        pmf = nb2_pmf_matrix(mu_up, alpha=float(nb2_alpha), max_goals=max_goals)
    else:
        pmf = poisson_mixture_pmf_matrix(mu_up, weight=float(mix_w), multiplier=float(mix_m), max_goals=max_goals)

    results = up[["gamePk", "date", "homeTeam", "awayTeam"]].copy()
    results["mu"] = mu_up.astype(float)

    for t in thresholds:
        results[f"p_over_{t:g}"] = prob_over_from_pmf(pmf, threshold=t)

    # Distributional predictive interval (80%)
    results["pi80_low"] = discrete_quantile_from_pmf(pmf, 0.10).astype(int)
    results["pi80_high"] = discrete_quantile_from_pmf(pmf, 0.90).astype(int)

    # Conformal interval around the mean (90%)
    results["conformal90_low"] = (results["mu"] - q90).clip(lower=0.0)
    results["conformal90_high"] = results["mu"] + q90

    # Store full PMF for API use
    results["pmf"] = [row.tolist() for row in pmf]
    results["max_goals"] = max_goals
    results["dist_model"] = dist_model
    if nb2_alpha is not None:
        results["nb2_alpha"] = float(nb2_alpha)
    if mix_w is not None:
        results["mix_weight"] = float(mix_w)
        results["mix_multiplier"] = float(mix_m)

    return results


def predict_games_live(
    upcoming_df: pd.DataFrame,
    model_path: Path,
    historical_df: pd.DataFrame,
    *,
    thresholds: Optional[List[float]] = None,
    max_goals: int = 20,
) -> pd.DataFrame:
    """Live-aware predictions with residual-goals updates for LIVE games."""
    if thresholds is None:
        thresholds = [6.5]

    pre = predict_games_probabilistic(
        upcoming_df,
        model_path,
        historical_df,
        dist_model="nb2",
        thresholds=thresholds,
        max_goals=max_goals,
    )
    if pre.empty:
        return pre

    alpha = float(pre["nb2_alpha"].iloc[0]) if "nb2_alpha" in pre.columns else 0.1
    state_by_game = fetch_live_states(pre["gamePk"].astype(int).tolist())

    rows = []
    for _, row in pre.iterrows():
        game_pk = int(row["gamePk"])
        state = state_by_game.get(
            game_pk,
            {
                "gameState": "UNKNOWN",
                "homeScore": 0,
                "awayScore": 0,
                "period": 1,
                "clock": "",
                "remaining_minutes": 60.0,
            },
        )

        pre_pmf = np.asarray(row["pmf"], dtype=float)
        live_pmf = pre_pmf.copy()
        mu_live = float(row["mu"])
        is_live_adjusted = False

        game_state = str(state.get("gameState", "UNKNOWN")).upper()
        if game_state == "LIVE":
            updated = apply_live_residual_update(
                mu_pregame=float(row["mu"]),
                current_home_goals=int(state.get("homeScore", 0)),
                current_away_goals=int(state.get("awayScore", 0)),
                period=int(state.get("period", 1)),
                clock=str(state.get("clock", "")),
                game_state=game_state,
                alpha_calibrated=alpha,
                max_goals=max_goals,
            )
            live_pmf = np.asarray(updated["pmf"], dtype=float)
            mu_live = float(updated["mu_live"])
            is_live_adjusted = True

        record = {
            "gamePk": game_pk,
            "date": row["date"],
            "homeTeam": row["homeTeam"],
            "awayTeam": row["awayTeam"],
            "gameState": game_state,
            "homeScore": int(state.get("homeScore", 0)),
            "awayScore": int(state.get("awayScore", 0)),
            "period": int(state.get("period", 1)),
            "clock": str(state.get("clock", "")),
            "remaining_minutes": float(state.get("remaining_minutes", 60.0)),
            "pregame_mu": float(row["mu"]),
            "live_mu": mu_live,
            "is_live_adjusted": bool(is_live_adjusted),
            "pi80_low": int(discrete_quantile_from_pmf(live_pmf[None, :], 0.10)[0]),
            "pi80_high": int(discrete_quantile_from_pmf(live_pmf[None, :], 0.90)[0]),
            "pmf": live_pmf.tolist(),
        }
        for t in thresholds:
            record[f"pregame_p_over_{t:g}"] = float(prob_over_from_pmf(pre_pmf[None, :], threshold=t)[0])
            record[f"live_p_over_{t:g}"] = float(prob_over_from_pmf(live_pmf[None, :], threshold=t)[0])
        rows.append(record)

    return pd.DataFrame(rows).sort_values(["date", "gamePk"]).reset_index(drop=True)


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
        default=recent_seasons(2),
        help="Seasons to use for historical data (default: previous and active season)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        help="Output distributional forecasts (over/under probabilities + intervals)",
    )
    parser.add_argument(
        "--dist-model",
        choices=["poisson", "nb2", "poisson_mixture"],
        default="nb2",
        help="Distribution family for probabilistic forecasts",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[5.5, 6.5, 7.5],
        help="Over/under thresholds (e.g., 5.5 6.5 7.5)",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=20,
        help="Max goals support for PMF (0..max_goals)",
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
    if args.probabilistic:
        predictions = predict_games_probabilistic(
            upcoming,
            args.model,
            historical,
            dist_model=args.dist_model,
            thresholds=args.thresholds,
            max_goals=args.max_goals,
        )
    else:
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
            if args.probabilistic:
                mu = float(row["mu"])
                print(f"  E[goals]: {mu:.2f}")
                for t in args.thresholds:
                    key = f"p_over_{t:g}"
                    if key in row:
                        print(f"  P(> {t:g}): {float(row[key]):.3f}")
                print(f"  PI80: [{int(row['pi80_low'])}, {int(row['pi80_high'])}]")
                print(
                    f"  Conformal90: [{float(row['conformal90_low']):.2f}, {float(row['conformal90_high']):.2f}]"
                )
            else:
                print(f"  Predicted Total Goals: {row['predicted_total_goals']:.1f}")
        print("=" * 60)
        print(f"\nTotal: {len(predictions)} games")


if __name__ == "__main__":
    main()
