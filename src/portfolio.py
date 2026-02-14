"""
Portfolio pipeline orchestrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analysis import run_ablation_study, write_ablation_report, write_error_analysis
from .champion import write_champion_reports
from .data import build_dataset
from .evaluation import time_series_cv_forecast
from .features import add_features
from .logging_config import setup_logging
from .model import optimize_hyperparameters


def _write_model_card(
    *,
    path: Path,
    seasons: list[str],
    champion_payload: dict[str, Any],
) -> None:
    winner = champion_payload["champion"]["model"]
    rationale = champion_payload["rationale"]
    content = "\n".join(
        [
            "# Model Card",
            "",
            "## Intended Use",
            "- NHL total-goals forecasting for analytical/educational use.",
            "- Supports pregame probabilities and live in-game updates.",
            "",
            "## Data Sources",
            "- NHL Web API schedule/game data.",
            "- MoneyPuck historical xG (cached to `data/xg/{season}.csv`).",
            "",
            "## Feature Families",
            "- Rolling team offense/defense trends.",
            "- Rest/back-to-back and temporal features.",
            "- Matchup interactions and optional goalie context.",
            "- Rolling xG features and xG-vs-goals differentials.",
            "",
            "## Evaluation Protocol",
            "- Expanding-window time-series CV (5 folds for final evaluation).",
            "- Proper scoring rules: MAE, CRPS, distributional NLL, Brier for over 6.5.",
            "",
            "## Champion Formula",
            "- `score = 0.35*(mae/base_mae) + 0.30*(crps/base_crps) + 0.20*(dist_nll/base_dist_nll) + 0.15*(over_brier/base_brier)`",
            "- Baseline model for normalization: `team_strength`.",
            f"- Current champion: `{winner}`.",
            f"- Rationale: {rationale}",
            "",
            "## Known Failure Modes",
            "- Sparse recent form early season.",
            "- Sudden lineup or goalie changes not visible in historical aggregates.",
            "- Extreme game states and overtime tails.",
            "",
            "## Monitoring Plan",
            "- Re-run CV and champion report weekly.",
            "- Track rolling calibration and segment MAE (month/back-to-back/confidence decile).",
            "- Alert when weighted score regresses >2% vs prior champion.",
            "",
            f"## Build Context",
            f"- Seasons: {', '.join(seasons)}",
        ]
    )
    path.write_text(content)


def run_portfolio_pipeline(
    *,
    seasons: list[str],
    tune_trials: int,
    dist_model: str = "nb2",
    threshold: float = 6.5,
    reports_dir: Path = Path("reports"),
) -> dict[str, Any]:
    raw_df = build_dataset(seasons, use_cache=True)
    if raw_df.empty:
        raise ValueError("No historical data loaded.")

    full_df = add_features(
        raw_df,
        include_goalies=True,
        include_xg=True,
        include_multi_window=True,
        include_interactions=True,
        include_temporal=True,
    )

    xgb_current = time_series_cv_forecast(
        full_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
    ).metrics_mean

    best_params = optimize_hyperparameters(
        full_df,
        n_trials=tune_trials,
        objective_metric="weighted_prob",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        tune_splits=3,
        show_progress=True,
    )
    xgb_tuned = time_series_cv_forecast(
        full_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
        xgb_params=best_params,
    ).metrics_mean

    team_strength = time_series_cv_forecast(
        raw_df,
        point_model="team_strength",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
    ).metrics_mean

    poisson_glm = time_series_cv_forecast(
        full_df,
        point_model="poisson_glm",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
    ).metrics_mean

    candidates = {
        "xgb_current": xgb_current,
        "xgb_tuned": xgb_tuned,
        "team_strength": team_strength,
        "poisson_glm": poisson_glm,
    }
    context = {
        "seasons": seasons,
        "dist_model": dist_model,
        "threshold": threshold,
        "tune_trials": tune_trials,
        "tuned_params": best_params,
    }
    champion_payload = write_champion_reports(candidates=candidates, output_dir=reports_dir, context=context)

    ablation = run_ablation_study(
        raw_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
    )
    write_ablation_report(ablation, output_dir=reports_dir)

    diag_result = time_series_cv_forecast(
        full_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
        return_diagnostics=True,
    )
    write_error_analysis(diag_result.diagnostics or [], output_dir=reports_dir)

    _write_model_card(
        path=Path("MODEL_CARD.md"),
        seasons=seasons,
        champion_payload=champion_payload,
    )

    payload = {
        "champion": champion_payload["champion"],
        "tuned_params": best_params,
        "ablation": ablation,
    }
    summary_path = reports_dir / "portfolio_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run portfolio-ready modeling pipeline")
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument("--tune-trials", type=int, default=150)
    parser.add_argument("--dist-model", choices=["poisson", "nb2", "poisson_mixture"], default="nb2")
    parser.add_argument("--threshold", type=float, default=6.5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    run_portfolio_pipeline(
        seasons=list(args.seasons),
        tune_trials=int(args.tune_trials),
        dist_model=str(args.dist_model),
        threshold=float(args.threshold),
    )


if __name__ == "__main__":
    main()

