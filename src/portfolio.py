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
from .decision import write_decision_report
from .evaluation import time_series_cv_forecast
from .features import add_features
from .logging_config import setup_logging
from .model import get_feature_columns, optimize_hyperparameters


def _write_model_card(
    *,
    path: Path,
    seasons: list[str],
    champion_payload: dict[str, Any],
) -> None:
    winner = champion_payload["champion"]["model"]
    rationale = champion_payload["rationale"]
    sig = champion_payload.get("champion_vs_runner_up")
    if sig is None:
        sig_line = "- Champion-vs-runner-up significance: not computed."
    elif sig["significant"]:
        sig_line = (
            f"- Champion beats runner-up with statistical significance "
            f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], p={sig['p_value']:.3f})."
        )
    else:
        sig_line = (
            f"- Champion margin over runner-up is within noise "
            f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], p={sig['p_value']:.3f}); "
            "treat the two models as statistically indistinguishable."
        )
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
            sig_line,
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
    include_xg: bool = True,
) -> dict[str, Any]:
    raw_df = build_dataset(seasons, use_cache=True)
    if raw_df.empty:
        raise ValueError("No historical data loaded.")

    # require_xg mirrors include_xg: when xG is requested it must materialize
    # (no silent "xG model" without xG); when disabled (e.g. the MoneyPuck feed
    # is unavailable) the pipeline still runs on the remaining feature families.
    full_df = add_features(
        raw_df,
        include_goalies=True,
        include_xg=include_xg,
        require_xg=include_xg,
        include_multi_window=True,
        include_interactions=True,
        include_temporal=True,
    )

    # Pre-filter to rows with complete features ONCE so every candidate is
    # evaluated on the *same* games (identical TimeSeriesSplit folds). This makes
    # the comparison apples-to-apples and lets per-game scores be paired across
    # all models, including the team_strength baseline.
    feature_cols = get_feature_columns(full_df)
    eval_df = full_df.dropna(subset=feature_cols + ["totalGoals"]).reset_index(drop=True)
    if eval_df.empty:
        raise ValueError("No rows with complete features after filtering.")

    def _cv(point_model: str, xgb_params: dict[str, Any] | None = None):
        return time_series_cv_forecast(
            eval_df,
            point_model=point_model,  # type: ignore[arg-type]
            dist_model=dist_model,  # type: ignore[arg-type]
            threshold=threshold,
            n_splits=5,
            feature_cols=feature_cols if point_model != "team_strength" else None,
            xgb_params=xgb_params,
        )

    res_xgb_current = _cv("xgb")

    best_params = optimize_hyperparameters(
        eval_df,
        n_trials=tune_trials,
        objective_metric="weighted_prob",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        tune_splits=3,
        show_progress=True,
    )
    res_xgb_tuned = _cv("xgb", xgb_params=best_params)
    res_team_strength = _cv("team_strength")
    res_poisson_glm = _cv("poisson_glm")

    results = {
        "xgb_current": res_xgb_current,
        "xgb_tuned": res_xgb_tuned,
        "team_strength": res_team_strength,
        "poisson_glm": res_poisson_glm,
    }
    candidates = {name: res.metrics_mean for name, res in results.items()}
    fold_std = {name: res.metrics_std for name, res in results.items()}
    per_game_map = {name: res.per_game for name, res in results.items() if res.per_game is not None}
    context = {
        "seasons": seasons,
        "dist_model": dist_model,
        "threshold": threshold,
        "tune_trials": tune_trials,
        "tuned_params": best_params,
    }
    champion_payload = write_champion_reports(
        candidates=candidates,
        output_dir=reports_dir,
        context=context,
        per_game_map=per_game_map,
        fold_std=fold_std,
    )

    ablation = run_ablation_study(
        raw_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
        include_xg=include_xg,
    )
    write_ablation_report(ablation, output_dir=reports_dir)

    diag_result = time_series_cv_forecast(
        eval_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        n_splits=5,
        feature_cols=feature_cols,
        return_diagnostics=True,
    )
    write_error_analysis(diag_result.diagnostics or [], output_dir=reports_dir)

    # Decision/edge lens on the diagnostics CV (same games as error analysis).
    # Synthetic fair reference only — educational, not a market claim.
    if diag_result.per_game is not None:
        write_decision_report(
            diag_result.per_game["p_over"],
            diag_result.per_game["y_over"],
            line=float(threshold),
            line_prob_over=0.5,
            output_dir=reports_dir,
            context={
                "point_model": "xgb",
                "dist_model": dist_model,
                "threshold": threshold,
                "seasons": seasons,
                "source": "portfolio_diagnostics_cv",
            },
        )

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
    parser.add_argument("--no-xg", action="store_true", help="Skip xG features (e.g. MoneyPuck feed unavailable)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    run_portfolio_pipeline(
        seasons=list(args.seasons),
        tune_trials=int(args.tune_trials),
        dist_model=str(args.dist_model),
        threshold=float(args.threshold),
        include_xg=not args.no_xg,
    )


if __name__ == "__main__":
    main()
