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
from .evaluation import CVForecastResult, time_series_cv_forecast
from .features import add_features
from .logging_config import setup_logging
from .model import get_feature_columns, optimize_hyperparameters

# Diagnostic multi-line over/under reporting (not used in champion weighting).
DEFAULT_CALIBRATION_THRESHOLDS = [5.5, 6.5, 7.5]


def _write_threshold_calibration_report(
    result: CVForecastResult,
    *,
    output_dir: Path,
    thresholds: list[float],
) -> dict[str, Any]:
    """Write multi-threshold Brier/log-loss + reliability summary for diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_metrics = result.threshold_metrics or {}
    reliability_by_threshold = {
        label: [b.__dict__ for b in bins]
        for label, bins in (result.reliability_by_threshold or {}).items()
    }
    payload: dict[str, Any] = {
        "point_model": result.point_model,
        "dist_model": result.dist_model,
        "primary_threshold": result.threshold,
        "thresholds": thresholds,
        "threshold_metrics": threshold_metrics,
        "reliability_by_threshold": reliability_by_threshold,
        "primary_metrics": {
            "over_brier": result.metrics_mean.get("over_brier"),
            "over_log_loss": result.metrics_mean.get("over_log_loss"),
        },
    }
    json_path = output_dir / "threshold_calibration.json"
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Threshold Calibration",
        "",
        f"Point model: `{result.point_model}` · Dist: `{result.dist_model}` · "
        f"Primary line: `{result.threshold:g}`",
        "",
        "Brier and log-loss for P(total goals > t) across common O/U lines. "
        "Champion weighting still uses the primary threshold only.",
        "",
        "| Threshold | Brier | Log loss |",
        "|-----------|------:|---------:|",
    ]
    for t in thresholds:
        label = f"{float(t):g}"
        m = threshold_metrics.get(label, {})
        brier = m.get("brier")
        ll = m.get("log_loss")
        brier_s = f"{brier:.4f}" if brier is not None else "—"
        ll_s = f"{ll:.4f}" if ll is not None else "—"
        lines.append(f"| >{label} | {brier_s} | {ll_s} |")
    lines.append("")
    md_path = output_dir / "threshold_calibration.md"
    md_path.write_text("\n".join(lines) + "\n")
    return payload


def _write_model_card(
    *,
    path: Path,
    seasons: list[str],
    champion_payload: dict[str, Any],
) -> None:
    winner = champion_payload["champion"]["model"]
    rationale = champion_payload["rationale"]
    raw_leader = champion_payload.get("raw_score_leader") or champion_payload["champion"]
    raw_leader_name = raw_leader["model"] if isinstance(raw_leader, dict) else str(raw_leader)
    policy = champion_payload.get("selection_policy", "significance_prefer_simpler")
    sig = champion_payload.get("champion_vs_runner_up")
    if sig is None:
        sig_line = (
            "- Score-leader vs runner-up significance: not computed "
            "(champion is the weighted-score leader)."
        )
    elif sig["significant"]:
        sig_line = (
            f"- Score leader (`{raw_leader_name}`) beats runner-up with statistical "
            f"significance (95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], "
            f"p={sig['p_value']:.3f}); champion remains the score leader."
        )
    else:
        demoted = winner != raw_leader_name
        if demoted:
            sig_line = (
                f"- Score leader (`{raw_leader_name}`) margin over runner-up is within noise "
                f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], p={sig['p_value']:.3f}); "
                f"champion demoted to simpler model `{winner}`."
            )
        else:
            sig_line = (
                f"- Score leader margin over runner-up is within noise "
                f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], p={sig['p_value']:.3f}); "
                f"keeping simpler/equal-complexity model `{winner}`."
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
            f"- Selection policy: `{policy}` — rank by weighted score; if the "
            "score-leader vs runner-up paired bootstrap is not significant, "
            "prefer the simpler/cheaper model as champion.",
            f"- Current champion: `{winner}`.",
            f"- Weighted-score leader: `{raw_leader_name}`.",
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
            "## Build Context",
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

    # Diagnostic CV only: multi-threshold calibration is cheap on an already-trained
    # fold path and is kept off champion CVs so selection stays single-line.
    calib_thresholds = list(DEFAULT_CALIBRATION_THRESHOLDS)
    if threshold not in calib_thresholds:
        calib_thresholds = sorted({*calib_thresholds, threshold})
    diag_result = time_series_cv_forecast(
        eval_df,
        point_model="xgb",
        dist_model=dist_model,  # type: ignore[arg-type]
        threshold=threshold,
        thresholds=calib_thresholds,
        n_splits=5,
        feature_cols=feature_cols,
        return_diagnostics=True,
    )
    write_error_analysis(diag_result.diagnostics or [], output_dir=reports_dir)
    threshold_calibration = _write_threshold_calibration_report(
        diag_result,
        output_dir=reports_dir,
        thresholds=calib_thresholds,
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
        "threshold_calibration": threshold_calibration.get("threshold_metrics", {}),
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
