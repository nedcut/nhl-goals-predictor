"""
CLI: probabilistic evaluation with time-series CV.

Examples:
  python -m src.evaluate --seasons 20222023 20232024 20242025 --point-model xgb --dist-model nb2
  python -m src.evaluate --point-model team_strength --dist-model poisson --threshold 6.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .data import build_dataset
from .decision import evaluate_decisions, write_decision_report
from .evaluation import time_series_cv_forecast
from .features import add_features
from .logging_config import setup_logging


def _threshold_label(threshold: float) -> str:
    return f"{float(threshold):g}"


def _plot_reliability(bins, *, title: str, out_path: Path) -> None:
    xs = [b.p_mean for b in bins]
    ys = [b.frac_pos for b in bins]
    ns = [b.count for b in bins]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.plot(xs, ys, marker="o")
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pit(pit: list[float], *, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(pit, bins=20, range=(0, 1), density=True, alpha=0.8, edgecolor="white")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("u")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate probabilistic forecasts with time-series CV"
    )
    parser.add_argument("--seasons", nargs="+", default=["20232024", "20242025"])
    parser.add_argument(
        "--point-model",
        choices=["xgb", "poisson_glm", "team_strength", "double_poisson"],
        default="xgb",
    )
    parser.add_argument(
        "--dist-model", choices=["poisson", "nb2", "poisson_mixture"], default="nb2"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.5,
        help="Primary over/under threshold for fold metrics (e.g., 6.5)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[5.5, 6.5, 7.5],
        help="Over/under lines to report Brier/log-loss/reliability for",
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--cal-fraction", type=float, default=0.2)
    parser.add_argument("--max-goals", type=int, default=20)
    parser.add_argument(
        "--include-xg", action="store_true", help="Include MoneyPuck xG features when available"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Save fold-level diagnostics CSV"
    )
    parser.add_argument(
        "--decision-eval",
        action="store_true",
        help="Opt in to base-rate decision diagnostics (not a sportsbook backtest)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("reports"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    df = build_dataset(args.seasons, use_cache=True)
    df = add_features(df, include_goalies=False, include_xg=args.include_xg)

    result = time_series_cv_forecast(
        df,
        point_model=args.point_model,
        dist_model=args.dist_model,
        threshold=args.threshold,
        thresholds=list(args.thresholds),
        n_splits=args.splits,
        cal_fraction=args.cal_fraction,
        max_goals=args.max_goals,
        return_diagnostics=args.diagnostics,
    )

    metrics = result.metrics_mean
    print("\nMean CV metrics")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:>14}: {v:.4f}")
    if result.threshold_metrics:
        print("\nOver/under by threshold")
        print("-" * 60)
        for label, m in result.threshold_metrics.items():
            print(f"  >{label:>4}: brier={m['brier']:.4f}  log_loss={m['log_loss']:.4f}")
    print("=" * 60)

    # Save metrics + folds
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    reliability_by_threshold = {
        label: [b.__dict__ for b in bins]
        for label, bins in (result.reliability_by_threshold or {}).items()
    }
    payload = {
        "point_model": result.point_model,
        "dist_model": result.dist_model,
        "threshold": result.threshold,
        "thresholds": list(args.thresholds),
        "max_goals": result.max_goals,
        "metrics_mean": metrics,
        "threshold_metrics": result.threshold_metrics or {},
        "folds": [f.__dict__ for f in result.folds],
        "reliability_bins": [b.__dict__ for b in result.reliability_bins],
        "reliability_by_threshold": reliability_by_threshold,
        "pit_values": result.pit_values,
    }
    metrics_path = outdir / f"cv_{args.point_model}_{args.dist_model}.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    # Reliability plot per reported threshold
    plot_paths: list[Path] = []
    rel_map = result.reliability_by_threshold or {}
    for t in args.thresholds:
        label = _threshold_label(t)
        bins = rel_map.get(label, result.reliability_bins if t == args.threshold else None)
        if bins is None:
            continue
        plot_path = outdir / f"reliability_over_{label}_{args.point_model}_{args.dist_model}.png"
        _plot_reliability(
            bins,
            title=f"Reliability: P(total goals > {label})",
            out_path=plot_path,
        )
        plot_paths.append(plot_path)

    # Always keep a primary-threshold plot name even if thresholds omitted it
    primary_label = _threshold_label(args.threshold)
    primary_plot = (
        outdir / f"reliability_over_{primary_label}_{args.point_model}_{args.dist_model}.png"
    )
    if primary_plot not in plot_paths:
        _plot_reliability(
            result.reliability_bins,
            title=f"Reliability: P(total goals > {primary_label})",
            out_path=primary_plot,
        )
        plot_paths.append(primary_plot)

    pit_path = outdir / f"pit_{args.point_model}_{args.dist_model}.png"
    _plot_pit(
        result.pit_values,
        title="Randomized PIT (distribution calibration)",
        out_path=pit_path,
    )

    if args.diagnostics and result.diagnostics is not None:
        diag_path = outdir / f"diagnostics_{args.point_model}_{args.dist_model}.csv"
        pd.DataFrame(result.diagnostics).to_csv(diag_path, index=False)
        print(f"- {diag_path}")

    if args.decision_eval and result.per_game is not None:
        decision = evaluate_decisions(
            result.per_game["p_over"],
            result.per_game["y_over"],
            reference_p_over=result.per_game["reference_p_over"],
            block_keys=result.per_game["block_key"],
        )
        write_decision_report(decision, outdir)
        print(f"- {outdir / 'decision_eval.json'}")
        print(f"- {outdir / 'decision_eval.md'}")

    saved_lines = [str(metrics_path), *(str(p) for p in plot_paths), str(pit_path)]
    print("\nSaved:\n" + "\n".join(f"- {p}" for p in saved_lines))


if __name__ == "__main__":
    main()
