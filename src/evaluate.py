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

from .data import build_dataset
from .evaluation import time_series_cv_forecast
from .features import add_features
from .logging_config import setup_logging


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
    parser = argparse.ArgumentParser(description="Evaluate probabilistic forecasts with time-series CV")
    parser.add_argument("--seasons", nargs="+", default=["20232024", "20242025"])
    parser.add_argument("--point-model", choices=["xgb", "poisson_glm", "team_strength"], default="xgb")
    parser.add_argument("--dist-model", choices=["poisson", "nb2", "poisson_mixture"], default="nb2")
    parser.add_argument("--threshold", type=float, default=6.5, help="Over/under threshold (e.g., 6.5)")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--cal-fraction", type=float, default=0.2)
    parser.add_argument("--max-goals", type=int, default=20)
    parser.add_argument("--outdir", type=Path, default=Path("reports"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    df = build_dataset(args.seasons, use_cache=True)
    df = add_features(df, include_goalies=False)

    result = time_series_cv_forecast(
        df,
        point_model=args.point_model,
        dist_model=args.dist_model,
        threshold=args.threshold,
        n_splits=args.splits,
        cal_fraction=args.cal_fraction,
        max_goals=args.max_goals,
    )

    metrics = result.metrics_mean
    print("\nMean CV metrics")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:>14}: {v:.4f}")
    print("=" * 60)

    # Save metrics + folds
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "point_model": result.point_model,
        "dist_model": result.dist_model,
        "threshold": result.threshold,
        "max_goals": result.max_goals,
        "metrics_mean": metrics,
        "folds": [f.__dict__ for f in result.folds],
        "reliability_bins": [b.__dict__ for b in result.reliability_bins],
        "pit_values": result.pit_values,
    }
    metrics_path = outdir / f"cv_{args.point_model}_{args.dist_model}.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    # Save reliability plot
    plot_path = outdir / f"reliability_over_{args.threshold:g}_{args.point_model}_{args.dist_model}.png"
    _plot_reliability(
        result.reliability_bins,
        title=f"Reliability: P(total goals > {args.threshold:g})",
        out_path=plot_path,
    )

    pit_path = outdir / f"pit_{args.point_model}_{args.dist_model}.png"
    _plot_pit(
        result.pit_values,
        title="Randomized PIT (distribution calibration)",
        out_path=pit_path,
    )

    print(f"\nSaved:\n- {metrics_path}\n- {plot_path}\n- {pit_path}")


if __name__ == "__main__":
    main()
