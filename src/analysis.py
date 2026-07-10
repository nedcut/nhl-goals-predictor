"""
Ablation and error-analysis utilities for portfolio artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .evaluation import time_series_cv_forecast
from .features import add_features


def run_ablation_study(
    raw_df: pd.DataFrame,
    *,
    point_model: str = "xgb",
    dist_model: str = "nb2",
    threshold: float = 6.5,
    n_splits: int = 5,
    include_xg: bool = True,
) -> dict[str, Any]:
    """Run fixed ablation scenarios and return metrics by scenario.

    When ``include_xg`` is False (e.g. the MoneyPuck feed is unavailable) the
    xG-requiring scenarios are run without xG so the study still completes; the
    xG dimension simply collapses rather than aborting the whole pipeline.
    """
    scenarios = [
        (
            "full_model",
            dict(
                include_xg=True,
                require_xg=True,
                include_goalies=True,
                include_interactions=True,
            ),
        ),
        ("no_xg", dict(include_xg=False, include_goalies=True, include_interactions=True)),
        (
            "no_goalie_features",
            dict(
                include_xg=True,
                require_xg=True,
                include_goalies=False,
                include_interactions=True,
            ),
        ),
        (
            "no_interactions",
            dict(
                include_xg=True,
                require_xg=True,
                include_goalies=True,
                include_interactions=False,
            ),
        ),
        (
            "no_h2h_venue",
            dict(
                include_xg=True,
                require_xg=True,
                include_goalies=True,
                include_interactions=True,
            ),
        ),
        ("team_strength", None),
    ]

    if not include_xg:
        # Neutralize xG requirements so the study runs on remaining families.
        for _name, kwargs in scenarios:
            if kwargs is not None:
                kwargs["include_xg"] = False
                kwargs.pop("require_xg", None)

    results: Dict[str, Dict[str, float]] = {}
    for name, feature_kwargs in scenarios:
        if name == "team_strength":
            cv = time_series_cv_forecast(
                raw_df,
                point_model="team_strength",
                dist_model=dist_model,
                threshold=threshold,
                n_splits=n_splits,
            )
            results[name] = cv.metrics_mean
            continue

        feat = add_features(
            raw_df, include_temporal=True, include_multi_window=True, **(feature_kwargs or {})
        )
        if name == "no_h2h_venue":
            drop_cols = [c for c in feat.columns if c.startswith("h2h_") or c.startswith("venue_")]
            feat = feat.drop(columns=drop_cols, errors="ignore")

        cv = time_series_cv_forecast(
            feat,
            point_model=point_model,  # type: ignore[arg-type]
            dist_model=dist_model,  # type: ignore[arg-type]
            threshold=threshold,
            n_splits=n_splits,
        )
        results[name] = cv.metrics_mean

    full = results["full_model"]
    deltas = {}
    for name, metrics in results.items():
        deltas[name] = {
            "delta_mae": float(metrics["mae"] - full["mae"]),
            "delta_crps": float(metrics["crps"] - full["crps"]),
            "delta_dist_nll": float(metrics["dist_nll"] - full["dist_nll"]),
            "delta_over_brier": float(metrics["over_brier"] - full["over_brier"]),
        }

    return {"metrics": results, "deltas_vs_full": deltas}


def write_ablation_report(report: dict[str, Any], *, output_dir: Path = Path("reports")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_report.json"
    md_path = output_dir / "ablation_report.md"
    json_path.write_text(json.dumps(report, indent=2))

    rows = []
    for name, metrics in report["metrics"].items():
        rows.append(
            f"| {name} | {metrics['mae']:.4f} | {metrics['crps']:.4f} | "
            f"{metrics['dist_nll']:.4f} | {metrics['over_brier']:.4f} |"
        )
    md = "\n".join(
        [
            "# Ablation Report",
            "",
            "| Scenario | MAE | CRPS | Dist NLL | Brier (>6.5) |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Notes",
            "- Deltas vs full model are included in JSON output.",
        ]
    )
    md_path.write_text(md)


def _segment_table(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(segment_col)
        .agg(
            games=("y_true", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            avg_mu=("mu_pred", "mean"),
            avg_true=("y_true", "mean"),
        )
        .reset_index()
        .sort_values("games", ascending=False)
    )
    return grouped


def write_error_analysis(
    diagnostics: list[dict[str, Any]],
    *,
    output_dir: Path = Path("reports"),
) -> None:
    """Write row-level diagnostics CSV and markdown segment summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diag = pd.DataFrame(diagnostics)
    if diag.empty:
        raise ValueError("No diagnostics rows available for error analysis.")

    diag["total_goal_bucket"] = pd.cut(
        diag["y_true"],
        bins=[-0.1, 3.5, 5.5, 7.5, 99],
        labels=["0-3", "4-5", "6-7", "8+"],
    ).astype(str)
    diag["month"] = pd.to_numeric(diag["month"], errors="coerce").fillna(-1).astype(int)
    diag["back_to_back_flag"] = np.where(diag["is_back_to_back"] == 1, "yes", "no")
    diag["confidence_decile"] = pd.qcut(
        diag["p_over_6_5"].rank(method="first"),
        q=10,
        labels=[str(i) for i in range(1, 11)],
    ).astype(str)

    csv_path = output_dir / "error_analysis.csv"
    md_path = output_dir / "error_analysis.md"
    diag.to_csv(csv_path, index=False)

    bucket_tbl = _segment_table(diag, "total_goal_bucket")
    month_tbl = _segment_table(diag, "month")
    b2b_tbl = _segment_table(diag, "back_to_back_flag")
    conf_tbl = _segment_table(diag, "confidence_decile")

    def _tbl_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows_"
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *body])

    md = "\n\n".join(
        [
            "# Error Analysis",
            "## By total-goal bucket\n" + _tbl_markdown(bucket_tbl),
            "## By month\n" + _tbl_markdown(month_tbl),
            "## By back-to-back\n" + _tbl_markdown(b2b_tbl),
            "## By confidence decile\n" + _tbl_markdown(conf_tbl),
        ]
    )
    md_path.write_text(md)
