"""
CLI: explainability + feature stability.

Examples:
  python -m src.explain --seasons 20222023 20232024 20242025 --outdir reports/explain
  python -m src.explain --model models/xgboost_v1 --seasons 20232024 20242025
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .artifacts import ModelArtifact
from .data import build_dataset
from .explainability import (
    feature_stability_matrix,
    plot_stability_heatmap,
    shap_summary_plot_xgb,
    train_xgb_per_season_importance,
)
from .features import add_features, feature_fill_values, impute_features
from .logging_config import setup_logging


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Explainability and feature stability tools")
    parser.add_argument("--seasons", nargs="+", default=["20232024", "20242025"])
    parser.add_argument("--model", type=Path, default=None, help="Optional model artifact path (without extension)")
    parser.add_argument("--outdir", type=Path, default=Path("reports/explain"))
    parser.add_argument("--shap-sample", type=int, default=500, help="Max rows for SHAP")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    df = build_dataset(args.seasons, use_cache=True)
    df = add_features(df, include_goalies=False).dropna().copy()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Feature stability across seasons
    imp_df = train_xgb_per_season_importance(df, seasons=list(args.seasons))
    imp_path = args.outdir / "feature_importance_by_season.csv"
    imp_df.to_csv(imp_path)

    stability = feature_stability_matrix(imp_df)
    stability_path = args.outdir / "feature_stability_spearman.csv"
    stability.to_csv(stability_path)

    heatmap_path = args.outdir / "feature_stability_heatmap.png"
    plot_stability_heatmap(stability, out_path=heatmap_path)

    saved = [imp_path, stability_path, heatmap_path]

    # Optional SHAP on a provided artifact
    if args.model is not None and not args.skip_shap:
        artifact = ModelArtifact.load(args.model)
        expected = artifact.metadata.feature_names
        fills = feature_fill_values(df, expected)
        X = impute_features(df.reindex(columns=expected).copy(), fills)

        if len(X) > args.shap_sample:
            X = X.sample(n=args.shap_sample, random_state=42)

        shap_path = args.outdir / "shap_summary.png"
        try:
            shap_summary_plot_xgb(artifact.model, X, out_path=shap_path)
            saved.append(shap_path)
        except ImportError as e:
            print(f"Skipping SHAP: {e}")

    print("Saved:")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()
