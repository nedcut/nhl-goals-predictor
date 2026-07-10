"""Render all public benchmark claims from one release manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

README_START = "<!-- BENCHMARK_RELEASE_START -->"
README_END = "<!-- BENCHMARK_RELEASE_END -->"


def _rank_rows(manifest: dict[str, Any]) -> list[str]:
    return [
        f"| {row['model']} | {row['weighted_score']:.4f} | {row['mae']:.4f} | "
        f"{row['crps']:.4f} | {row['dist_nll']:.4f} | {row['over_brier']:.4f} |"
        for row in manifest["selection"]["ranking"]
    ]


def render_readme_results(manifest: dict[str, Any]) -> str:
    champion = manifest["selection"]["champion"]
    protocol = manifest["protocol"]
    cohort = manifest["cohort"]
    return "\n".join(
        [
            README_START,
            "## Current benchmark release",
            "",
            f"- Release: `{manifest['release_id']}` using protocol `{protocol['version']}`",
            f"- Untouched holdout: `{protocol['holdout_season']}` regular season "
            f"({cohort['holdout_rows']} games after the common-feature filter)",
            f"- Champion: **`{champion['model']}`** under "
            f"`{manifest['selection']['selection_policy']}`",
            f"- Holdout MAE: **{champion['mae']:.4f}**; CRPS: **{champion['crps']:.4f}**; "
            f"Brier (>6.5): **{champion['over_brier']:.4f}**",
            f"- Indistinguishable set: `{', '.join(manifest['selection']['equivalence_set'])}`",
            "",
            "The release manifest and technical report are tracked under "
            "`reports/releases/benchmark-v1/`. Decision/ROI diagnostics are opt-in and are "
            "not used for champion selection.",
            README_END,
        ]
    )


def render_model_card(manifest: dict[str, Any]) -> str:
    protocol = manifest["protocol"]
    champion = manifest["selection"]["champion"]
    artifact = manifest.get("production_artifact", {})
    return "\n".join(
        [
            "# Model Card",
            "",
            "## Intended use",
            "",
            "- Probabilistic NHL regular-season total-goals forecasting for analytical use.",
            "- Pregame mean, full discrete PMF, over/under probabilities, and intervals.",
            "- Not a betting recommendation; decision diagnostics require explicit references.",
            "",
            "## Authoritative release",
            "",
            f"- Release: `{manifest['release_id']}`",
            f"- Protocol: `{protocol['version']}`",
            f"- Champion: `{champion['model']}`",
            f"- Artifact: `{artifact.get('artifact_id', 'not built')}`",
            f"- Data fingerprint: `{manifest['data_quality']['data_fingerprint']}`",
            f"- Feature-schema hash: `{artifact.get('feature_schema_hash', 'not built')}`",
            "",
            "## Evaluation design",
            "",
            f"- Training/tuning seasons: {', '.join(protocol['training_seasons'])}.",
            f"- Untouched holdout: {protocol['holdout_season']}.",
            f"- Primary cohort: game types {', '.join(protocol['game_types'])}; "
            "preseason and playoffs are excluded.",
            "- Every candidate is evaluated on one common complete-game cohort.",
            "- Distribution calibration uses only the final historical calibration slice.",
            "- Uncertainty uses paired ISO-week block bootstrap with Holm adjustment.",
            "- The simplest model in the full statistically indistinguishable set is selected.",
            "",
            "## Holdout performance",
            "",
            "| Model | Weighted score | MAE | CRPS | Dist NLL | Brier (>6.5) |",
            "|---|---:|---:|---:|---:|---:|",
            *_rank_rows(manifest),
            "",
            "## Feature and data scope",
            "",
            f"- Common feature count: {manifest['cohort']['feature_count']}.",
            "- Core-v1 intentionally excludes goalie and xG features until historical "
            "coverage and source freshness pass the same release checks.",
            "- Source scores must satisfy unique game IDs, valid dates/types, and "
            "home + away = total consistency.",
            "",
            "## Monitoring",
            "",
            "- The API loads only a promoted release-grade registry artifact.",
            "- SQLite monitoring deduplicates by game, artifact, and forecast kind.",
            "- Realized monitoring includes MAE, RMSE, bias, CRPS, NLL, and mid-PIT diagnostics.",
            "- Drift is reported only against references saved in the artifact; unavailable "
            "feature drift is labeled unavailable.",
            "",
            "## Limitations",
            "",
            "- The holdout establishes predictive comparison, not causal effects or profit.",
            "- Week blocking reduces but does not eliminate team/schedule dependence.",
            "- Lineup and confirmed-goalie information are outside the core-v1 release.",
            "- A new season requires a new locked release rather than silently updating claims.",
            "",
        ]
    )


def render_technical_report(manifest: dict[str, Any]) -> str:
    protocol = manifest["protocol"]
    selection = manifest["selection"]
    champion = selection["champion"]
    comparisons = selection.get("leader_comparisons", [])
    comparison_rows = [
        f"| {row['candidate']} | {row['mean_diff']:+.5f} | "
        f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] | "
        f"{row['p_value_adjusted']:.3f} |"
        for row in comparisons
    ]
    quality = manifest["data_quality"]
    return "\n".join(
        [
            "# NHL Total-Goals Benchmark v1",
            "",
            "## Technical summary",
            "",
            f"The locked {protocol['holdout_season']} regular-season holdout selects "
            f"**{champion['model']}**. The raw score leader was "
            f"**{selection['raw_score_leader']['model']}**, but the full "
            f"Holm-adjusted week-block comparison set was "
            f"**{', '.join(selection['equivalence_set'])}**; the protocol therefore "
            "chooses the simplest indistinguishable candidate.",
            "",
            "## Candidate performance on the untouched holdout",
            "",
            "| Model | Weighted score | MAE | CRPS | Dist NLL | Brier (>6.5) |",
            "|---|---:|---:|---:|---:|---:|",
            *_rank_rows(manifest),
            "",
            "These metrics all use the same games and the same team-strength normalization. "
            "Lower is better. Small score differences are not treated as model improvements "
            "unless the blocked comparison supports that conclusion.",
            "",
            "## The simpler choice survives multiplicity-adjusted uncertainty",
            "",
            "| Candidate vs score leader | Mean score difference | 95% interval | Holm p |",
            "|---|---:|---:|---:|",
            *comparison_rows,
            "",
            "The bootstrap resamples ISO weeks, preserving within-week schedule dependence. "
            "Holm adjustment controls the family of leader-versus-candidate comparisons.",
            "",
            "## Scope, data, and metric definitions",
            "",
            f"- Training/tuning: {', '.join(protocol['training_seasons'])}.",
            f"- Holdout: {protocol['holdout_season']}.",
            f"- Cohort: regular season only; {manifest['cohort']['training_rows']} training "
            f"rows and {manifest['cohort']['holdout_rows']} holdout rows after a common "
            "complete-feature filter.",
            f"- Data as of: {quality['date_max']}; fingerprint `{quality['data_fingerprint']}`.",
            "- Weighted score: 35% normalized MAE, 30% CRPS, 20% distribution NLL, "
            "15% Brier at 6.5.",
            "",
            "## Validation and data quality",
            "",
            f"The release source contains {quality['row_count']} rows, including "
            f"{quality['primary_row_count']} primary-cohort rows. Duplicate game IDs: "
            f"{quality['duplicate_game_ids']}; score mismatches: {quality['score_mismatches']}; "
            f"invalid dates: {quality['invalid_dates']}; invalid game types: "
            f"{quality['invalid_game_types']}. Release readiness: **{quality['ready']}**.",
            "",
            "## Methodology",
            "",
            "Hyperparameters are tuned only on historical seasons using expanding-window CV. "
            "The selected parameters are then frozen and every candidate is scored exactly "
            "once on the later holdout. Rolling features use prior games only. Distribution "
            "parameters are fit on a historical calibration slice, never holdout outcomes.",
            "",
            "## Limitations and robustness",
            "",
            "- Statistical indistinguishability is failure to detect a difference, not proof "
            "that models are mathematically equivalent.",
            "- Week blocking addresses short-range dependence but not every shared-team effect.",
            "- Core-v1 excludes goalie and xG inputs because their release-grade longitudinal "
            "coverage has not yet been established.",
            "- The benchmark supports forecast-quality claims, not sportsbook profitability.",
            "",
            "## Recommended next steps",
            "",
            "1. Serve only the promoted release artifact and accumulate a deduplicated ledger.",
            "2. Re-run the locked protocol for each completed season before changing champion.",
            "3. Admit enriched goalie/xG candidates only after coverage and freshness gates pass.",
            "4. Evaluate real market lines separately; never mix synthetic ROI with championing.",
            "",
            "## Further questions",
            "",
            "- Does a team-cluster or moving-block bootstrap change the equivalence set?",
            "- Do enriched features improve an untouched season once source coverage is complete?",
            "- How stable are CRPS and calibration by month and rest/fatigue segment?",
            "",
        ]
    )


def _replace_readme_block(text: str, block: str) -> str:
    if README_START not in text or README_END not in text:
        raise ValueError("README benchmark release markers are missing")
    prefix, remainder = text.split(README_START, 1)
    _, suffix = remainder.split(README_END, 1)
    return prefix.rstrip() + "\n\n" + block + "\n\n" + suffix.lstrip()


def write_release_documents(
    manifest: dict[str, Any],
    *,
    release_dir: Path,
    readme_path: Path = Path("README.md"),
    model_card_path: Path = Path("MODEL_CARD.md"),
) -> None:
    """Write every public benchmark document from the same manifest."""
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "technical_report.md").write_text(render_technical_report(manifest))
    (release_dir / "source_notes.json").write_text(
        json.dumps(
            {
                "release_id": manifest["release_id"],
                "git_commit": manifest.get("git_commit"),
                "data_fingerprint": manifest["data_quality"]["data_fingerprint"],
                "protocol": manifest["protocol"],
                "chart_map": [
                    {
                        "section": "Candidate performance",
                        "question": "How do candidate weighted scores compare?",
                        "family": "comparison and ranking",
                        "type": "bar",
                        "fields": ["model", "weighted_score", "mae", "crps"],
                        "takeaway": "Model score differences are small and require uncertainty checks.",
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    model_card_path.write_text(render_model_card(manifest))
    readme_path.write_text(
        _replace_readme_block(readme_path.read_text(), render_readme_results(manifest))
    )
