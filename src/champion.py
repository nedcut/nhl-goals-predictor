"""
Champion selection with weighted probabilistic scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


WEIGHTS = {
    "mae": 0.35,
    "crps": 0.30,
    "dist_nll": 0.20,
    "over_brier": 0.15,
}


def weighted_score(
    metrics: Dict[str, float],
    baseline: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> float:
    """Compute weighted normalized score (lower is better)."""
    w = weights or WEIGHTS
    return float(
        w["mae"] * (metrics["mae"] / max(baseline["mae"], 1e-9))
        + w["crps"] * (metrics["crps"] / max(baseline["crps"], 1e-9))
        + w["dist_nll"] * (metrics["dist_nll"] / max(baseline["dist_nll"], 1e-9))
        + w["over_brier"] * (metrics["over_brier"] / max(baseline["over_brier"], 1e-9))
    )


def rank_candidates(
    candidates: Dict[str, Dict[str, float]],
    *,
    baseline_name: str = "team_strength",
    include_poisson_glm: bool = True,
) -> list[dict[str, Any]]:
    """Rank candidate models by weighted score, then MAE, then CRPS."""
    pool = dict(candidates)
    if not include_poisson_glm and "poisson_glm" in pool:
        pool.pop("poisson_glm")

    if baseline_name not in pool:
        raise ValueError(f"baseline_name='{baseline_name}' not found in candidates")
    baseline = pool[baseline_name]

    rows: list[dict[str, Any]] = []
    for name, metrics in pool.items():
        row = {
            "model": name,
            "weighted_score": weighted_score(metrics, baseline),
            "mae": float(metrics["mae"]),
            "crps": float(metrics["crps"]),
            "dist_nll": float(metrics["dist_nll"]),
            "over_brier": float(metrics["over_brier"]),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["weighted_score"], r["mae"], r["crps"]))
    return rows


def choose_champion(candidates: Dict[str, Dict[str, float]]) -> dict[str, Any]:
    ranking = rank_candidates(candidates)
    winner = ranking[0]
    runner = ranking[1] if len(ranking) > 1 else None
    reason = "Lowest weighted probabilistic score."
    if runner is not None:
        reason = (
            f"Lowest weighted score ({winner['weighted_score']:.4f}) vs "
            f"{runner['model']} ({runner['weighted_score']:.4f}); tie-breakers MAE then CRPS."
        )
    return {"winner": winner, "ranking": ranking, "reason": reason}


def write_champion_reports(
    *,
    candidates: Dict[str, Dict[str, float]],
    output_dir: Path = Path("reports"),
    context: Dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write champion json + markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = choose_champion(candidates)
    payload = {
        "weights": WEIGHTS,
        "context": context or {},
        "candidates": candidates,
        "ranking": decision["ranking"],
        "champion": decision["winner"],
        "rationale": decision["reason"],
    }

    json_path = output_dir / "champion_model_report.json"
    md_path = output_dir / "champion_model_report.md"
    json_path.write_text(json.dumps(payload, indent=2))

    rows = []
    for row in decision["ranking"]:
        rows.append(
            f"| {row['model']} | {row['weighted_score']:.4f} | {row['mae']:.4f} | "
            f"{row['crps']:.4f} | {row['dist_nll']:.4f} | {row['over_brier']:.4f} |"
        )
    md = "\n".join(
        [
            "# Champion Model Report",
            "",
            "## Weighted Probabilistic Objective",
            "- Score = 0.35*(MAE/base) + 0.30*(CRPS/base) + 0.20*(Dist NLL/base) + 0.15*(Brier/base)",
            "- Baseline: team_strength",
            "",
            "## Candidate Ranking",
            "| Model | Weighted Score | MAE | CRPS | Dist NLL | Brier (>6.5) |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Champion",
            f"- Winner: **{decision['winner']['model']}**",
            f"- Rationale: {decision['reason']}",
        ]
    )
    md_path.write_text(md)
    return payload

