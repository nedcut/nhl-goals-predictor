"""
Champion selection with weighted probabilistic scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .significance import PairedComparison, paired_bootstrap

WEIGHTS = {
    "mae": 0.35,
    "crps": 0.30,
    "dist_nll": 0.20,
    "over_brier": 0.15,
}

# Lower = simpler/cheaper. Used when the champion margin is not significant.
MODEL_COMPLEXITY = {
    "team_strength": 0,
    "double_poisson": 1,  # reserved for future; fine if model absent
    "poisson_glm": 2,
    "xgb_current": 3,
    "xgb_tuned": 4,
}
_DEFAULT_COMPLEXITY = 100
SELECTION_POLICY = "significance_prefer_simpler"

# Maps weighted-score component -> the per-game array key produced by
# time_series_cv_forecast (CVForecastResult.per_game).
_COMPONENT_TO_PER_GAME = {
    "mae": "abs_error",
    "crps": "crps",
    "dist_nll": "dist_nll",
    "over_brier": "over_brier",
}


def model_complexity(name: str) -> int:
    """Return complexity rank for a model name (lower is simpler)."""
    return MODEL_COMPLEXITY.get(name, _DEFAULT_COMPLEXITY)


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
    """Pure weighted-score ranking (no significance / complexity demotion)."""
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


def select_champion_with_significance(
    ranking: list[dict[str, Any]],
    significance: Optional[PairedComparison],
    *,
    score_reason: str,
) -> dict[str, Any]:
    """Apply significance-aware champion selection with simpler-model preference.

    Ranking remains the weighted-score order. When a paired bootstrap between the
    provisional winner and runner-up is available and **not** significant, crown
    the simpler of the two (by ``MODEL_COMPLEXITY``). Equal complexity keeps the
    score leader. When significance is missing or the margin is significant,
    keep the weighted-score winner.
    """
    raw_leader = ranking[0]
    runner = ranking[1] if len(ranking) > 1 else None
    champion = raw_leader
    demoted = False
    rationale = score_reason

    if significance is not None and runner is not None and not significance.significant:
        leader_c = model_complexity(raw_leader["model"])
        runner_c = model_complexity(runner["model"])
        if runner_c < leader_c:
            champion = runner
            demoted = True
            rationale = (
                f"Weighted-score leader was {raw_leader['model']} "
                f"({raw_leader['weighted_score']:.4f}) vs {runner['model']} "
                f"({runner['weighted_score']:.4f}), but the paired-bootstrap margin "
                f"is not significant (p={significance.p_value:.3f}). "
                f"Preferring simpler model `{runner['model']}` "
                f"(complexity {runner_c} < {leader_c})."
            )
        else:
            # Score leader is already simpler or equal complexity.
            rationale = (
                f"{score_reason} Margin vs {runner['model']} is not statistically "
                f"significant (p={significance.p_value:.3f}); keeping "
                f"`{raw_leader['model']}` "
                f"(complexity {leader_c} <= {runner_c})."
            )
    elif significance is not None and runner is not None and significance.significant:
        rationale = (
            f"{score_reason} Paired-bootstrap confirms the margin over "
            f"{runner['model']} is significant (p={significance.p_value:.3f})."
        )

    return {
        "champion": champion,
        "raw_score_leader": raw_leader,
        "demoted": demoted,
        "rationale": rationale,
        "selection_policy": SELECTION_POLICY,
    }


def per_game_weighted_scores(
    per_game: Dict[str, Any],
    baseline: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Map each game_key to its weighted score under the same formula as
    ``weighted_score``, but evaluated per game.

    Each component is normalized by the baseline's *aggregate* metric (a
    constant), so the mean of these per-game scores reconstructs the model's
    aggregate weighted score. Returning a key->score dict lets two models be
    paired on the games they share.
    """
    w = weights or WEIGHTS
    keys = per_game["game_key"]
    score = np.zeros(len(keys), dtype=float)
    for component, pg_key in _COMPONENT_TO_PER_GAME.items():
        denom = max(baseline[component], 1e-9)
        score += w[component] * (np.asarray(per_game[pg_key], dtype=float) / denom)
    return {str(k): float(s) for k, s in zip(keys, score)}


def compare_models_significance(
    per_game_a: Dict[str, Any],
    per_game_b: Dict[str, Any],
    baseline: Dict[str, float],
    *,
    name_a: str,
    name_b: str,
    weights: Dict[str, float] | None = None,
    n_boot: int = 5000,
    seed: int = 42,
) -> Optional[PairedComparison]:
    """Paired-bootstrap comparison of two models' per-game weighted scores.

    Returns None when the two models share fewer than two games (e.g. they were
    evaluated on different row sets and cannot be paired honestly).
    """
    scores_a = per_game_weighted_scores(per_game_a, baseline, weights)
    scores_b = per_game_weighted_scores(per_game_b, baseline, weights)

    shared = sorted(set(scores_a) & set(scores_b))
    if len(shared) < 2:
        return None

    a = np.array([scores_a[k] for k in shared], dtype=float)
    b = np.array([scores_b[k] for k in shared], dtype=float)
    return paired_bootstrap(a, b, name_a=name_a, name_b=name_b, n_boot=n_boot, seed=seed)


def write_champion_reports(
    *,
    candidates: Dict[str, Dict[str, float]],
    output_dir: Path = Path("reports"),
    context: Dict[str, Any] | None = None,
    per_game_map: Dict[str, Dict[str, Any]] | None = None,
    fold_std: Dict[str, Dict[str, float]] | None = None,
    baseline_name: str = "team_strength",
) -> dict[str, Any]:
    """Write champion json + markdown reports.

    When ``per_game_map`` is supplied (per-model CVForecastResult.per_game dicts),
    the weighted-score leader is compared against the runner-up with a paired
    bootstrap. If that margin is not significant, the named champion is demoted
    to the simpler of the two (``selection_policy=significance_prefer_simpler``).
    Ranking by weighted score is unchanged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = choose_champion(candidates)
    ranking = decision["ranking"]
    provisional_name = decision["winner"]["model"]
    runner = ranking[1] if len(ranking) > 1 else None

    # Significance is always computed on the score leader vs score runner-up
    # (the provisional comparison), even if we later demote the champion.
    significance: Optional[PairedComparison] = None
    if per_game_map and runner is not None and baseline_name in candidates:
        runner_name = runner["model"]
        if provisional_name in per_game_map and runner_name in per_game_map:
            significance = compare_models_significance(
                per_game_map[provisional_name],
                per_game_map[runner_name],
                candidates[baseline_name],
                name_a=provisional_name,
                name_b=runner_name,
            )

    selection = select_champion_with_significance(
        ranking,
        significance,
        score_reason=decision["reason"],
    )
    champion = selection["champion"]
    raw_score_leader = selection["raw_score_leader"]

    payload = {
        "weights": WEIGHTS,
        "context": context or {},
        "candidates": candidates,
        "fold_std": fold_std or {},
        "ranking": ranking,
        "champion": champion,
        "raw_score_leader": raw_score_leader,
        "rationale": selection["rationale"],
        "selection_policy": selection["selection_policy"],
        "champion_vs_runner_up": significance.to_dict() if significance else None,
    }

    json_path = output_dir / "champion_model_report.json"
    md_path = output_dir / "champion_model_report.md"
    json_path.write_text(json.dumps(payload, indent=2))

    rows = []
    for row in ranking:
        std = (fold_std or {}).get(row["model"], {})
        mae_std = f" ±{std['mae']:.4f}" if "mae" in std else ""
        rows.append(
            f"| {row['model']} | {row['weighted_score']:.4f} | {row['mae']:.4f}{mae_std} | "
            f"{row['crps']:.4f} | {row['dist_nll']:.4f} | {row['over_brier']:.4f} |"
        )

    # Label the significance section against the score pairing (provisional).
    score_runner = runner["model"] if runner is not None else "n/a"
    sig_lines: list[str] = [
        "",
        f"## Statistical Significance (score leader vs runner-up: "
        f"{raw_score_leader['model']} vs {score_runner})",
    ]
    if significance is not None:
        sig_lines.append(
            f"- Paired bootstrap over {significance.n_games} shared games "
            f"({'SIGNIFICANT' if significance.significant else 'NOT significant'} at 95%)."
        )
        sig_lines.append(f"- {significance.verdict}")
    else:
        sig_lines.append("- Not computed (per-game scores unavailable or models not paired).")

    champion_lines = [
        "",
        "## Champion",
        f"- Selection policy: `{SELECTION_POLICY}`",
        f"- Champion: **{champion['model']}**",
        f"- Weighted-score leader: **{raw_score_leader['model']}**",
        f"- Rationale: {selection['rationale']}",
    ]
    if selection["demoted"]:
        champion_lines.append(
            f"- Note: demoted from `{raw_score_leader['model']}` to simpler "
            f"`{champion['model']}` because the margin was not significant."
        )

    md = "\n".join(
        [
            "# Champion Model Report",
            "",
            "## Weighted Probabilistic Objective",
            "- Score = 0.35*(MAE/base) + 0.30*(CRPS/base) + 0.20*(Dist NLL/base) + 0.15*(Brier/base)",
            f"- Baseline: {baseline_name}",
            f"- Selection policy: when the score-leader vs runner-up margin is not "
            f"significant, prefer the simpler model (`{SELECTION_POLICY}`).",
            "",
            "## Candidate Ranking",
            "| Model | Weighted Score | MAE (±fold std) | CRPS | Dist NLL | Brier (>6.5) |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            *sig_lines,
            *champion_lines,
        ]
    )
    md_path.write_text(md)
    return payload
