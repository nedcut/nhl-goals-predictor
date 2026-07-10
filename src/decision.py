"""Opt-in decision evaluation against explicit, evidence-based references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _aligned_1d(values: Sequence[float] | np.ndarray, name: str, n: int | None = None):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} must have length {n}, got {len(array)}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def outcomes_vs_line(total_goals: Sequence[float] | np.ndarray, line: float) -> np.ndarray:
    """Return binary OVER outcomes for a half-goal line."""
    return (_aligned_1d(total_goals, "total_goals") > float(line)).astype(int)


def model_edge(
    p_over: Sequence[float] | np.ndarray,
    reference_p_over: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return model probability minus an aligned market/base-rate reference."""
    model = _aligned_1d(p_over, "p_over")
    reference = _aligned_1d(reference_p_over, "reference_p_over", len(model))
    if np.any((model < 0.0) | (model > 1.0)):
        raise ValueError("p_over must be in [0, 1]")
    if np.any((reference <= 0.0) | (reference >= 1.0)):
        raise ValueError("reference_p_over must be in (0, 1)")
    return model - reference


def flat_stake_pnl(
    p_over: Sequence[float] | np.ndarray,
    y_over: Sequence[int] | np.ndarray,
    *,
    reference_p_over: Sequence[float] | np.ndarray,
    min_edge: float = 0.0,
    stake: float = 1.0,
) -> dict[str, Any]:
    """Evaluate flat-stake decisions at aligned fair reference probabilities."""
    model = _aligned_1d(p_over, "p_over")
    outcomes = _aligned_1d(y_over, "y_over", len(model))
    reference = _aligned_1d(reference_p_over, "reference_p_over", len(model))
    edge = model_edge(model, reference)
    if min_edge < 0.0 or stake <= 0.0:
        raise ValueError("min_edge must be non-negative and stake must be positive")
    if np.any(~np.isin(outcomes, [0.0, 1.0])):
        raise ValueError("y_over must be binary")

    over = edge >= min_edge if min_edge > 0 else edge > 0
    under = edge <= -min_edge if min_edge > 0 else edge < 0
    take = over | under
    pnl = np.zeros(len(model), dtype=float)
    wins = (over & (outcomes == 1.0)) | (under & (outcomes == 0.0))
    pnl[over & (outcomes == 1.0)] = (
        stake * (1.0 - reference[over & (outcomes == 1.0)]) / reference[over & (outcomes == 1.0)]
    )
    pnl[over & (outcomes == 0.0)] = -stake
    pnl[under & (outcomes == 0.0)] = (
        stake * reference[under & (outcomes == 0.0)] / (1.0 - reference[under & (outcomes == 0.0)])
    )
    pnl[under & (outcomes == 1.0)] = -stake
    n_bets = int(take.sum())
    total_pnl = float(pnl[take].sum()) if n_bets else 0.0
    directed_edge = np.where(over, edge, np.where(under, -edge, 0.0))
    return {
        "n_games": int(len(model)),
        "n_bets": n_bets,
        "n_wins": int((wins & take).sum()),
        "hit_rate": float((wins & take).sum() / n_bets) if n_bets else 0.0,
        "total_pnl": total_pnl,
        "roi": float(total_pnl / (n_bets * stake)) if n_bets else 0.0,
        "avg_directed_edge": float(directed_edge[take].mean()) if n_bets else 0.0,
        "pnl_by_game": pnl,
        "bet_mask": take,
    }


def block_bootstrap_roi_interval(
    pnl_by_game: Sequence[float] | np.ndarray,
    bet_mask: Sequence[bool] | np.ndarray,
    block_keys: Sequence[object] | np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Week-block bootstrap interval for ROI, preserving within-week dependence."""
    pnl = _aligned_1d(pnl_by_game, "pnl_by_game")
    bets = np.asarray(bet_mask, dtype=bool)
    blocks = np.asarray(block_keys, dtype=object)
    if bets.shape != pnl.shape or blocks.shape != pnl.shape:
        raise ValueError("pnl_by_game, bet_mask, and block_keys must align")
    unique, inverse = np.unique(blocks.astype(str), return_inverse=True)
    if len(unique) < 2:
        raise ValueError("at least two blocks are required")
    block_pnl = np.bincount(inverse, weights=pnl, minlength=len(unique))
    block_bets = np.bincount(inverse, weights=bets.astype(float), minlength=len(unique))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    totals = block_pnl[sampled].sum(axis=1)
    stakes = block_bets[sampled].sum(axis=1)
    rois = np.divide(totals, stakes, out=np.zeros_like(totals), where=stakes > 0)
    return {
        "n_blocks": int(len(unique)),
        "ci_low": float(np.quantile(rois, 0.025)),
        "ci_high": float(np.quantile(rois, 0.975)),
    }


def evaluate_decisions(
    p_over: Sequence[float] | np.ndarray,
    y_over: Sequence[int] | np.ndarray,
    *,
    reference_p_over: Sequence[float] | np.ndarray,
    block_keys: Sequence[object] | np.ndarray,
    min_edges: Sequence[float] = (0.02, 0.05),
) -> dict[str, Any]:
    """Evaluate a model and constant-reference control at explicit edge cutoffs."""
    model = _aligned_1d(p_over, "p_over")
    reference = _aligned_1d(reference_p_over, "reference_p_over", len(model))
    rows: dict[str, Any] = {}
    for value in min_edges:
        result = flat_stake_pnl(
            model,
            y_over,
            reference_p_over=reference,
            min_edge=float(value),
        )
        interval = block_bootstrap_roi_interval(
            result.pop("pnl_by_game"),
            result.pop("bet_mask"),
            block_keys,
        )
        rows[f"{float(value):g}"] = {**result, **interval}

    control = flat_stake_pnl(
        reference,
        y_over,
        reference_p_over=reference,
        min_edge=0.0,
    )
    control.pop("pnl_by_game")
    control.pop("bet_mask")
    return {
        "reference": "training-fold empirical P(OVER)",
        "strategies": rows,
        "constant_reference_control": control,
        "disclaimer": (
            "Decision diagnostics use fair odds implied by historical training-fold base rates. "
            "They are not sportsbook backtests and are not evidence of tradable profit."
        ),
    }


def write_decision_report(payload: dict[str, Any], output_dir: Path) -> None:
    """Write opt-in machine-readable and narrative decision diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision_eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = []
    for edge, stats in payload["strategies"].items():
        rows.append(
            f"| {edge} | {stats['n_bets']} | {stats['roi']:+.3f} | "
            f"[{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}] |"
        )
    text = "\n".join(
        [
            "# Decision Evaluation",
            "",
            "This report is opt-in and separate from model championing.",
            "",
            "| Minimum edge | Bets | ROI | Week-block 95% interval |",
            "|---:|---:|---:|---:|",
            *rows,
            "",
            f"> {payload['disclaimer']}",
            "",
        ]
    )
    (output_dir / "decision_eval.md").write_text(text)
