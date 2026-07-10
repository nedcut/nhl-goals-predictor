"""
Decision / edge evaluation against a total-goals line.

Proper scoring rules measure forecast quality. This module turns probabilistic
over/under forecasts into simple *decision* metrics: edge vs a reference line
probability, flat-stake PnL/ROI under fair odds, and edge-bucket summaries.

The default reference market is a synthetic fair coin at the line (p=0.5). That
is intentionally market-agnostic and **educational only** — not a betting
recommendation and not a claim about real sportsbook prices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def outcomes_vs_line(
    total_goals: np.ndarray | Sequence[float],
    line: float = 6.5,
) -> np.ndarray:
    """Binary OVER outcomes: 1 if total goals > line, else 0.

    Parameters
    ----------
    total_goals
        Realized goal totals (integers or floats).
    line
        Totals line threshold (e.g. 6.5). Strict greater-than matches the
        usual half-line OVER definition used elsewhere in the pipeline.
    """
    y = np.asarray(total_goals, dtype=float)
    return (y > float(line)).astype(int)


def model_edge(
    p_over: np.ndarray | Sequence[float],
    line_prob_over: float | np.ndarray | Sequence[float],
) -> np.ndarray:
    """Model probability minus reference market probability of OVER.

    Positive edge means the model is more bullish on OVER than the reference;
    negative means more bullish on UNDER.
    """
    p = np.asarray(p_over, dtype=float)
    ref = np.asarray(line_prob_over, dtype=float)
    return p - ref


def flat_stake_pnl(
    p_over: np.ndarray | Sequence[float],
    y_over: np.ndarray | Sequence[int],
    *,
    line_prob_over: float = 0.5,
    min_edge: float = 0.0,
    stake: float = 1.0,
) -> dict[str, float]:
    """Flat-stake PnL betting OVER/UNDER when |edge| meets ``min_edge``.

    Strategy
    --------
    - edge = p_over - line_prob_over
    - Bet OVER when edge >= min_edge
    - Bet UNDER when edge <= -min_edge
    - Otherwise no bet

    Fair-odds pricing at reference probability ``p = line_prob_over``:
    - OVER win:  +stake * (1 - p) / p
    - OVER lose: -stake
    - UNDER win: +stake * p / (1 - p)
    - UNDER lose: -stake

    Returns
    -------
    dict with n_bets, n_wins, hit_rate, total_pnl, roi, avg_edge.
    When no bets are placed, rates and ROI are 0.0 and avg_edge is 0.0.
    """
    p = np.asarray(p_over, dtype=float)
    y = np.asarray(y_over, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"p_over and y_over must align, got {p.shape} vs {y.shape}")
    if p.ndim != 1:
        raise ValueError("p_over and y_over must be 1D")
    if stake <= 0:
        raise ValueError("stake must be positive")
    if min_edge < 0:
        raise ValueError("min_edge must be >= 0")

    ref = float(line_prob_over)
    if not (0.0 < ref < 1.0):
        raise ValueError("line_prob_over must be in (0, 1)")

    edge = model_edge(p, ref)
    bet_over = edge >= float(min_edge)
    bet_under = edge <= -float(min_edge)
    # When min_edge == 0, edge==0 would match both; prefer no double-bet.
    if min_edge == 0.0:
        bet_under = edge < 0.0
        bet_over = edge > 0.0
        # edge exactly 0: no bet
        # (already excluded by strict inequalities)

    take = bet_over | bet_under
    n_bets = int(np.sum(take))
    if n_bets == 0:
        return {
            "n_bets": 0.0,
            "n_wins": 0.0,
            "hit_rate": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "avg_edge": 0.0,
        }

    # Per-bet PnL under fair odds
    over_odds = (1.0 - ref) / ref
    under_odds = ref / (1.0 - ref)

    pnl = np.zeros(p.shape[0], dtype=float)
    wins = np.zeros(p.shape[0], dtype=bool)

    # OVER bets
    if np.any(bet_over):
        over_win = bet_over & (y == 1)
        over_lose = bet_over & (y == 0)
        pnl[over_win] = stake * over_odds
        pnl[over_lose] = -stake
        wins[over_win] = True

    # UNDER bets
    if np.any(bet_under):
        under_win = bet_under & (y == 0)
        under_lose = bet_under & (y == 1)
        pnl[under_win] = stake * under_odds
        pnl[under_lose] = -stake
        wins[under_win] = True

    n_wins = int(np.sum(wins & take))
    total_pnl = float(np.sum(pnl[take]))
    # Edge in the direction of the bet taken
    directed_edge = np.where(bet_over, edge, np.where(bet_under, -edge, 0.0))
    avg_edge = float(np.mean(directed_edge[take]))
    total_staked = float(n_bets) * float(stake)
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    return {
        "n_bets": float(n_bets),
        "n_wins": float(n_wins),
        "hit_rate": float(n_wins / n_bets),
        "total_pnl": total_pnl,
        "roi": float(roi),
        "avg_edge": avg_edge,
    }


def edge_bucket_summary(
    p_over: np.ndarray | Sequence[float],
    y_over: np.ndarray | Sequence[int],
    *,
    line_prob_over: float = 0.5,
    n_buckets: int = 5,
) -> list[dict[str, Any]]:
    """Summarize outcomes by equal-width buckets of model edge.

    Each bucket reports count, edge bounds, mean edge, mean model p_over,
    observed OVER rate, and flat-stake ROI at min_edge=0 within the bucket
    (bets only the side implied by the signed edge for games in that bucket).
    """
    p = np.asarray(p_over, dtype=float)
    y = np.asarray(y_over, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"p_over and y_over must align, got {p.shape} vs {y.shape}")
    if p.ndim != 1:
        raise ValueError("p_over and y_over must be 1D")
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")

    edge = model_edge(p, line_prob_over)
    n = edge.shape[0]
    if n == 0:
        return []

    lo = float(np.min(edge))
    hi = float(np.max(edge))
    # Degenerate range: single bucket covering the constant edge
    if hi - lo < 1e-15:
        edges_bounds = np.array([lo, hi + 1e-9])
        n_buckets = 1
    else:
        edges_bounds = np.linspace(lo, hi, n_buckets + 1)

    rows: list[dict[str, Any]] = []
    for i in range(n_buckets):
        left = float(edges_bounds[i])
        right = float(edges_bounds[i + 1])
        if i < n_buckets - 1:
            mask = (edge >= left) & (edge < right)
        else:
            mask = (edge >= left) & (edge <= right)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bucket": i,
                    "edge_low": left,
                    "edge_high": right,
                    "n": 0,
                    "mean_edge": float("nan"),
                    "mean_p_over": float("nan"),
                    "observed_over_rate": float("nan"),
                    "roi": float("nan"),
                }
            )
            continue

        p_b = p[mask]
        y_b = y[mask]
        e_b = edge[mask]
        pnl = flat_stake_pnl(
            p_b,
            y_b,
            line_prob_over=line_prob_over,
            min_edge=0.0,
            stake=1.0,
        )
        rows.append(
            {
                "bucket": i,
                "edge_low": left,
                "edge_high": right,
                "n": count,
                "mean_edge": float(np.mean(e_b)),
                "mean_p_over": float(np.mean(p_b)),
                "observed_over_rate": float(np.mean(y_b)),
                "roi": float(pnl["roi"]),
            }
        )
    return rows


def write_decision_report(
    p_over: np.ndarray | Sequence[float],
    y_over: np.ndarray | Sequence[int],
    *,
    line: float = 6.5,
    line_prob_over: float = 0.5,
    output_dir: Path | str = Path("reports"),
    min_edges: Sequence[float] = (0.0, 0.02, 0.05),
    n_buckets: int = 5,
    context: dict[str, Any] | None = None,
    stake: float = 1.0,
) -> dict[str, Any]:
    """Write decision-evaluation JSON + Markdown under ``output_dir``.

    Files:
    - ``decision_eval.json``
    - ``decision_eval.md``

    Includes flat-stake ROI at several min_edge thresholds, an edge-bucket
    table, and an educational synthetic-market disclaimer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p = np.asarray(p_over, dtype=float)
    y = np.asarray(y_over, dtype=float)

    flat_by_edge: dict[str, dict[str, float]] = {}
    for me in min_edges:
        key = f"{float(me):g}"
        flat_by_edge[key] = flat_stake_pnl(
            p,
            y,
            line_prob_over=line_prob_over,
            min_edge=float(me),
            stake=stake,
        )

    buckets = edge_bucket_summary(
        p,
        y,
        line_prob_over=line_prob_over,
        n_buckets=n_buckets,
    )

    disclaimer = (
        "Synthetic fair reference market (default p=0.5 at the line). "
        "Educational decision metrics only — not betting advice and not "
        "a claim about real sportsbook lines or odds."
    )

    payload: dict[str, Any] = {
        "line": float(line),
        "line_prob_over": float(line_prob_over),
        "reference_market": "synthetic_fair",
        "n_games": int(p.shape[0]),
        "stake": float(stake),
        "flat_stake_by_min_edge": flat_by_edge,
        "edge_buckets": buckets,
        "disclaimer": disclaimer,
        "context": context or {},
    }

    json_path = output_dir / "decision_eval.json"
    md_path = output_dir / "decision_eval.md"
    json_path.write_text(json.dumps(payload, indent=2))

    flat_rows = []
    for me, stats in flat_by_edge.items():
        flat_rows.append(
            f"| {me} | {int(stats['n_bets'])} | {int(stats['n_wins'])} | "
            f"{stats['hit_rate']:.3f} | {stats['total_pnl']:+.3f} | "
            f"{stats['roi']:+.3f} | {stats['avg_edge']:+.4f} |"
        )

    bucket_rows = []
    for b in buckets:
        if b["n"] == 0:
            bucket_rows.append(
                f"| {b['bucket']} | {b['edge_low']:+.3f} | {b['edge_high']:+.3f} | "
                f"0 | — | — | — | — |"
            )
        else:
            bucket_rows.append(
                f"| {b['bucket']} | {b['edge_low']:+.3f} | {b['edge_high']:+.3f} | "
                f"{b['n']} | {b['mean_edge']:+.4f} | {b['mean_p_over']:.3f} | "
                f"{b['observed_over_rate']:.3f} | {b['roi']:+.3f} |"
            )

    ctx = context or {}
    ctx_lines = [f"- {k}: `{v}`" for k, v in ctx.items()] if ctx else ["- (none)"]

    md = "\n".join(
        [
            "# Decision / Edge Evaluation",
            "",
            "## Setup",
            f"- Totals line: **{line:g}**",
            f"- Reference P(OVER): **{line_prob_over:g}** (synthetic fair market)",
            f"- Games: **{int(p.shape[0])}**",
            f"- Unit stake: **{stake:g}**",
            "",
            "## Context",
            *ctx_lines,
            "",
            "## Flat-stake ROI by min edge",
            "| min_edge | n_bets | n_wins | hit_rate | total_pnl | ROI | avg_edge |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *flat_rows,
            "",
            "## Edge buckets",
            "| bucket | edge_low | edge_high | n | mean_edge | mean_p_over | observed_over | ROI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *bucket_rows,
            "",
            "## Disclaimer",
            disclaimer,
            "",
        ]
    )
    md_path.write_text(md)
    return payload
