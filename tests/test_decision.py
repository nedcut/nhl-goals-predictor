"""Unit tests for decision / edge evaluation against a totals line."""

import numpy as np
import pytest


def test_outcomes_vs_line_65():
    from src.decision import outcomes_vs_line

    totals = np.array([5, 6, 7, 6.5, 8, 0])
    y = outcomes_vs_line(totals, line=6.5)
    # Strict greater-than: 6 and 6.5 are UNDER, 7+ is OVER
    np.testing.assert_array_equal(y, np.array([0, 0, 1, 0, 1, 0]))


def test_model_edge_signs():
    from src.decision import model_edge

    p = np.array([0.7, 0.5, 0.3])
    edge = model_edge(p, 0.5)
    np.testing.assert_allclose(edge, np.array([0.2, 0.0, -0.2]))
    assert edge[0] > 0  # model more bullish OVER
    assert edge[2] < 0  # model more bullish UNDER


def test_no_bet_when_abs_edge_below_min():
    from src.decision import flat_stake_pnl

    # Edges are ±0.03 and 0 — with min_edge=0.05, nothing qualifies
    p = np.array([0.53, 0.50, 0.47])
    y = np.array([1, 0, 0])
    out = flat_stake_pnl(p, y, line_prob_over=0.5, min_edge=0.05, stake=1.0)
    assert out["n_bets"] == 0
    assert out["total_pnl"] == 0.0
    assert out["roi"] == 0.0


def test_min_edge_filters_reduce_n_bets():
    from src.decision import flat_stake_pnl

    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, size=500)
    y = (rng.random(500) < p).astype(int)

    loose = flat_stake_pnl(p, y, line_prob_over=0.5, min_edge=0.0)
    mid = flat_stake_pnl(p, y, line_prob_over=0.5, min_edge=0.05)
    tight = flat_stake_pnl(p, y, line_prob_over=0.5, min_edge=0.15)

    assert loose["n_bets"] >= mid["n_bets"] >= tight["n_bets"]
    assert tight["n_bets"] < loose["n_bets"]


def test_perfect_model_non_negative_roi():
    from src.decision import flat_stake_pnl

    # Perfect binary foresight: p=1 when OVER, p=0 when UNDER
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    p = y.astype(float)
    out = flat_stake_pnl(p, y, line_prob_over=0.5, min_edge=0.0, stake=1.0)
    # Every bet wins at even money: +1 per bet
    assert out["n_bets"] == len(y)
    assert out["n_wins"] == len(y)
    assert out["hit_rate"] == pytest.approx(1.0)
    assert out["roi"] == pytest.approx(1.0)
    assert out["total_pnl"] > 0


def test_random_p_near_half_roi_near_zero():
    from src.decision import flat_stake_pnl

    rng = np.random.default_rng(42)
    n = 5000
    # Fair coin outcomes; model is pure noise around 0.5
    y = rng.integers(0, 2, size=n)
    p = rng.normal(0.5, 0.05, size=n).clip(0.01, 0.99)
    out = flat_stake_pnl(p, y, line_prob_over=0.5, min_edge=0.0, stake=1.0)
    # Noise edge vs fair coin should not produce large ROI
    assert abs(out["roi"]) < 0.08


def test_edge_bucket_summary_counts():
    from src.decision import edge_bucket_summary

    p = np.linspace(0.2, 0.8, 100)
    y = (p > 0.5).astype(int)
    buckets = edge_bucket_summary(p, y, line_prob_over=0.5, n_buckets=5)
    assert len(buckets) == 5
    assert sum(b["n"] for b in buckets) == 100
    # Highest-edge bucket should have high observed over rate for this synthetic y
    assert buckets[-1]["observed_over_rate"] >= buckets[0]["observed_over_rate"]


def test_write_decision_report(tmp_path):
    from src.decision import write_decision_report

    p = np.array([0.8, 0.7, 0.2, 0.3, 0.55])
    y = np.array([1, 1, 0, 0, 1])
    payload = write_decision_report(
        p,
        y,
        line=6.5,
        line_prob_over=0.5,
        output_dir=tmp_path,
        context={"point_model": "test"},
    )
    assert (tmp_path / "decision_eval.json").exists()
    assert (tmp_path / "decision_eval.md").exists()
    assert "flat_stake_by_min_edge" in payload
    assert "0" in payload["flat_stake_by_min_edge"]
    assert "0.02" in payload["flat_stake_by_min_edge"]
    assert "0.05" in payload["flat_stake_by_min_edge"]
    assert "synthetic" in payload["disclaimer"].lower()
    md = (tmp_path / "decision_eval.md").read_text()
    assert "Disclaimer" in md
    assert "Educational" in md or "educational" in md
