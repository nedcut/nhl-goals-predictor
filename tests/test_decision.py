from __future__ import annotations

import numpy as np

from src.decision import evaluate_decisions, flat_stake_pnl, model_edge


def test_model_edge_requires_aligned_explicit_reference():
    edge = model_edge([0.6, 0.4], [0.55, 0.45])
    np.testing.assert_allclose(edge, [0.05, -0.05])


def test_constant_reference_control_places_no_bets():
    reference = np.array([0.42, 0.42, 0.58, 0.58])
    result = flat_stake_pnl(
        reference,
        [0, 1, 1, 0],
        reference_p_over=reference,
        min_edge=0.0,
    )
    assert result["n_bets"] == 0
    assert result["roi"] == 0.0


def test_decision_eval_includes_week_block_uncertainty():
    p = np.array([0.65, 0.35] * 10)
    ref = np.full(20, 0.5)
    outcomes = np.array([1, 0] * 10)
    blocks = np.repeat([f"w{i}" for i in range(10)], 2)
    report = evaluate_decisions(
        p,
        outcomes,
        reference_p_over=ref,
        block_keys=blocks,
        min_edges=[0.05],
    )
    strategy = report["strategies"]["0.05"]
    assert strategy["n_bets"] == 20
    assert strategy["roi"] == 1.0
    assert strategy["n_blocks"] == 10
    assert report["constant_reference_control"]["n_bets"] == 0
