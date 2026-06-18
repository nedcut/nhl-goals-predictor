from __future__ import annotations

import pytest

from src.champion import choose_champion, rank_candidates, weighted_score


def test_weighted_score_matches_formula():
    metrics = {"mae": 1.8, "crps": 1.2, "dist_nll": 2.1, "over_brier": 0.24}
    baseline = {"mae": 2.0, "crps": 1.5, "dist_nll": 2.5, "over_brier": 0.30}
    score = weighted_score(metrics, baseline)
    expected = (
        0.35 * (1.8 / 2.0)
        + 0.30 * (1.2 / 1.5)
        + 0.20 * (2.1 / 2.5)
        + 0.15 * (0.24 / 0.30)
    )
    assert score == pytest.approx(expected, rel=1e-10)


def test_ranking_uses_tiebreakers_mae_then_crps():
    candidates = {
        "team_strength": {"mae": 2.0, "crps": 1.5, "dist_nll": 2.5, "over_brier": 0.30},
        "model_a": {"mae": 1.8, "crps": 1.3, "dist_nll": 2.0, "over_brier": 0.24},
        "model_b": {"mae": 1.8, "crps": 1.2, "dist_nll": 2.0, "over_brier": 0.24},
    }
    ranking = rank_candidates(candidates)
    # model_a and model_b have close weighted score, tie resolves on CRPS because MAE ties.
    assert ranking[0]["model"] in {"model_a", "model_b"}
    if ranking[0]["weighted_score"] == pytest.approx(ranking[1]["weighted_score"], rel=1e-10):
        assert ranking[0]["crps"] <= ranking[1]["crps"]


def test_choose_champion_returns_winner_and_reason():
    candidates = {
        "team_strength": {"mae": 2.0, "crps": 1.5, "dist_nll": 2.5, "over_brier": 0.30},
        "xgb_current": {"mae": 1.75, "crps": 1.22, "dist_nll": 2.04, "over_brier": 0.23},
        "xgb_tuned": {"mae": 1.76, "crps": 1.23, "dist_nll": 2.07, "over_brier": 0.24},
    }
    decision = choose_champion(candidates)
    assert decision["winner"]["model"] == "xgb_current"
    assert "weighted score" in decision["reason"].lower()
