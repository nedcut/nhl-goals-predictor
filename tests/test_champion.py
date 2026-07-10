from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.champion import (
    SELECTION_POLICY,
    choose_champion,
    rank_candidates,
    select_champion_from_equivalence_set,
    select_champion_with_significance,
    weighted_score,
    write_champion_reports,
)
from src.significance import PairedComparison


def _metrics(mae: float, crps: float = 1.2, dist_nll: float = 2.0, over_brier: float = 0.24):
    return {"mae": mae, "crps": crps, "dist_nll": dist_nll, "over_brier": over_brier}


def _sig(*, significant: bool, p_value: float = 0.4) -> PairedComparison:
    return PairedComparison(
        n_games=200,
        mean_a=0.95,
        mean_b=0.96,
        mean_diff=-0.01,
        ci_low=-0.05 if not significant else -0.08,
        ci_high=0.03 if not significant else -0.01,
        p_value=p_value if not significant else 0.01,
        significant=significant,
        better_model="xgb_tuned" if significant else None,
        verdict="fake verdict",
    )


def _fake_per_game(n: int = 40, seed: int = 0, bias: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "game_key": np.array([f"g{i}" for i in range(n)], dtype=object),
        "abs_error": rng.uniform(0.5, 3.0, n) + bias,
        "crps": rng.uniform(0.8, 2.0, n) + bias,
        "dist_nll": rng.uniform(1.8, 2.8, n) + bias,
        "over_brier": rng.uniform(0.1, 0.4, n),
    }


def test_weighted_score_matches_formula():
    metrics = {"mae": 1.8, "crps": 1.2, "dist_nll": 2.1, "over_brier": 0.24}
    baseline = {"mae": 2.0, "crps": 1.5, "dist_nll": 2.5, "over_brier": 0.30}
    score = weighted_score(metrics, baseline)
    expected = 0.35 * (1.8 / 2.0) + 0.30 * (1.2 / 1.5) + 0.20 * (2.1 / 2.5) + 0.15 * (0.24 / 0.30)
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


def test_select_demotes_to_simpler_when_not_significant():
    # xgb_tuned slightly better on score; xgb_current is simpler.
    ranking = rank_candidates(
        {
            "team_strength": _metrics(2.0, 1.5, 2.5, 0.30),
            "xgb_tuned": _metrics(1.74, 1.20, 2.00, 0.23),
            "xgb_current": _metrics(1.75, 1.22, 2.04, 0.24),
        }
    )
    assert ranking[0]["model"] == "xgb_tuned"
    assert ranking[1]["model"] == "xgb_current"

    selection = select_champion_with_significance(
        ranking,
        _sig(significant=False, p_value=0.42),
        score_reason="Lowest weighted score.",
    )
    assert selection["champion"]["model"] == "xgb_current"
    assert selection["raw_score_leader"]["model"] == "xgb_tuned"
    assert selection["demoted"] is True
    assert selection["selection_policy"] == SELECTION_POLICY
    assert "simpler" in selection["rationale"].lower()


def test_select_keeps_complex_when_significant():
    ranking = rank_candidates(
        {
            "team_strength": _metrics(2.0, 1.5, 2.5, 0.30),
            "xgb_tuned": _metrics(1.70, 1.15, 1.95, 0.22),
            "xgb_current": _metrics(1.80, 1.25, 2.10, 0.25),
        }
    )
    assert ranking[0]["model"] == "xgb_tuned"

    selection = select_champion_with_significance(
        ranking,
        _sig(significant=True, p_value=0.01),
        score_reason="Lowest weighted score.",
    )
    assert selection["champion"]["model"] == "xgb_tuned"
    assert selection["raw_score_leader"]["model"] == "xgb_tuned"
    assert selection["demoted"] is False
    assert "significant" in selection["rationale"].lower()


def test_full_equivalence_set_prefers_simplest_not_just_runner_up():
    ranking = rank_candidates(
        {
            "team_strength": _metrics(2.0, 1.5, 2.5, 0.30),
            "xgb_tuned": _metrics(1.70, 1.15, 1.95, 0.22),
            "xgb_current": _metrics(1.71, 1.16, 1.96, 0.22),
            "double_poisson": _metrics(1.72, 1.17, 1.97, 0.23),
        }
    )
    comparisons = [
        {"candidate": row["model"], "significant_adjusted": False} for row in ranking[1:]
    ]
    selection = select_champion_from_equivalence_set(
        ranking,
        comparisons,
        score_reason="Lowest weighted score.",
    )
    assert selection["raw_score_leader"]["model"] == "xgb_tuned"
    assert selection["champion"]["model"] == "team_strength"
    assert set(selection["equivalence_set"]) == {
        "xgb_tuned",
        "xgb_current",
        "double_poisson",
        "team_strength",
    }


def test_write_champion_reports_without_per_game_uses_score_only(tmp_path: Path):
    candidates = {
        "team_strength": _metrics(2.0, 1.5, 2.5, 0.30),
        "xgb_tuned": _metrics(1.74, 1.20, 2.00, 0.23),
        "xgb_current": _metrics(1.75, 1.22, 2.04, 0.24),
    }
    payload = write_champion_reports(candidates=candidates, output_dir=tmp_path)
    # No per_game_map => no significance => weighted-score champion.
    assert payload["champion"]["model"] == "xgb_tuned"
    assert payload["raw_score_leader"]["model"] == "xgb_tuned"
    assert payload["champion_vs_runner_up"] is None
    assert payload["selection_policy"] == SELECTION_POLICY
    md = (tmp_path / "champion_model_report.md").read_text()
    assert "xgb_tuned" in md
    assert SELECTION_POLICY in md


def test_write_champion_reports_demotes_when_not_significant(tmp_path: Path, monkeypatch):
    candidates = {
        "team_strength": _metrics(2.0, 1.5, 2.5, 0.30),
        "xgb_tuned": _metrics(1.74, 1.20, 2.00, 0.23),
        "xgb_current": _metrics(1.75, 1.22, 2.04, 0.24),
    }
    per_game_map = {
        "xgb_tuned": _fake_per_game(seed=1, bias=0.0),
        "xgb_current": _fake_per_game(seed=2, bias=0.01),
    }

    def _fake_compare(*_args, **_kwargs):
        return _sig(significant=False, p_value=0.55)

    monkeypatch.setattr("src.champion.compare_models_significance", _fake_compare)
    payload = write_champion_reports(
        candidates=candidates,
        output_dir=tmp_path,
        per_game_map=per_game_map,
    )
    assert payload["raw_score_leader"]["model"] == "xgb_tuned"
    assert payload["champion"]["model"] == "xgb_current"
    assert payload["champion_vs_runner_up"] is not None
    assert payload["champion_vs_runner_up"]["significant"] is False
    assert "simpler" in payload["rationale"].lower()
    md = (tmp_path / "champion_model_report.md").read_text()
    assert "demoted" in md.lower() or "simpler" in md.lower()
    assert "**xgb_current**" in md


def test_write_champion_reports_keeps_complex_when_significant(tmp_path: Path, monkeypatch):
    candidates = {
        "team_strength": _metrics(2.0, 1.5, 2.5, 0.30),
        "xgb_tuned": _metrics(1.70, 1.15, 1.95, 0.22),
        "xgb_current": _metrics(1.80, 1.25, 2.10, 0.25),
    }
    per_game_map = {
        "xgb_tuned": _fake_per_game(seed=3, bias=0.0),
        "xgb_current": _fake_per_game(seed=4, bias=0.2),
    }

    def _fake_compare(*_args, **_kwargs):
        return _sig(significant=True, p_value=0.01)

    monkeypatch.setattr("src.champion.compare_models_significance", _fake_compare)
    payload = write_champion_reports(
        candidates=candidates,
        output_dir=tmp_path,
        per_game_map=per_game_map,
    )
    assert payload["champion"]["model"] == "xgb_tuned"
    assert payload["raw_score_leader"]["model"] == "xgb_tuned"
    assert payload["champion_vs_runner_up"]["significant"] is True
