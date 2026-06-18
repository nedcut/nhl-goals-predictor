from __future__ import annotations

import numpy as np

from src.live import apply_live_residual_update, parse_clock_to_minutes


def test_residual_update_remaining_zero_collapses_to_current_total():
    result = apply_live_residual_update(
        mu_pregame=6.4,
        current_home_goals=3,
        current_away_goals=2,
        period=3,
        clock="00:00",
        game_state="LIVE",
        alpha_calibrated=0.2,
        max_goals=20,
    )
    pmf = result["pmf"]
    assert np.isclose(pmf.sum(), 1.0)
    assert pmf[5] == 1.0
    assert result["remaining_minutes"] == 0.0


def test_residual_update_large_goal_diff_increases_pace_multiplier():
    base = apply_live_residual_update(
        mu_pregame=6.0,
        current_home_goals=2,
        current_away_goals=2,
        period=2,
        clock="10:00",
        game_state="LIVE",
        alpha_calibrated=0.2,
        max_goals=20,
    )
    high_diff = apply_live_residual_update(
        mu_pregame=6.0,
        current_home_goals=6,
        current_away_goals=1,
        period=2,
        clock="10:00",
        game_state="LIVE",
        alpha_calibrated=0.2,
        max_goals=20,
    )
    assert high_diff["pace_mult"] > base["pace_mult"]
    assert high_diff["mu_remaining"] > base["mu_remaining"]


def test_invalid_clock_is_handled_without_crash():
    assert np.isnan(parse_clock_to_minutes("invalid"))
    result = apply_live_residual_update(
        mu_pregame=6.0,
        current_home_goals=1,
        current_away_goals=1,
        period=2,
        clock="invalid",
        game_state="LIVE",
        alpha_calibrated=0.2,
        max_goals=20,
    )
    assert np.isclose(result["pmf"].sum(), 1.0)
