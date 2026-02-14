"""
Live in-game forecast utilities.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable

import numpy as np
import requests

from .config import config
from .logging_config import get_logger
from .probabilistic import nb2_pmf_matrix

logger = get_logger(__name__)


def parse_clock_to_minutes(clock: str | None) -> float:
    """Parse MM:SS period clock into minutes remaining."""
    if not clock:
        return float("nan")
    match = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", str(clock))
    if not match:
        return float("nan")
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    if seconds < 0 or seconds > 59:
        return float("nan")
    return minutes + (seconds / 60.0)


def remaining_minutes_from_state(period: int | None, clock: str | None, game_state: str | None) -> float:
    """Estimate game minutes remaining from period + clock."""
    state = (game_state or "").upper()
    if state in {"OFF", "FINAL"}:
        return 0.0

    p = int(period or 1)
    per_remaining = parse_clock_to_minutes(clock)
    if np.isnan(per_remaining):
        return float("nan")

    if p <= 3:
        rem = (3 - p) * 20.0 + per_remaining
    elif p == 4:
        # Regular season OT length in modern NHL.
        rem = min(per_remaining, 5.0)
    else:
        rem = 0.0
    return float(np.clip(rem, 0.0, 60.0))


def apply_live_residual_update(
    *,
    mu_pregame: float,
    current_home_goals: int,
    current_away_goals: int,
    period: int | None,
    clock: str | None,
    game_state: str | None,
    alpha_calibrated: float,
    max_goals: int = 20,
) -> dict[str, Any]:
    """Residual-goals NB2 updater shifted by current in-game total."""
    current_total = int(current_home_goals) + int(current_away_goals)
    goal_diff = abs(int(current_home_goals) - int(current_away_goals))
    remaining_minutes = remaining_minutes_from_state(period, clock, game_state)
    if np.isnan(remaining_minutes):
        remaining_minutes = 0.0

    remaining_frac = remaining_minutes / 60.0
    pace_mult = 1.0 + 0.04 * goal_diff
    mu_remaining = max(0.05, float(mu_pregame) * remaining_frac * pace_mult)

    if remaining_minutes <= 0.0:
        pmf = np.zeros(max_goals + 1, dtype=float)
        pmf[min(current_total, max_goals)] = 1.0
        return {
            "pmf": pmf,
            "remaining_minutes": 0.0,
            "remaining_frac": 0.0,
            "pace_mult": pace_mult,
            "mu_remaining": 0.0,
            "mu_live": float(min(current_total, max_goals)),
        }

    rem_pmf = nb2_pmf_matrix(np.array([mu_remaining]), alpha=float(alpha_calibrated), max_goals=max_goals)[0]
    final_pmf = np.zeros(max_goals + 1, dtype=float)
    for rem_goals, p in enumerate(rem_pmf):
        final_goals = current_total + rem_goals
        if final_goals <= max_goals:
            final_pmf[final_goals] += p
        else:
            final_pmf[max_goals] += p

    final_pmf = final_pmf / max(final_pmf.sum(), 1e-12)
    mu_live = float(np.sum(np.arange(max_goals + 1) * final_pmf))
    return {
        "pmf": final_pmf,
        "remaining_minutes": float(remaining_minutes),
        "remaining_frac": float(remaining_frac),
        "pace_mult": float(pace_mult),
        "mu_remaining": float(mu_remaining),
        "mu_live": mu_live,
    }


def _safe_nested(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_gamecenter_payload(game_pk: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize NHL gamecenter payload to lightweight live state fields."""
    game_state = str(payload.get("gameState") or payload.get("gameStatus") or "UNKNOWN").upper()
    period = payload.get("period")
    if period is None:
        period = _safe_nested(payload, "periodDescriptor", "number", default=1)
    clock = _safe_nested(payload, "clock", "timeRemaining", default=None)
    if clock is None:
        clock = _safe_nested(payload, "clock", "inIntermission", default=None)
    if isinstance(clock, bool):
        clock = "00:00" if clock else None

    home_score = _safe_nested(payload, "homeTeam", "score", default=0)
    away_score = _safe_nested(payload, "awayTeam", "score", default=0)

    remaining_minutes = remaining_minutes_from_state(int(period or 1), str(clock) if clock else None, game_state)

    return {
        "gamePk": int(game_pk),
        "gameState": game_state,
        "period": int(period or 1),
        "clock": str(clock) if clock is not None else "",
        "homeScore": int(home_score or 0),
        "awayScore": int(away_score or 0),
        "remaining_minutes": float(remaining_minutes if not np.isnan(remaining_minutes) else 0.0),
    }


def fetch_live_game_state(game_pk: int) -> dict[str, Any]:
    """Fetch one game's live state from NHL gamecenter."""
    url = f"{config.data.api_base}/gamecenter/{int(game_pk)}/landing"
    response = requests.get(url, timeout=config.data.request_timeout)
    response.raise_for_status()
    payload = response.json()
    return parse_gamecenter_payload(int(game_pk), payload)


def fetch_live_states(game_pks: Iterable[int]) -> Dict[int, dict[str, Any]]:
    """Fetch live states for multiple games; failures are logged and skipped."""
    out: Dict[int, dict[str, Any]] = {}
    for game_pk in game_pks:
        try:
            out[int(game_pk)] = fetch_live_game_state(int(game_pk))
        except Exception as e:
            logger.warning("Failed to fetch live state for gamePk=%s: %s", game_pk, e)
    return out

