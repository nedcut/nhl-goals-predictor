from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import pandas as pd

import src.api as api


@dataclass
class _DummyMeta:
    model_type: str = "XGBoost"
    training_date: str = "2026-02-14T00:00:00"
    feature_names: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = ["home_avg_GF", "away_avg_GF"]


@dataclass
class _DummyArtifact:
    metadata: _DummyMeta = field(default_factory=_DummyMeta)


def test_predict_live_contract(monkeypatch):
    monkeypatch.setattr(api, "_artifact", _DummyArtifact())
    monkeypatch.setattr(
        api,
        "_historical_df",
        pd.DataFrame(
            [
                {
                    "gamePk": 1,
                    "season": "20242025",
                    "date": "2024-10-10",
                    "homeTeam": "Boston Bruins",
                    "awayTeam": "Toronto Maple Leafs",
                    "homeScore": 3,
                    "awayScore": 2,
                    "totalGoals": 5,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        api,
        "fetch_upcoming_games",
        lambda days_ahead: pd.DataFrame([{"gamePk": 99, "date": "2025-01-01"}]),
    )
    monkeypatch.setattr(
        api,
        "predict_games_live",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "gamePk": 99,
                    "date": "2025-01-01",
                    "homeTeam": "Boston Bruins",
                    "awayTeam": "Toronto Maple Leafs",
                    "gameState": "LIVE",
                    "homeScore": 2,
                    "awayScore": 1,
                    "period": 2,
                    "clock": "10:12",
                    "remaining_minutes": 30.2,
                    "pregame_mu": 6.2,
                    "live_mu": 5.7,
                    "pregame_p_over_6.5": 0.45,
                    "live_p_over_6.5": 0.38,
                    "pi80_low": 4,
                    "pi80_high": 8,
                    "is_live_adjusted": True,
                }
            ]
        ),
    )

    resp = asyncio.run(api.get_live_predictions(days_ahead=1, thresholds=[6.5]))
    assert resp.count == 1
    row = resp.predictions[0]
    assert row.game_state == "LIVE"
    assert ">6.5" in row.pregame_over_probs
    assert row.is_live_adjusted is True


def test_dashboard_live_html(monkeypatch):
    monkeypatch.setattr(api, "_artifact", _DummyArtifact())
    monkeypatch.setattr(api, "_historical_df", pd.DataFrame([{"gamePk": 1}]))

    async def _fake_live_predictions(days_ahead, thresholds):
        return api.LivePredictionResponse(
            predictions=[
                api.LiveGamePrediction(
                    date="2025-01-01",
                    home_team="Boston Bruins",
                    away_team="Toronto Maple Leafs",
                    game_state="PRE",
                    home_score=0,
                    away_score=0,
                    period=1,
                    clock="20:00",
                    remaining_minutes=60.0,
                    pregame_mu=6.1,
                    live_mu=6.1,
                    pregame_over_probs={">6.5": 0.43},
                    live_over_probs={">6.5": 0.43},
                    pi80_low=4,
                    pi80_high=8,
                    is_live_adjusted=False,
                )
            ],
            model_type="XGBoost",
            model_trained_at="2026-02-14T00:00:00",
            generated_at="2026-02-14T01:00:00",
            count=1,
        )

    monkeypatch.setattr(
        api,
        "get_live_predictions",
        _fake_live_predictions,
    )
    html = asyncio.run(api.live_dashboard())
    assert "Auto-refresh: 20s" in html.body.decode("utf-8")
    assert "Live P(&gt;6.5)" in html.body.decode("utf-8")


def test_model_info_serializes_legacy_nan_metrics(monkeypatch):
    """Legacy artifacts should report unknown metrics instead of crashing JSON."""

    @dataclass
    class LegacyMeta:
        model_type: str = "XGBRegressor"
        training_date: str = "unknown"
        mae: float = math.nan
        rmse: float = math.nan
        baseline_mae: float = math.nan
        improvement_pct: float = math.nan
        n_training_samples: int = 0
        n_test_samples: int = 0
        feature_names: list[str] = field(default_factory=lambda: ["home_avg_GF"])
        data_seasons: list[str] = field(default_factory=list)
        git_commit: str | None = None

    @dataclass
    class LegacyArtifact:
        metadata: LegacyMeta = field(default_factory=LegacyMeta)

    monkeypatch.setattr(api, "_artifact", LegacyArtifact())
    info = asyncio.run(api.get_model_info())

    assert info.mae is None
    assert info.rmse is None
    assert info.baseline_mae is None
    assert info.improvement_pct is None


def test_health_requires_model_and_historical_data(monkeypatch):
    monkeypatch.setattr(api, "_artifact", _DummyArtifact())
    monkeypatch.setattr(api, "_historical_df", None)

    health = asyncio.run(api.health_check())

    assert health.status == "degraded"
    assert health.model_loaded is True
    assert health.historical_data_loaded is False
