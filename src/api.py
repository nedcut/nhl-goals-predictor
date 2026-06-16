"""
FastAPI REST API for NHL goals predictions.

Provides endpoints for getting predictions on upcoming games
and model information.

Usage:
    uvicorn src.api:app --reload
    # Then visit http://localhost:8000/docs for API documentation
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .logging_config import get_logger, setup_logging

# Setup logging before other imports
setup_logging(level="INFO")
logger = get_logger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI and Pydantic are required for the API. "
        "Install them with: pip install fastapi uvicorn pydantic"
    )

import pandas as pd

from .artifacts import ModelArtifact
from .data import build_dataset, recent_seasons
from .features import add_features
from .predict import (
    _prepare_upcoming_rows,
    fetch_upcoming_games,
    predict_games,
    predict_games_live,
)
from .probabilistic import (
    discrete_quantile_from_pmf,
    fit_nb2_alpha,
    fit_poisson_mixture,
    nb2_pmf_matrix,
    poisson_mixture_pmf_matrix,
    poisson_pmf_matrix,
    prob_over_from_pmf,
)
from .conformal import split_conformal_interval


# Pydantic models for API responses
class GamePrediction(BaseModel):
    """Prediction for a single game."""

    date: str
    home_team: str
    away_team: str
    predicted_total_goals: float


class PredictionResponse(BaseModel):
    """Response containing multiple game predictions."""

    predictions: List[GamePrediction]
    model_type: str
    model_trained_at: str
    generated_at: str
    count: int


class ModelInfo(BaseModel):
    """Information about the loaded model."""

    model_type: str
    training_date: str
    mae: Optional[float]
    rmse: Optional[float]
    baseline_mae: Optional[float]
    improvement_pct: Optional[float]
    n_training_samples: int
    n_test_samples: int
    n_features: int
    feature_names: List[str]
    data_seasons: List[str]
    git_commit: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    historical_data_loaded: bool


class ProbabilisticGamePrediction(BaseModel):
    """Probabilistic prediction for a single game."""

    date: str
    home_team: str
    away_team: str
    mu: float
    over_probs: Dict[str, float]
    pi80_low: int
    pi80_high: int
    conformal90_low: float
    conformal90_high: float
    max_goals: int
    pmf: List[float]


class ProbabilisticPredictionResponse(BaseModel):
    """Response containing probabilistic predictions."""

    predictions: List[ProbabilisticGamePrediction]
    model_type: str
    model_trained_at: str
    generated_at: str
    dist_model: str
    calibration: Dict[str, float]
    count: int


class LiveGamePrediction(BaseModel):
    date: str
    home_team: str
    away_team: str
    game_state: str
    home_score: int
    away_score: int
    period: int
    clock: str
    remaining_minutes: float
    pregame_mu: float
    live_mu: float
    pregame_over_probs: Dict[str, float]
    live_over_probs: Dict[str, float]
    pi80_low: int
    pi80_high: int
    is_live_adjusted: bool


class LivePredictionResponse(BaseModel):
    predictions: List[LiveGamePrediction]
    model_type: str
    model_trained_at: str
    generated_at: str
    count: int


# Global state
_artifact: Optional[ModelArtifact] = None
_historical_df: Optional[pd.DataFrame] = None
_prob_calibration: Optional[Dict[str, float]] = None

# Default configuration
DEFAULT_MODEL_PATH = Path(os.getenv("NHL_MODEL_PATH", "models/xgboost_v1"))
DEFAULT_SEASONS = [
    season.strip()
    for season in os.getenv("NHL_HISTORICAL_SEASONS", ",".join(recent_seasons(2))).split(",")
    if season.strip()
]


def _finite_or_none(value: float) -> Optional[float]:
    """Return JSON-safe model metrics for legacy artifacts."""
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _load_startup_state() -> None:
    """Load model and historical data into module globals (called at startup)."""
    global _artifact, _historical_df, _prob_calibration

    # Load model
    model_path = DEFAULT_MODEL_PATH
    if model_path.with_suffix(".json").exists() or model_path.with_suffix(".joblib").exists():
        try:
            _artifact = ModelArtifact.load(model_path)
            logger.info("Loaded model: %s", _artifact.metadata.model_type)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model_path, e)
    else:
        logger.warning("Model not found at %s", model_path)

    # Load historical data
    try:
        _historical_df = build_dataset(DEFAULT_SEASONS, use_cache=True)
        logger.info("Loaded %d historical games", len(_historical_df))
    except Exception as e:
        logger.error("Failed to load historical data: %s", e)

    # Probabilistic calibration (best-effort; used for /predict/probabilistic and /dashboard)
    if _artifact is not None and _historical_df is not None and not _historical_df.empty:
        try:
            include_xg = any("xg" in c.lower() for c in _artifact.metadata.feature_names)
            hist = add_features(_historical_df, include_goalies=False, include_xg=include_xg).dropna().copy()
            expected = _artifact.metadata.feature_names
            hist = hist.dropna(subset=expected + ["totalGoals"]).sort_values("date").reset_index(drop=True)
            if hist.empty:
                raise ValueError("No historical rows with complete features for calibration.")

            # Use the last 20% of historical games as a calibration slice
            n_hist = len(hist)
            cal_size = max(1, int(0.2 * n_hist))
            cal = hist.iloc[n_hist - cal_size :].copy()
            X_cal = cal.reindex(columns=expected).copy()
            for col in X_cal.columns:
                if X_cal[col].isna().any():
                    m = X_cal[col].mean()
                    X_cal[col] = X_cal[col].fillna(m if pd.notna(m) else 0.0)

            y_cal = cal["totalGoals"].to_numpy(dtype=float)
            mu_cal = _artifact.predict(X_cal)

            nb2_alpha = fit_nb2_alpha(y_cal, mu_cal)
            mix_w, mix_m = fit_poisson_mixture(y_cal, mu_cal, max_goals=20)
            _, _, q90 = split_conformal_interval(y_cal, mu_cal, mu_cal, alpha=0.1, clip_lower=0.0)
            _prob_calibration = {
                "nb2_alpha": float(nb2_alpha),
                "mix_weight": float(mix_w),
                "mix_multiplier": float(mix_m),
                "conformal90_q": float(q90),
            }
            logger.info("Calibrated probabilistic params: %s", _prob_calibration)
        except Exception as e:
            logger.warning("Probabilistic calibration failed: %s", e)
            _prob_calibration = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI startup/shutdown hook (replaces deprecated on_event)."""
    _load_startup_state()
    yield


# Create FastAPI app
app = FastAPI(
    title="NHL Goals Predictor API",
    description="Predict total goals for upcoming NHL games using machine learning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and readiness."""
    historical_loaded = _historical_df is not None and not _historical_df.empty
    return HealthResponse(
        status="healthy" if _artifact is not None and historical_loaded else "degraded",
        model_loaded=_artifact is not None,
        historical_data_loaded=historical_loaded,
    )


@app.get("/predict", response_model=PredictionResponse)
async def get_predictions(
    days_ahead: int = Query(default=7, ge=1, le=30, description="Days ahead to look for games"),
):
    """Get predictions for upcoming NHL games.

    Parameters
    ----------
    days_ahead : int
        Number of days ahead to look for games (1-30).

    Returns
    -------
    PredictionResponse
        Predictions for upcoming games.
    """
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if _historical_df is None or _historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded")

    # Fetch upcoming games
    upcoming = fetch_upcoming_games(days_ahead)

    if upcoming.empty:
        return PredictionResponse(
            predictions=[],
            model_type=_artifact.metadata.model_type,
            model_trained_at=_artifact.metadata.training_date,
            generated_at=datetime.now().isoformat(),
            count=0,
        )

    # Make predictions
    results = predict_games(upcoming, DEFAULT_MODEL_PATH, _historical_df)

    if results.empty:
        return PredictionResponse(
            predictions=[],
            model_type=_artifact.metadata.model_type,
            model_trained_at=_artifact.metadata.training_date,
            generated_at=datetime.now().isoformat(),
            count=0,
        )

    # Convert to response format
    predictions = [
        GamePrediction(
            date=row["date"],
            home_team=row["homeTeam"],
            away_team=row["awayTeam"],
            predicted_total_goals=row["predicted_total_goals"],
        )
        for _, row in results.iterrows()
    ]

    return PredictionResponse(
        predictions=predictions,
        model_type=_artifact.metadata.model_type,
        model_trained_at=_artifact.metadata.training_date,
        generated_at=datetime.now().isoformat(),
        count=len(predictions),
    )


@app.get("/predict/probabilistic", response_model=ProbabilisticPredictionResponse)
async def get_probabilistic_predictions(
    days_ahead: int = Query(default=7, ge=1, le=30, description="Days ahead to look for games"),
    dist_model: str = Query(default="nb2", pattern="^(poisson|nb2|poisson_mixture)$"),
    max_goals: int = Query(default=20, ge=10, le=40),
    thresholds: List[float] = Query(default=[5.5, 6.5, 7.5], description="Over/under thresholds"),
):
    """Probabilistic predictions with a full PMF and over/under probabilities."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _historical_df is None or _historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded")
    if _prob_calibration is None:
        raise HTTPException(status_code=503, detail="Probabilistic calibration not available")

    upcoming = fetch_upcoming_games(days_ahead)
    if upcoming.empty:
        return ProbabilisticPredictionResponse(
            predictions=[],
            model_type=_artifact.metadata.model_type,
            model_trained_at=_artifact.metadata.training_date,
            generated_at=datetime.now().isoformat(),
            dist_model=dist_model,
            calibration=_prob_calibration,
            count=0,
        )

    # Compute features for upcoming games using historical context
    upcoming = _prepare_upcoming_rows(upcoming)

    hist = _historical_df.copy()
    hist["_is_upcoming"] = False
    combined = pd.concat([hist, upcoming], ignore_index=True).drop_duplicates(subset=["gamePk"], keep="first")
    include_xg = any("xg" in c.lower() for c in _artifact.metadata.feature_names)
    combined = add_features(combined, include_goalies=False, include_xg=include_xg)
    up = combined[combined["_is_upcoming"] == True].copy()  # noqa: E712
    if up.empty:
        raise HTTPException(status_code=503, detail="Could not compute features for upcoming games")

    expected = _artifact.metadata.feature_names
    X = up.reindex(columns=expected).copy()
    for col in X.columns:
        if X[col].isna().any():
            m = X[col].mean()
            X[col] = X[col].fillna(m if pd.notna(m) else 0.0)

    mu = _artifact.predict(X).astype(float)
    if dist_model == "poisson":
        pmf = poisson_pmf_matrix(mu, max_goals=max_goals)
    elif dist_model == "nb2":
        pmf = nb2_pmf_matrix(mu, alpha=float(_prob_calibration["nb2_alpha"]), max_goals=max_goals)
    else:
        pmf = poisson_mixture_pmf_matrix(
            mu,
            weight=float(_prob_calibration["mix_weight"]),
            multiplier=float(_prob_calibration["mix_multiplier"]),
            max_goals=max_goals,
        )

    q90 = float(_prob_calibration["conformal90_q"])
    pi80_low = discrete_quantile_from_pmf(pmf, 0.10).astype(int)
    pi80_high = discrete_quantile_from_pmf(pmf, 0.90).astype(int)

    preds: List[ProbabilisticGamePrediction] = []
    for i, (_, row) in enumerate(up.iterrows()):
        over_probs = {f">{t:g}": float(prob_over_from_pmf(pmf[i : i + 1], threshold=t)[0]) for t in thresholds}
        preds.append(
            ProbabilisticGamePrediction(
                date=str(row["date"]),
                home_team=str(row["homeTeam"]),
                away_team=str(row["awayTeam"]),
                mu=float(mu[i]),
                over_probs=over_probs,
                pi80_low=int(pi80_low[i]),
                pi80_high=int(pi80_high[i]),
                conformal90_low=float(max(0.0, mu[i] - q90)),
                conformal90_high=float(mu[i] + q90),
                max_goals=int(max_goals),
                pmf=pmf[i].tolist(),
            )
        )

    return ProbabilisticPredictionResponse(
        predictions=preds,
        model_type=_artifact.metadata.model_type,
        model_trained_at=_artifact.metadata.training_date,
        generated_at=datetime.now().isoformat(),
        dist_model=dist_model,
        calibration=_prob_calibration,
        count=len(preds),
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML dashboard for the next day's slate."""
    if _artifact is None or _historical_df is None or _historical_df.empty:
        raise HTTPException(status_code=503, detail="Service not ready")
    if _prob_calibration is None:
        raise HTTPException(status_code=503, detail="Probabilistic calibration not available")

    resp = await get_probabilistic_predictions(days_ahead=1, dist_model="nb2", max_goals=20, thresholds=[6.5])
    rows = []
    for g in resp.predictions:
        rows.append(
            f"<tr><td>{g.date}</td><td>{g.away_team} @ {g.home_team}</td>"
            f"<td style='text-align:right'>{g.mu:.2f}</td>"
            f"<td style='text-align:right'>{g.over_probs.get('>6.5', 0.0):.3f}</td>"
            f"<td style='text-align:right'>[{g.pi80_low}, {g.pi80_high}]</td>"
            "</tr>"
        )
    table_rows = "".join(rows) if rows else '<tr><td colspan="5">No games found.</td></tr>'

    html = f"""
    <html>
      <head>
        <title>NHL Goals Forecast</title>
        <style>
          body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
          h1 {{ margin: 0 0 8px 0; }}
          .meta {{ color: #555; margin-bottom: 16px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; }}
          th {{ text-align: left; font-weight: 600; color: #333; }}
        </style>
      </head>
      <body>
        <h1>Tonight's Slate</h1>
        <div class="meta">Model: {resp.model_type} · Dist: {resp.dist_model} · Generated: {resp.generated_at}</div>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Matchup</th>
              <th style="text-align:right">E[Goals]</th>
              <th style="text-align:right">P(&gt; 6.5)</th>
              <th style="text-align:right">PI80</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/predict/live", response_model=LivePredictionResponse)
async def get_live_predictions(
    days_ahead: int = Query(default=1, ge=1, le=7, description="Days ahead to look for games"),
    thresholds: List[float] = Query(default=[6.5], description="Over/under thresholds"),
):
    """Live-aware predictions with in-game residual-goals adjustment."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _historical_df is None or _historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded")

    upcoming = fetch_upcoming_games(days_ahead)
    if upcoming.empty:
        return LivePredictionResponse(
            predictions=[],
            model_type=_artifact.metadata.model_type,
            model_trained_at=_artifact.metadata.training_date,
            generated_at=datetime.now().isoformat(),
            count=0,
        )

    live_df = predict_games_live(
        upcoming,
        DEFAULT_MODEL_PATH,
        _historical_df,
        thresholds=thresholds,
        max_goals=20,
    )
    preds: List[LiveGamePrediction] = []
    for _, row in live_df.iterrows():
        pre_probs = {f">{t:g}": float(row.get(f"pregame_p_over_{t:g}", 0.0)) for t in thresholds}
        live_probs = {f">{t:g}": float(row.get(f"live_p_over_{t:g}", 0.0)) for t in thresholds}
        preds.append(
            LiveGamePrediction(
                date=str(row["date"]),
                home_team=str(row["homeTeam"]),
                away_team=str(row["awayTeam"]),
                game_state=str(row["gameState"]),
                home_score=int(row["homeScore"]),
                away_score=int(row["awayScore"]),
                period=int(row["period"]),
                clock=str(row["clock"]),
                remaining_minutes=float(row["remaining_minutes"]),
                pregame_mu=float(row["pregame_mu"]),
                live_mu=float(row["live_mu"]),
                pregame_over_probs=pre_probs,
                live_over_probs=live_probs,
                pi80_low=int(row["pi80_low"]),
                pi80_high=int(row["pi80_high"]),
                is_live_adjusted=bool(row["is_live_adjusted"]),
            )
        )

    return LivePredictionResponse(
        predictions=preds,
        model_type=_artifact.metadata.model_type,
        model_trained_at=_artifact.metadata.training_date,
        generated_at=datetime.now().isoformat(),
        count=len(preds),
    )


@app.get("/dashboard/live", response_class=HTMLResponse)
async def live_dashboard():
    """Live dashboard with 20-second auto-refresh."""
    if _artifact is None or _historical_df is None or _historical_df.empty:
        raise HTTPException(status_code=503, detail="Service not ready")

    resp = await get_live_predictions(days_ahead=1, thresholds=[6.5])
    rows = []
    for g in resp.predictions:
        score = f"{g.away_score}-{g.home_score}"
        rows.append(
            f"<tr><td>{g.date}</td><td>{g.away_team} @ {g.home_team}</td>"
            f"<td>{g.game_state}</td><td>{score}</td>"
            f"<td style='text-align:right'>{g.pregame_mu:.2f}</td>"
            f"<td style='text-align:right'>{g.live_mu:.2f}</td>"
            f"<td style='text-align:right'>{g.live_over_probs.get('>6.5', 0.0):.3f}</td>"
            f"<td style='text-align:right'>{g.period} {g.clock}</td></tr>"
        )

    html = f"""
    <html>
      <head>
        <title>NHL Live Goals Forecast</title>
        <style>
          body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
          h1 {{ margin: 0 0 8px 0; }}
          .meta {{ color: #555; margin-bottom: 16px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; }}
          th {{ text-align: left; font-weight: 600; color: #333; }}
        </style>
      </head>
      <body>
        <h1>Live Slate</h1>
        <div class="meta">Generated: {resp.generated_at} · Auto-refresh: 20s</div>
        <table>
          <thead>
            <tr>
              <th>Date</th><th>Matchup</th><th>State</th><th>Score</th>
              <th style="text-align:right">Pregame E[Goals]</th>
              <th style="text-align:right">Live E[Goals]</th>
              <th style="text-align:right">Live P(&gt;6.5)</th>
              <th style="text-align:right">Period/Clock</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows) if rows else '<tr><td colspan="8">No games found.</td></tr>'}
          </tbody>
        </table>
        <script>
          setTimeout(function() {{ window.location.reload(); }}, 20000);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = _artifact.metadata
    return ModelInfo(
        model_type=meta.model_type,
        training_date=meta.training_date,
        mae=_finite_or_none(meta.mae),
        rmse=_finite_or_none(meta.rmse),
        baseline_mae=_finite_or_none(meta.baseline_mae),
        improvement_pct=_finite_or_none(meta.improvement_pct),
        n_training_samples=meta.n_training_samples,
        n_test_samples=meta.n_test_samples,
        n_features=len(meta.feature_names),
        feature_names=meta.feature_names,
        data_seasons=meta.data_seasons,
        git_commit=meta.git_commit,
    )


@app.get("/model/summary")
async def get_model_summary():
    """Get a human-readable summary of the model."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"summary": _artifact.summary()}
