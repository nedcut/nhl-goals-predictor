"""
FastAPI REST API for NHL goals predictions.

Provides endpoints for getting predictions on upcoming games
and model information.

Usage:
    uvicorn src.api:app --reload
    # Then visit http://localhost:8000/docs for API documentation
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .logging_config import get_logger, setup_logging

# Setup logging before other imports
setup_logging(level="INFO")
logger = get_logger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI and Pydantic are required for the API. "
        "Install them with: pip install fastapi uvicorn pydantic"
    )

import pandas as pd

from .artifacts import ModelArtifact
from .data import build_dataset
from .predict import fetch_upcoming_games, predict_games


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
    mae: float
    rmse: float
    baseline_mae: float
    improvement_pct: float
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


# Global state
_artifact: Optional[ModelArtifact] = None
_historical_df: Optional[pd.DataFrame] = None

# Default configuration
DEFAULT_MODEL_PATH = Path("models/xgboost_v1")
DEFAULT_SEASONS = ["20232024", "20242025"]


# Create FastAPI app
app = FastAPI(
    title="NHL Goals Predictor API",
    description="Predict total goals for upcoming NHL games using machine learning",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load model and historical data on startup."""
    global _artifact, _historical_df

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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and readiness."""
    return HealthResponse(
        status="healthy" if _artifact is not None else "degraded",
        model_loaded=_artifact is not None,
        historical_data_loaded=_historical_df is not None and not _historical_df.empty,
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


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = _artifact.metadata
    return ModelInfo(
        model_type=meta.model_type,
        training_date=meta.training_date,
        mae=meta.mae,
        rmse=meta.rmse,
        baseline_mae=meta.baseline_mae,
        improvement_pct=meta.improvement_pct,
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
