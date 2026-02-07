"""
Hierarchical-ish team strength baseline models.

These models are intentionally simple and "deployable": they rely only on team
identities (and optional season), using regularization to shrink team effects.

This provides a strong, interpretable baseline for probabilistic forecasting:
- Inputs are robust (no feature engineering required)
- Outputs are Poisson means (mu) suitable for distributional forecasts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class TeamStrengthConfig:
    """Configuration for the team-strength Poisson model."""

    alpha: float = 1.0  # L2 strength; higher => more shrinkage (more "hierarchical")
    include_season: bool = False


class TeamStrengthPoissonModel:
    """Regularized Poisson regression on team IDs (shrinkage baseline)."""

    def __init__(self, config: Optional[TeamStrengthConfig] = None):
        self.config = config or TeamStrengthConfig()
        self.pipeline: Pipeline | None = None

    def fit(self, df: pd.DataFrame) -> "TeamStrengthPoissonModel":
        required = {"homeTeam", "awayTeam", "totalGoals"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        features = ["homeTeam", "awayTeam"]
        if self.config.include_season:
            if "season" not in df.columns:
                raise ValueError("include_season=True requires a 'season' column")
            features.append("season")

        pre = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    features,
                )
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        model = PoissonRegressor(alpha=self.config.alpha, max_iter=2000)
        self.pipeline = Pipeline([("pre", pre), ("model", model)])
        self.pipeline.fit(df[features], df["totalGoals"].astype(float))
        return self

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Model is not fit yet")

        features = ["homeTeam", "awayTeam"]
        if self.config.include_season:
            features.append("season")
        mu = self.pipeline.predict(df[features])
        return np.clip(mu, 1e-9, None)

