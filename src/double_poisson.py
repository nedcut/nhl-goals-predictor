"""
Double-Poisson home/away scoring-rate model.

Generative rationale
--------------------
NHL total goals are the sum of two scoring processes: home goals and away
goals. A natural generative model treats each as Poisson with its own rate:

    homeScore ~ Poisson(λ_home)
    awayScore ~ Poisson(λ_away)
    totalGoals = homeScore + awayScore  ⇒  E[totalGoals] = λ_home + λ_away

This module fits two regularized Poisson regressions on team identity
features (``homeTeam``, ``awayTeam``):

    λ_home = E[homeScore | homeTeam, awayTeam]
    λ_away = E[awayScore | homeTeam, awayTeam]

That is *not* a shared Dixon–Coles attack/defense parameterization: home and
away scoring get separate coefficient vectors, so a team's "attack" when
playing at home is not constrained to match its away-attack coefficient.
Regularization (``alpha``) shrinks rare-team effects toward the league mean,
mirroring :class:`TeamStrengthPoissonModel`, but with separate home/away
rates rather than a single total-goals mean.

Downstream CV currently uses μ = λ_home + λ_away as the point forecast and
calibrates a total-goals distribution (NB2 / mixture) on that mean — it does
**not** yet form the total PMF by convolving the two Poissons. That remains
a natural follow-up.

This is intentionally deployable: only team IDs are required at predict time;
no engineered rolling features are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


_TEAM_FEATURES = ["homeTeam", "awayTeam"]
_REQUIRED_FIT = {"homeTeam", "awayTeam", "homeScore", "awayScore"}


@dataclass
class DoublePoissonConfig:
    """Configuration for the double-Poisson rate model."""

    alpha: float = 1.0  # L2 strength; higher => more shrinkage


class DoublePoissonModel:
    """Regularized double-Poisson attack/defense rates on team IDs.

    Fits separate Poisson models for home and away scoring rates, then
    predicts total-goals mean as μ = λ_home + λ_away.
    """

    def __init__(self, config: Optional[DoublePoissonConfig] = None):
        self.config = config or DoublePoissonConfig()
        self.home_pipeline: Pipeline | None = None
        self.away_pipeline: Pipeline | None = None

    def _make_pipeline(self) -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(_TEAM_FEATURES),
                )
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        model = PoissonRegressor(alpha=self.config.alpha, max_iter=2000)
        return Pipeline([("pre", pre), ("model", model)])

    def fit(self, df: pd.DataFrame) -> "DoublePoissonModel":
        missing = _REQUIRED_FIT - set(df.columns)
        if missing:
            raise ValueError(
                f"double_poisson fit requires homeScore/awayScore (and team IDs); "
                f"missing columns: {sorted(missing)}"
            )

        X = df[list(_TEAM_FEATURES)]
        self.home_pipeline = self._make_pipeline()
        self.away_pipeline = self._make_pipeline()
        self.home_pipeline.fit(X, df["homeScore"].astype(float))
        self.away_pipeline.fit(X, df["awayScore"].astype(float))
        return self

    def predict_rates(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (λ_home, λ_away) clipped to be strictly positive."""
        if self.home_pipeline is None or self.away_pipeline is None:
            raise ValueError("Model is not fit yet")

        X = df[list(_TEAM_FEATURES)]
        lambda_home = np.clip(self.home_pipeline.predict(X), 1e-9, None)
        lambda_away = np.clip(self.away_pipeline.predict(X), 1e-9, None)
        return lambda_home, lambda_away

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Return total-goals mean μ = λ_home + λ_away."""
        lambda_home, lambda_away = self.predict_rates(df)
        return np.clip(lambda_home + lambda_away, 1e-9, None)
