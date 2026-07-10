"""Tests for the double-Poisson attack/defense rate model."""

from __future__ import annotations

import numpy as np
import pytest

from src.double_poisson import DoublePoissonConfig, DoublePoissonModel
from src.evaluation import time_series_cv_forecast


def test_fit_predict_mu_shape_and_positive(sample_game_data):
    model = DoublePoissonModel().fit(sample_game_data)
    mu = model.predict_mu(sample_game_data)

    assert mu.shape == (len(sample_game_data),)
    assert np.all(np.isfinite(mu))
    assert np.all(mu > 0)


def test_rates_sum_to_mu(sample_game_data):
    model = DoublePoissonModel(DoublePoissonConfig(alpha=1.0)).fit(sample_game_data)
    lambda_home, lambda_away = model.predict_rates(sample_game_data)
    mu = model.predict_mu(sample_game_data)

    assert lambda_home.shape == mu.shape
    assert lambda_away.shape == mu.shape
    assert np.all(np.isfinite(lambda_home))
    assert np.all(np.isfinite(lambda_away))
    assert np.all(lambda_home > 0)
    assert np.all(lambda_away > 0)
    np.testing.assert_allclose(lambda_home + lambda_away, mu, rtol=1e-10)


def test_fit_requires_score_columns(sample_game_data):
    df = sample_game_data.drop(columns=["homeScore", "awayScore"])
    with pytest.raises(ValueError, match="homeScore/awayScore"):
        DoublePoissonModel().fit(df)


def test_unknown_teams_at_predict_do_not_crash(sample_game_data):
    model = DoublePoissonModel().fit(sample_game_data)
    unseen = sample_game_data.head(3).copy()
    unseen["homeTeam"] = "Atlantis Aquanauts"
    unseen["awayTeam"] = "Mars Rovers"

    mu = model.predict_mu(unseen)
    assert mu.shape == (3,)
    assert np.all(np.isfinite(mu))
    assert np.all(mu > 0)


def test_time_series_cv_double_poisson(sample_game_data):
    result = time_series_cv_forecast(
        sample_game_data,
        point_model="double_poisson",
        dist_model="nb2",
        n_splits=3,
        threshold=6.5,
    )
    metrics = result.metrics_mean
    for key in ("mae", "rmse", "crps", "dist_nll", "over_brier"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert result.point_model == "double_poisson"
    assert len(result.folds) == 3


def test_predict_before_fit_raises(sample_game_data):
    model = DoublePoissonModel()
    with pytest.raises(ValueError, match="not fit"):
        model.predict_mu(sample_game_data)
