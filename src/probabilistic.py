"""
Probabilistic forecasting utilities for NHL total goals.

This module turns point predictions for expected total goals (mu) into full
distributions over integer goal totals (0, 1, 2, ...), enabling:
- Over/under probabilities (e.g., P(total_goals > 6.5))
- Proper scoring rules (log-likelihood, CRPS)
- Calibration/reliability curves for betting-style events

The code avoids SciPy to keep dependencies lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log
from typing import Iterable, Literal, Tuple

import numpy as np

DistributionKind = Literal["poisson", "nb2", "poisson_mixture"]

# Vectorized lgamma created once at import. np.vectorize rebuilds its wrapper on
# every call; np.frompyfunc builds the ufunc a single time, so reusing this is
# both faster and clearer than scattering np.vectorize(lgamma) through the module.
_lgamma = np.frompyfunc(lgamma, 1, 1)


def _lgamma_arr(x: np.ndarray) -> np.ndarray:
    """Vectorized lgamma returning a float64 array (frompyfunc yields object)."""
    return _lgamma(x).astype(float)


def _as_1d(x: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}")
    return arr


def _clip_mu(mu: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return np.clip(mu, eps, None)


def poisson_logpmf(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson log PMF for integer k and mean mu (vectorized)."""
    k = np.asarray(k, dtype=float)
    mu = _clip_mu(mu)
    return k * np.log(mu) - mu - _lgamma_arr(k + 1.0)


def nb2_logpmf(k: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Negative binomial (NB2) log PMF.

    Parameterization:
      Var(Y|mu) = mu + alpha * mu^2
    with alpha >= 0. For alpha -> 0, converges to Poisson(mu).
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if alpha == 0:
        return poisson_logpmf(k, mu)

    k = np.asarray(k, dtype=float)
    mu = _clip_mu(mu)

    r = 1.0 / alpha
    p = r / (r + mu)  # success probability in NB(r, p)
    # log pmf = lgamma(k+r) - lgamma(r) - lgamma(k+1) + r log p + k log(1-p)
    return (
        _lgamma_arr(k + r)
        - lgamma(r)
        - _lgamma_arr(k + 1.0)
        + r * np.log(p)
        + k * np.log1p(-p)
    )


def poisson_pmf_matrix(mu: np.ndarray | Iterable[float], max_goals: int = 20) -> np.ndarray:
    """Return PMF over {0..max_goals} for each mu."""
    mu = _as_1d(mu)
    ks = np.arange(max_goals + 1, dtype=float)[None, :]
    logpmf = poisson_logpmf(ks, mu[:, None])
    pmf = np.exp(logpmf)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


def nb2_pmf_matrix(
    mu: np.ndarray | Iterable[float],
    *,
    alpha: float,
    max_goals: int = 20,
) -> np.ndarray:
    """Return NB2 PMF over {0..max_goals} for each mu."""
    mu = _as_1d(mu)
    ks = np.arange(max_goals + 1, dtype=float)[None, :]
    logpmf = nb2_logpmf(ks, mu[:, None], alpha=alpha)
    pmf = np.exp(logpmf)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


def poisson_mixture_pmf_matrix(
    mu: np.ndarray | Iterable[float],
    *,
    weight: float,
    multiplier: float,
    max_goals: int = 20,
) -> np.ndarray:
    """Mixture of two Poissons centered on mu for extra dispersion.

    Components:
      lambda_low = mu / multiplier
      lambda_high = mu * multiplier
      pmf = w * Pois(lambda_high) + (1-w) * Pois(lambda_low)
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError("weight must be in [0, 1]")
    if multiplier <= 1.0:
        raise ValueError("multiplier must be > 1")

    mu = _as_1d(mu)
    mu_hi = mu * multiplier
    mu_lo = mu / multiplier
    pmf_hi = poisson_pmf_matrix(mu_hi, max_goals=max_goals)
    pmf_lo = poisson_pmf_matrix(mu_lo, max_goals=max_goals)
    pmf = weight * pmf_hi + (1.0 - weight) * pmf_lo
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


def prob_over_from_pmf(pmf: np.ndarray, threshold: float) -> np.ndarray:
    """Compute P(Y > threshold) for integer Y given PMF over {0..K}.

    Example: threshold=6.5 => P(Y >= 7).
    """
    if pmf.ndim != 2:
        raise ValueError(f"pmf must be 2D, got shape={pmf.shape}")
    k0 = int(np.floor(threshold) + 1)  # smallest integer strictly greater than threshold
    k0 = max(k0, 0)
    if k0 >= pmf.shape[1]:
        return np.zeros(pmf.shape[0], dtype=float)
    return pmf[:, k0:].sum(axis=1)


def discrete_quantile_from_pmf(pmf: np.ndarray, q: float) -> np.ndarray:
    """Smallest integer k such that CDF(k) >= q, per row."""
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    cdf = np.cumsum(pmf, axis=1)
    return (cdf >= q).argmax(axis=1)


def poisson_nll(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean negative log-likelihood under Poisson(mu)."""
    y = _as_1d(y)
    mu = _as_1d(mu)
    if len(y) != len(mu):
        raise ValueError("y and mu must have same length")
    ll = poisson_logpmf(y, mu)
    return float(-np.mean(ll))


def nb2_nll(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Mean negative log-likelihood under NB2(mu, alpha)."""
    y = _as_1d(y)
    mu = _as_1d(mu)
    if len(y) != len(mu):
        raise ValueError("y and mu must have same length")
    ll = nb2_logpmf(y, mu, alpha=alpha)
    return float(-np.mean(ll))


def fit_nb2_alpha(
    y: np.ndarray | Iterable[float],
    mu: np.ndarray | Iterable[float],
    *,
    alpha_grid: Iterable[float] | None = None,
) -> float:
    """Fit NB2 dispersion alpha by maximizing conditional log-likelihood.

    This treats mu as fixed (from any point model) and finds the alpha that best
    explains the residual dispersion beyond Poisson.
    """
    y = _as_1d(y)
    mu = _as_1d(mu)
    if len(y) != len(mu):
        raise ValueError("y and mu must have same length")

    if alpha_grid is None:
        # Wide but conservative range; hockey totals are not extremely overdispersed.
        alpha_grid = np.concatenate(
            [
                np.array([0.0]),
                np.logspace(-4, 0.7, 50),  # ~[1e-4, 5]
            ]
        )

    best_alpha = 0.0
    best_nll = float("inf")
    for a in alpha_grid:
        nll = nb2_nll(y, mu, alpha=float(a))
        if nll < best_nll:
            best_nll = nll
            best_alpha = float(a)
    return best_alpha


def fit_poisson_mixture(
    y: np.ndarray | Iterable[float],
    mu: np.ndarray | Iterable[float],
    *,
    weight_grid: Iterable[float] | None = None,
    multiplier_grid: Iterable[float] | None = None,
    max_goals: int = 20,
) -> Tuple[float, float]:
    """Fit a 2-component Poisson mixture (w, multiplier) by grid search."""
    y = _as_1d(y).astype(int)
    mu = _as_1d(mu)
    if len(y) != len(mu):
        raise ValueError("y and mu must have same length")

    if weight_grid is None:
        weight_grid = np.linspace(0.1, 0.9, 9)
    if multiplier_grid is None:
        multiplier_grid = np.linspace(1.05, 1.8, 16)

    best_w, best_m = 0.5, 1.2
    best_nll = float("inf")
    for w in weight_grid:
        for m in multiplier_grid:
            pmf = poisson_mixture_pmf_matrix(mu, weight=float(w), multiplier=float(m), max_goals=max_goals)
            # Clip to avoid log(0)
            p_y = np.take_along_axis(pmf, y[:, None].clip(0, max_goals), axis=1).squeeze(1)
            nll = float(-np.mean(np.log(np.clip(p_y, 1e-12, 1.0))))
            if nll < best_nll:
                best_nll = nll
                best_w, best_m = float(w), float(m)
    return best_w, best_m


def crps_per_game_from_pmf(pmf: np.ndarray, y: np.ndarray | Iterable[int]) -> np.ndarray:
    """Per-observation CRPS for a discrete distribution over {0..K}.

    CRPS_i = sum_k (F_i(k) - 1{k >= y_i})^2 where F is the forecast CDF. Returning
    the per-game vector (rather than only its mean) is what lets downstream code
    bootstrap paired score differences between two models on the same games.
    """
    if pmf.ndim != 2:
        raise ValueError(f"pmf must be 2D, got shape={pmf.shape}")
    y = _as_1d(y).astype(int)
    if len(y) != pmf.shape[0]:
        raise ValueError("y and pmf must have same number of rows")

    cdf = np.cumsum(pmf, axis=1)
    ks = np.arange(pmf.shape[1])[None, :]
    obs_cdf = (ks >= y[:, None]).astype(float)
    return np.sum((cdf - obs_cdf) ** 2, axis=1)


def crps_from_pmf(pmf: np.ndarray, y: np.ndarray | Iterable[int]) -> float:
    """Mean CRPS for a discrete distribution over {0..K}."""
    return float(np.mean(crps_per_game_from_pmf(pmf, y)))

def randomized_pit(
    pmf: np.ndarray,
    y: np.ndarray | Iterable[int],
    *,
    seed: int = 42,
) -> np.ndarray:
    """Randomized PIT values for a discrete distribution over {0..K}.

    For discrete Y, the randomized PIT uses:
      u = F(y-1) + v * P(Y=y),  v ~ Uniform(0,1)
    If the forecast distribution is calibrated, u is approximately Uniform(0,1).
    """
    if pmf.ndim != 2:
        raise ValueError(f"pmf must be 2D, got shape={pmf.shape}")
    y = _as_1d(y).astype(int)
    if len(y) != pmf.shape[0]:
        raise ValueError("y and pmf must have same number of rows")

    rng = np.random.default_rng(seed)
    cdf = np.cumsum(pmf, axis=1)
    K = pmf.shape[1] - 1

    y_clipped = np.clip(y, 0, K)
    p_y = np.take_along_axis(pmf, y_clipped[:, None], axis=1).squeeze(1)
    f_ym1 = np.where(
        y_clipped > 0,
        np.take_along_axis(cdf, (y_clipped - 1)[:, None], axis=1).squeeze(1),
        0.0,
    )
    v = rng.random(len(y))
    return np.clip(f_ym1 + v * p_y, 0.0, 1.0)


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of a reliability curve for a binary event."""

    p_mean: float
    frac_pos: float
    count: int


def reliability_curve(
    p: np.ndarray | Iterable[float],
    y: np.ndarray | Iterable[int],
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Compute a simple reliability curve for binary outcomes y in {0,1}."""
    p = _as_1d(p)
    y = _as_1d(y).astype(int)
    if len(p) != len(y):
        raise ValueError("p and y must have same length")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        bins.append(
            CalibrationBin(
                p_mean=float(np.mean(p[mask])),
                frac_pos=float(np.mean(y[mask])),
                count=int(np.sum(mask)),
            )
        )
    return bins
