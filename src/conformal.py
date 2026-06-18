"""
Conformal prediction utilities (distribution-free uncertainty).

This module implements simple split-conformal prediction intervals for
regression targets like total goals.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _as_1d(x: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}")
    return arr


def split_conformal_interval(
    y_cal: np.ndarray | Iterable[float],
    yhat_cal: np.ndarray | Iterable[float],
    yhat: np.ndarray | Iterable[float],
    *,
    alpha: float = 0.1,
    clip_lower: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute split-conformal (1-alpha) prediction interval.

    Nonconformity score: |y - yhat|.

    Returns
    -------
    lo, hi, q
        Arrays of lower/upper bounds for yhat, and the fitted quantile radius q.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    y_cal = _as_1d(y_cal)
    yhat_cal = _as_1d(yhat_cal)
    yhat = _as_1d(yhat)
    if len(y_cal) != len(yhat_cal):
        raise ValueError("y_cal and yhat_cal must have same length")

    scores = np.abs(y_cal - yhat_cal)
    # Conservative quantile for finite-sample coverage.
    q_level = 1.0 - alpha
    try:
        q = float(np.quantile(scores, q_level, method="higher"))
    except TypeError:  # numpy<1.22
        # Legacy keyword removed in modern numpy; current stubs don't type it.
        q = float(np.quantile(scores, q_level, interpolation="higher"))  # type: ignore[call-overload]

    lo = yhat - q
    hi = yhat + q
    if clip_lower is not None:
        lo = np.maximum(lo, clip_lower)
    return lo, hi, q
