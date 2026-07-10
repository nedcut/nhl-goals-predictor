"""
Significance testing for model comparison.

Model selection on tiny fold counts (here, 5) is dangerous: a 0.05% gap in a
mean-of-5-folds is almost always sampling noise. This module compares two models
on a *paired* per-game basis and quantifies whether the observed difference is
distinguishable from zero.

The estimator is a paired bootstrap over games rather than a t-test over folds:
- It makes no normality assumption (scores are skewed and bounded).
- It resamples hundreds/thousands of games, not 5 folds, so it has real power.
- It is dependency-free (NumPy only; no SciPy).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing model A against model B on paired per-game scores.

    Scores are "lower is better" (e.g. CRPS, weighted score). `mean_diff` is
    mean(A) - mean(B): negative means A is better than B.
    """

    n_games: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    better_model: str | None
    verdict: str
    n_blocks: int | None = None
    resampling_unit: str = "game"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Iterable[object] | None = None,
) -> PairedComparison:
    """Paired bootstrap of the difference in mean per-game score (lower is better).

    Parameters
    ----------
    scores_a, scores_b
        Per-game scores for the two models, aligned game-for-game.
    n_boot
        Number of bootstrap resamples of the paired differences.
    alpha
        Two-sided significance level (0.05 => 95% CI).

    Notes
    -----
    The two-sided p-value is the bootstrap-symmetry estimate
    ``2 * min(P(diff* >= 0), P(diff* <= 0))`` over resampled mean differences,
    which avoids assuming a parametric sampling distribution.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores must be aligned and same length, got {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError("scores must be 1D")
    n = a.shape[0]
    if n < 2:
        raise ValueError("need at least 2 paired observations")

    diff = a - b  # per-game difference; negative => A better
    mean_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    n_blocks: int | None = None
    resampling_unit = "game"
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = diff[idx].mean(axis=1)
    else:
        group_values = np.asarray(list(groups), dtype=object)
        if group_values.shape != a.shape:
            raise ValueError(
                f"groups must align with scores, got {group_values.shape} vs {a.shape}"
            )
        unique_groups, inverse = np.unique(group_values.astype(str), return_inverse=True)
        n_blocks = int(len(unique_groups))
        if n_blocks < 2:
            raise ValueError("need at least 2 distinct bootstrap groups")

        # Resample whole temporal blocks, preserving within-week dependence. Use
        # block sums/counts so unequal week sizes retain the correct game weight.
        block_sums = np.bincount(inverse, weights=diff, minlength=n_blocks)
        block_counts = np.bincount(inverse, minlength=n_blocks).astype(float)
        block_idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
        boot_means = block_sums[block_idx].sum(axis=1) / block_counts[block_idx].sum(axis=1)
        resampling_unit = "block"

    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    frac_ge = float(np.mean(boot_means >= 0.0))
    frac_le = float(np.mean(boot_means <= 0.0))
    p_value = float(min(1.0, 2.0 * min(frac_ge, frac_le)))

    significant = p_value < alpha
    better_model: str | None = None
    if significant:
        better_model = name_a if mean_diff < 0 else name_b

    if not significant:
        verdict = (
            f"{name_a} and {name_b} are statistically indistinguishable "
            f"(mean diff {mean_diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], p={p_value:.3f}). "
            f"Prefer the simpler/cheaper model."
        )
    else:
        verdict = (
            f"{better_model} is better with high confidence "
            f"(mean diff {mean_diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], p={p_value:.3f})."
        )

    return PairedComparison(
        n_games=n,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        mean_diff=mean_diff,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        significant=significant,
        better_model=better_model,
        verdict=verdict,
        n_blocks=n_blocks,
        resampling_unit=resampling_unit,
    )


def holm_adjusted_p_values(p_values: Iterable[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return []
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and in [0, 1]")

    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=float)
    running = 0.0
    m = values.size
    for rank, original_index in enumerate(order):
        adjusted_value = min(1.0, float(values[original_index]) * (m - rank))
        running = max(running, adjusted_value)
        adjusted_sorted[rank] = running

    adjusted_values = np.empty(values.size, dtype=float)
    for rank, original_index in enumerate(order):
        adjusted_values[original_index] = adjusted_sorted[rank]
    return adjusted_values.tolist()
