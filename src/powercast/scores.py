"""Scores for quantile forecasts (docs/EVALUATION_PLAN.md, section 5).

Arrays: ``q`` has shape (n, K) with quantiles at levels ``taus`` (K,), sorted along
axis 1; ``y`` has shape (n,).
"""

from __future__ import annotations

import numpy as np

TAUS = np.round(np.arange(1, 100) / 100, 2)  # 0.01 ... 0.99
INTERVALS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 98: (0.01, 0.99)}


def pinball(y: np.ndarray, q: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Pinball loss for every (MTU, level): shape (n, K)."""
    y = np.asarray(y, float)[:, None]
    q = np.asarray(q, float)
    return ((y < q).astype(float) - taus[None, :]) * (q - y)


def crps(y: np.ndarray, q: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """CRPS per MTU, approximated as 2 × mean pinball loss over the levels."""
    return 2.0 * pinball(y, q, taus).mean(axis=1)


def _level_index(taus: np.ndarray, tau: float) -> int:
    i = int(np.argmin(np.abs(taus - tau)))
    if abs(taus[i] - tau) > 1e-9:
        raise ValueError(f"level {tau} not in taus")
    return i


def median(q: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    return np.asarray(q)[:, _level_index(taus, 0.5)]


def inside(y: np.ndarray, q: np.ndarray, coverage: int, taus: np.ndarray = TAUS):
    """Boolean per MTU: is y inside the central ``coverage`` % interval?"""
    lo, hi = INTERVALS[coverage]
    q = np.asarray(q)
    return (y >= q[:, _level_index(taus, lo)]) & (y <= q[:, _level_index(taus, hi)])


def width(q: np.ndarray, coverage: int, taus: np.ndarray = TAUS) -> np.ndarray:
    lo, hi = INTERVALS[coverage]
    q = np.asarray(q)
    return q[:, _level_index(taus, hi)] - q[:, _level_index(taus, lo)]


def pit(y: np.ndarray, q: np.ndarray, rng: np.random.Generator,
        taus: np.ndarray = TAUS) -> np.ndarray:
    """Randomised PIT from quantiles: uniform within the bin between adjacent levels.

    With K levels the bins are [0, τ_1), [τ_1, τ_2), …, [τ_K, 1].
    """
    q = np.asarray(q)
    k = (q < np.asarray(y)[:, None]).sum(axis=1)  # number of quantiles below y
    edges = np.concatenate([[0.0], taus, [1.0]])
    lo, hi = edges[k], edges[k + 1]
    return lo + (hi - lo) * rng.uniform(size=len(k))


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    """Remove quantile crossing by sorting each row."""
    return np.sort(np.asarray(q, float), axis=1)


def interpolate_levels(q_src: np.ndarray, taus_src, taus: np.ndarray = TAUS) -> np.ndarray:
    """Linear interpolation of quantiles to new levels; flat beyond the source range."""
    q_src = sort_quantiles(q_src)
    taus_src = np.asarray(taus_src, float)
    return np.vstack([np.interp(taus, taus_src, row) for row in q_src])
