"""Error bars and significance tests (docs/EVALUATION_PLAN.md, section 6)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def block_bootstrap_indices(n: int, block: int, n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: (n_boot, n) row indices built from random blocks of days."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def bootstrap_mean_ci(x: np.ndarray, block: int = 14, n_boot: int = 2000, level: float = 0.95,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean of daily values with a block-bootstrap percentile interval."""
    x = np.asarray(x, float)
    idx = block_bootstrap_indices(len(x), block, n_boot, np.random.default_rng(seed))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def newey_west_var(d: np.ndarray, lags: int) -> float:
    """Long-run variance of the mean of d (Bartlett kernel)."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    v = d @ d / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return v / n


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 7) -> tuple[float, float]:
    """One-sided DM test on daily losses. H1: model A has lower expected loss than B.

    Returns (statistic, p-value); the statistic is negative when A is better.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    ok = np.isfinite(d)
    d = d[ok]
    stat = d.mean() / np.sqrt(newey_west_var(d, lags))
    return float(stat), float(stats.norm.cdf(stat))


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
