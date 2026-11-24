"""Bootstrap and Diebold–Mariano behave as expected on synthetic data."""

import numpy as np

from powercast import stats


def test_bootstrap_interval_covers_true_mean_for_ar1():
    # AR(1) daily losses: a naive i.i.d. bootstrap would be too narrow, blocks fix it
    rng = np.random.default_rng(3)
    hits = 0
    for rep in range(100):
        e = rng.normal(size=730)
        x = np.empty(730)
        x[0] = e[0]
        for t in range(1, 730):
            x[t] = 0.6 * x[t - 1] + e[t]
        _, lo, hi = stats.bootstrap_mean_ci(x, block=14, n_boot=500, seed=rep)
        hits += lo <= 0 <= hi
    assert hits >= 85  # nominal 95; block bootstrap slightly undercovers at n = 730


def test_dm_size_and_power():
    rng = np.random.default_rng(0)
    p_null = [stats.diebold_mariano(rng.normal(size=500), rng.normal(size=500))[1]
              for _ in range(400)]
    assert 0.02 < np.mean(np.array(p_null) < 0.05) < 0.09  # size about 5 %
    a = rng.normal(size=500) - 0.3
    b = rng.normal(size=500)
    assert stats.diebold_mariano(a, b)[1] < 0.01
    assert stats.diebold_mariano(b, a)[1] > 0.99


def test_holm():
    adj = stats.holm(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
