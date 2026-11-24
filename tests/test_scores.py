"""Scores agree with closed-form results."""

import numpy as np
from scipy import stats

from powercast import scores


def test_crps_of_point_forecast_is_absolute_error():
    y = np.array([10.0, -3.0, 50.0])
    q = np.repeat(np.array([[12.0], [-3.0], [40.0]]), 99, axis=1)
    np.testing.assert_allclose(scores.crps(y, q), np.abs(y - q[:, 0]), rtol=1e-12)


def test_crps_matches_normal_closed_form():
    # CRPS of N(mu, s) at y: s * (z(2Φ(z)−1) + 2φ(z) − 1/√π), z = (y−mu)/s
    mu, s = 0.0, 1.0
    y = np.linspace(-3, 3, 13)
    q = np.tile(stats.norm.ppf(scores.TAUS, mu, s), (len(y), 1))
    z = (y - mu) / s
    exact = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    approx = scores.crps(y, q)
    # 99 levels miss the extreme tails: agreement to about 2 %
    np.testing.assert_allclose(approx, exact, rtol=0.03)


def test_coverage_and_pit_of_calibrated_forecast():
    rng = np.random.default_rng(0)
    n = 200_000
    y = rng.normal(size=n)
    q = np.tile(stats.norm.ppf(scores.TAUS), (n, 1))
    for c in (50, 80, 90, 98):
        assert abs(scores.inside(y, q, c).mean() - c / 100) < 0.005
    p = scores.pit(y, q, rng)
    hist, _ = np.histogram(p, bins=20, range=(0, 1))
    assert np.all(np.abs(hist / n - 0.05) < 0.003)


def test_interpolate_levels_is_exact_for_linear_quantiles():
    src = np.array([0.1, 0.5, 0.9])
    q = np.array([[1.0, 5.0, 9.0]])
    out = scores.interpolate_levels(q, src)
    i = np.searchsorted(scores.TAUS, 0.3)
    assert abs(out[0, i] - 3.0) < 1e-12
    assert out[0, 0] == 1.0 and out[0, -1] == 9.0  # flat beyond the source range
