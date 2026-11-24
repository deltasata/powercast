"""Distributional-network likelihoods and quantiles agree with scipy.stats."""

import numpy as np
import torch
from scipy import stats

from powercast.models import ddnn


def _inv_softplus(v):
    return np.log(np.expm1(v - 1e-3))


def test_jsu_matches_scipy():
    g, d, xi, lam = 0.7, 1.6, 0.3, 1.2
    out = np.array([[[g, _inv_softplus(d), xi, _inv_softplus(lam)]] * 24])
    y = np.linspace(-3, 3, 24)[None, :]
    mine = float(ddnn.nll(torch.tensor(out), torch.tensor(y), "jsu"))
    ref = -stats.johnsonsu.logpdf(y, g, d, loc=xi, scale=lam).mean()
    assert abs(mine - ref) < 1e-10
    q = ddnn.quantiles(out, "jsu")[0, 0]
    np.testing.assert_allclose(q, stats.johnsonsu.ppf(np.arange(1, 100) / 100, g, d,
                                                      loc=xi, scale=lam), rtol=1e-10)


def test_normal_matches_scipy():
    out = np.array([[[0.5, _inv_softplus(2.0)]] * 24])
    y = np.linspace(-3, 3, 24)[None, :]
    mine = float(ddnn.nll(torch.tensor(out), torch.tensor(y), "normal"))
    assert abs(mine + stats.norm.logpdf(y, 0.5, 2.0).mean()) < 1e-10
    np.testing.assert_allclose(ddnn.quantiles(out, "normal")[0, 0],
                               stats.norm.ppf(np.arange(1, 100) / 100, 0.5, 2.0), rtol=1e-10)
