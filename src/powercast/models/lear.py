"""M1: LEAR point forecasts combined by quantile regression averaging (QRA).

LEAR (Lago et al. 2021, epftoolbox): for each local-hour slot h, a LASSO-regularised
linear regression of the transformed price on the transformed daily feature vector
(powercast.features). The regularisation strength is chosen by LARS with the Akaike
information criterion, then the LASSO is refitted at that strength. Fitted separately
for several calibration windows (number of past days used for training).

QRA (Nowotarski & Weron 2015): for each slot h and level τ, a linear quantile regression
of the price on the LEAR point forecasts of the different windows, fitted on the last N
days for which both the LEAR forecasts and the real prices are known.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC
from statsmodels.regression.quantile_regression import QuantReg

from powercast.scores import TAUS
from powercast.transforms import Asinh

WINDOWS = (56, 84, 364, 728)


def lear_day(X: pd.DataFrame, Y: pd.DataFrame, day: pd.Timestamp, window: int) -> np.ndarray:
    """24 point forecasts for ``day`` from a LEAR fitted on the ``window`` days before it."""
    train = pd.date_range(day - pd.Timedelta(days=window), day - pd.Timedelta(days=1))
    Xt, Yt = X.reindex(train), Y.reindex(train)
    ok = Xt.notna().all(axis=1) & Yt.notna().all(axis=1)
    Xt, Yt = Xt[ok].to_numpy(float), Yt[ok].to_numpy(float)
    x0 = X.loc[[day]].to_numpy(float)
    dummies = np.array([c.startswith("dow_") for c in X.columns])
    tx = Asinh().fit(Xt[:, ~dummies])
    Zt = np.hstack([tx.transform(Xt[:, ~dummies]), Xt[:, dummies]])
    z0 = np.hstack([tx.transform(x0[:, ~dummies]), x0[:, dummies]])
    out = np.empty(24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for h in range(24):
            ty = Asinh().fit(Yt[:, h])
            yt = ty.transform(Yt[:, h])
            # noise variance = Var(y), as scikit-learn < 1.1 did inside the epftoolbox
            # reference; also defined when samples < features (56- and 84-day windows)
            ic = LassoLarsIC(criterion="aic", max_iter=2500, noise_variance=max(yt.var(), 1e-6))
            alpha = ic.fit(Zt, yt).alpha_
            m = Lasso(alpha=alpha, max_iter=2500).fit(Zt, yt)
            out[h] = ty.inverse(m.predict(z0))[0]
    return out


def qra_day(points: np.ndarray, Y: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Quantiles (24, 99) for one day.

    points: (n_days, 24, n_windows) past LEAR forecasts; Y: (n_days, 24) real prices;
    x0: (24, n_windows) LEAR forecasts for the target day.
    """
    out = np.empty((24, len(TAUS)))
    for h in range(24):
        ok = np.isfinite(Y[:, h]) & np.isfinite(points[:, h]).all(axis=1)
        A = np.column_stack([np.ones(ok.sum()), points[ok, h]])
        a0 = np.concatenate([[1.0], x0[h]])
        model = QuantReg(Y[ok, h], A)
        for j, tau in enumerate(TAUS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beta = model.fit(q=tau, max_iter=5000).params
            out[h, j] = a0 @ beta
    return np.sort(out, axis=1)
