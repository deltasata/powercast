"""M2: LightGBM with quantile loss.

One model per quantile level for all 24 local-hour slots: each training row is one
(day, slot) with the day's shared feature vector (powercast.features), the slot (hour of
day) and the month. The target is the asinh-transformed price; quantiles are
back-transformed exactly (quantiles are equivariant under monotone maps). Levels not
fitted directly are linearly interpolated.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from powercast.scores import TAUS, interpolate_levels
from powercast.transforms import Asinh

LEVELS = np.round(np.r_[0.01, 0.02, np.arange(0.05, 0.951, 0.05), 0.98, 0.99], 2)

DEFAULT_PARAMS = {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400,
                  "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "lambda_l2": 1.0}


def _long(X: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    x = X.reindex(days).to_numpy(np.float32)
    n = len(days)
    rep = np.repeat(x, 24, axis=0)
    slot = np.tile(np.arange(24, dtype=np.float32), n)[:, None]
    month = np.repeat(days.month.to_numpy(np.float32), 24)[:, None]
    return np.hstack([rep, slot, month])


class GBTQuantile:
    def __init__(self, params: dict | None = None, levels=LEVELS, seed: int = 0, threads: int = 8):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.levels = np.asarray(levels)
        self.seed = seed
        self.threads = threads

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, train_days: pd.DatetimeIndex) -> GBTQuantile:
        Xt = X.reindex(train_days)
        Yt = Y.reindex(train_days)
        ok = Xt.notna().all(axis=1) & Yt.notna().all(axis=1)
        days = train_days[ok.to_numpy()]
        A = _long(X, days)
        y = Y.reindex(days).to_numpy(float).reshape(-1)
        self.ty = Asinh().fit(y)
        z = self.ty.transform(y)
        self.models = []
        for tau in self.levels:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), random_state=self.seed,
                                  n_jobs=self.threads, verbose=-1, **self.params)
            m.fit(A, z)
            self.models.append(m)
        return self

    def predict(self, X: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
        """Quantiles (len(days), 24, 99)."""
        A = _long(X, days)
        z = np.column_stack([m.predict(A) for m in self.models])
        q = self.ty.inverse(np.sort(z, axis=1))
        if len(self.levels) != len(TAUS) or not np.allclose(self.levels, TAUS):
            q = interpolate_levels(q, self.levels)
        return q.reshape(len(days), 24, -1)
