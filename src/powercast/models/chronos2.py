"""M4: Chronos-2 (Amazon, Apache-2.0) used zero-shot, without any training.

For delivery day D the context is the hourly slot series of prices up to the end of D−1
(at most the model's 8192-step context), and the model forecasts the next 24 slots.

Variants:
* ``univariate``: prices only.
* ``covariates``: load forecast as a known-future covariate (public for D at issue time)
  and wind + solar forecast as a past-only covariate (its value for D is not public).

The 21 native quantile levels (0.01, 0.05, …, 0.95, 0.99) are interpolated linearly to the
99 evaluation levels.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from powercast.config import ROOT
from powercast.scores import interpolate_levels

# model weights live inside the project (never in the home directory)
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

NATIVE = [0.01] + [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]
CONTEXT = 8192
# weights identical to the first release of 30 Oct 2025 (D10.2)
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def _series(slots: pd.DataFrame) -> pd.Series:
    """Day × 24 slot table → one hourly series indexed by (day, slot) order."""
    s = slots.stack(future_stack=True)
    return s


class Chronos2:
    def __init__(self, variant: str = "covariates", device: str = "cpu",
                 model: str = "amazon/chronos-2", batch: int = 32):
        from chronos import BaseChronosPipeline

        self.variant = variant
        self.pipe = BaseChronosPipeline.from_pretrained(model, device_map=device,
                                                        revision=REVISION)
        self.batch = batch

    def forecast(self, price: pd.DataFrame, exog: dict[str, pd.DataFrame],
                 days: pd.DatetimeIndex) -> np.ndarray:
        """Quantiles (len(days), 24, 99). ``price``/``exog`` are day × 24 slot tables."""
        p = _series(price)
        load = _series(exog["load"])
        res = _series(exog["res"])
        inputs = []
        for d in days:
            end = d - pd.Timedelta(days=1)
            hist = p.loc[:end].iloc[-CONTEXT:]
            item = {"target": hist.to_numpy(np.float32)}
            if self.variant == "covariates":
                idx = hist.index
                item["past_covariates"] = {
                    "load": load.reindex(idx).to_numpy(np.float32),
                    "res": res.reindex(idx).to_numpy(np.float32),
                }
                item["future_covariates"] = {
                    "load": load.loc[d].to_numpy(np.float32),
                }
            inputs.append(item)
        out = []
        for k in range(0, len(inputs), self.batch):
            q, _ = self.pipe.predict_quantiles(inputs[k:k + self.batch], prediction_length=24,
                                               quantile_levels=NATIVE)
            out += [t[0].cpu().numpy() for t in q]  # (24, 21) each
        q = np.stack(out)  # (days, 24, 21)
        return interpolate_levels(q.reshape(-1, len(NATIVE)), NATIVE).reshape(len(days), 24, -1)
