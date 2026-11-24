"""Variance-stabilising transform asinh((x − median) / MAD) (Uniejewski et al. 2018).

MAD is the median absolute deviation, scaled by 1/Φ⁻¹(3/4) ≈ 1.4826 so that it equals
the standard deviation for normal data. Median and MAD come from the training window only.
"""

from __future__ import annotations

import numpy as np

MAD_SCALE = 1.482602218505602


class Asinh:
    def fit(self, x: np.ndarray) -> Asinh:
        x = np.asarray(x, float)
        self.med = np.nanmedian(x, axis=0)
        mad = MAD_SCALE * np.nanmedian(np.abs(x - self.med), axis=0)
        self.mad = np.where(mad > 0, mad, 1.0)
        return self

    def transform(self, x):
        return np.arcsinh((np.asarray(x, float) - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(np.asarray(z, float)) * self.mad + self.med
