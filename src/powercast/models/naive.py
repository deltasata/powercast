"""M0: same local hour one week earlier, plus the empirical spread of its past errors.

For delivery day D and slot h:
    point(D, h) = P(D−7, h)
    e(d, h)     = P(d, h) − P(d−7, h)                       (error of the rule on day d)
    q_τ(D, h)   = point(D, h) + Quantile_τ{ e(d, h) : D−W ≤ d ≤ D−1 }

All inputs are prices of days up to D−1, published by D−2 13:00, before the issue time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from powercast.scores import TAUS


def forecast(price_slots: pd.DataFrame, days: pd.DatetimeIndex, window: int) -> np.ndarray:
    """Quantiles (len(days), 24, 99) for the delivery ``days``."""
    full = price_slots.reindex(pd.date_range(price_slots.index.min(),
                                             days.max(), freq="D"))
    p = full.to_numpy(float)
    err = p - np.vstack([np.full((7, 24), np.nan), p[:-7]])
    pos = full.index.get_indexer(days)
    if (pos < window + 7).any():
        raise ValueError("not enough history before the first day")
    # windows[i] covers error rows i .. i+window-1; for day at row r use rows r-window .. r-1
    win = sliding_window_view(err, window, axis=0)  # (n-window+1, 24, window)
    e = win[pos - window]
    qe = np.nanquantile(e, TAUS, axis=2)  # (99, ndays, 24)
    point = p[pos - 7]
    return point[:, :, None] + np.moveaxis(qe, 0, 2)
