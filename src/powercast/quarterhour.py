"""Common intra-hour layer for quarter-hour prices (docs/EVALUATION_PLAN.md, section 3).

Notation: P_{h,q} = price of quarter q of local-hour slot h; P̄_h = its slot mean (the
hourly mean every model forecasts); δ_{h,q} = P_{h,q} − P̄_h, the intra-hour deviation.

δ is regressed on Δr_{h,q} = r_{h,q} − r̄_h, the within-hour deviation of the quarter-hour
residual-load forecast r = load − wind onshore − wind offshore − solar, plus one dummy
per (slot, quarter). Components of r that are not public at issue time are replaced by
their value one day earlier at the same clock time (decided by powercast.nopeek rules).

The predictive distribution of P_{h,q} is the Monte Carlo sum of draws from the model's
P̄_h quantiles and draws from the regression's past residuals of the same (slot, quarter),
assuming independence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from powercast import features
from powercast.config import MARKET_TZ, QUARTER_HOUR_START_LOCAL
from powercast.scores import TAUS

MIN_DAYS = 14
N_DRAWS = 2000
QH_START_DAY = QUARTER_HOUR_START_LOCAL.tz_localize(None).normalize()


def residual_load_forecast(fund: pd.DataFrame) -> pd.Series:
    """r(t) in MW at 15 minutes; late components shifted by whole days (D−k)."""
    r = fund["load_forecast"].copy() * 0
    signs = {"load_forecast": 1, "wind_onshore_forecast": -1, "wind_offshore_forecast": -1,
             "solar_forecast": -1}
    for col, sign in signs.items():
        k = features.first_allowed_lag([col])
        s = fund[col]
        if k:
            loc = s.index.tz_convert(MARKET_TZ)
            shifted = pd.Series(s.to_numpy(), index=(loc.tz_localize(None) + pd.Timedelta(days=k)))
            # same Berlin clock time k days later; ambiguous / missing DST times dropped
            shifted = shifted[~shifted.index.duplicated()]
            s = shifted.reindex(loc.tz_localize(None)).set_axis(s.index)
        r = r + sign * s
    return r


def _frame(prices: pd.DataFrame, rload: pd.Series) -> pd.DataFrame:
    """Quarter-hour rows from 1 Oct 2025 with day, slot, quarter, δ and Δr."""
    p = prices.loc[prices["mtu_min"] == 15, "price"]
    loc = p.index.tz_convert(MARKET_TZ)
    df = pd.DataFrame({"y": p.to_numpy(), "day": loc.normalize().tz_localize(None),
                       "slot": loc.hour, "quarter": loc.minute // 15,
                       "r": rload.reindex(p.index).to_numpy()}, index=p.index)
    g = df.groupby(["day", "slot"])
    df["ybar"] = g["y"].transform("mean")
    df["delta"] = df["y"] - df["ybar"]
    df["dr"] = df["r"] - g["r"].transform("mean")
    df["cell"] = df["slot"] * 4 + df["quarter"]
    return df


def _design(df: pd.DataFrame) -> np.ndarray:
    dummies = np.zeros((len(df), 96))
    dummies[np.arange(len(df)), df["cell"].to_numpy()] = 1.0
    return np.column_stack([df["dr"].fillna(0.0).to_numpy() / 1000.0, dummies])  # Δr in GW


class QuarterHourLayer:
    def __init__(self, prices: pd.DataFrame, fund: pd.DataFrame):
        self.df = _frame(prices, residual_load_forecast(fund))

    def apply(self, day: pd.Timestamp, mtus: pd.DataFrame, q_hour: np.ndarray,
              seed: int = 0) -> np.ndarray:
        """Quarter-hour quantiles (n, 99) for the quarter-hour ``mtus`` of ``day``.

        ``q_hour`` are the slot-mean quantiles already mapped to these MTUs.
        """
        past = self.df[(self.df["day"] >= QH_START_DAY) & (self.df["day"] < day)]
        past = past.dropna(subset=["delta"])
        if past["day"].nunique() < MIN_DAYS:
            return q_hour
        A = _design(past)
        beta, *_ = np.linalg.lstsq(A, past["delta"].to_numpy(), rcond=None)
        resid = past["delta"].to_numpy() - A @ beta
        cells = past["cell"].to_numpy()
        target = self.df.loc[mtus.index]
        shift = _design(target) @ beta
        rng = np.random.default_rng(seed)
        out = np.empty_like(q_hour)
        u = rng.uniform(0.005, 0.995, size=N_DRAWS)
        for i, c in enumerate(target["cell"].to_numpy()):
            pool = resid[cells == c]
            draws_h = np.interp(u, TAUS, q_hour[i])
            draws_d = rng.choice(pool, size=N_DRAWS) if len(pool) else 0.0
            out[i] = np.quantile(draws_h + shift[i] + draws_d, TAUS)
        return out
