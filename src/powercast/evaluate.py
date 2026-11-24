"""Map slot forecasts to market time units (MTUs) and score them.

Slot forecasts have shape (days, 24, 99). Each MTU of a delivery day takes the quantiles
of its local-hour slot: on 25-hour days both 02:00 hours get slot 2, on 23-hour days
slot 2 has no MTU. Quarter-hour MTUs (from 1 Oct 2025) get their hour's quantiles here;
the intra-hour layer (powercast.quarterhour) then adjusts them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from powercast import scores
from powercast.config import MARKET_TZ


def mtu_table(prices: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Prices of the MTUs delivered on ``days``, with their day and slot."""
    loc = prices.index.tz_convert(MARKET_TZ)
    t = pd.DataFrame({"y": prices["price"].to_numpy(),
                      "mtu_min": prices["mtu_min"].to_numpy(),
                      "day": loc.normalize().tz_localize(None),
                      "slot": loc.hour}, index=prices.index)
    t = t[t["day"].isin(days) & t["y"].notna()]
    return t


# Harmonised clearing-price limits of the single day-ahead coupling (SDAC), known in
# advance: minimum -500 EUR/MWh; maximum 3000 EUR/MWh, raised to 4000 from delivery
# 10 May 2022 (docs/DECISIONS.md D10.1).
PRICE_MIN = -500.0
PRICE_MAX_STEPS = [(pd.Timestamp("2022-05-10"), 4000.0)]
PRICE_MAX_BEFORE = 3000.0


def price_max(days) -> np.ndarray:
    days = pd.DatetimeIndex(days)
    out = np.full(len(days), PRICE_MAX_BEFORE)
    for start, value in PRICE_MAX_STEPS:
        out[days >= start] = value
    return out


def clip_to_limits(q: np.ndarray, days) -> np.ndarray:
    """Clip quantiles to the auction's price limits; ``days`` gives the delivery day of
    each row of ``q`` (first axis). Infinite values go to the limit of their sign."""
    q = np.asarray(q, float)
    hi = price_max(days).reshape((-1,) + (1,) * (q.ndim - 1))
    return np.clip(q, PRICE_MIN, hi)


def to_mtu(slot_q: np.ndarray, days: pd.DatetimeIndex, mtus: pd.DataFrame) -> np.ndarray:
    """Quantiles (n_mtu, 99) for the rows of ``mtus``, clipped to the price limits."""
    di = days.get_indexer(mtus["day"])
    if (di < 0).any():
        raise ValueError("MTU on a day without forecast")
    return clip_to_limits(slot_q[di, mtus["slot"].to_numpy()], mtus["day"])


def mtu_quantiles(prices, fund, days, slot_q, layer=None):
    """MTU table and quantiles for ``days``, with the quarter-hour layer from 1 Oct 2025."""
    from powercast.config import QUARTER_HOUR_START_LOCAL
    from powercast.quarterhour import QuarterHourLayer

    qh_day = QUARTER_HOUR_START_LOCAL.tz_localize(None).normalize()
    m = mtu_table(prices, days)
    q = to_mtu(slot_q, days, m)
    qh_days = days[days >= qh_day]
    if len(qh_days):
        layer = layer or QuarterHourLayer(prices, fund)
        for d in qh_days:
            sel = (m["day"] == d).to_numpy() & (m["mtu_min"] == 15).to_numpy()
            if sel.any():
                q[sel] = layer.apply(d, m[sel], q[sel], seed=int(d.strftime("%Y%m%d")))
        q = clip_to_limits(q, m["day"])
    return m, q


def daily_scores(mtus: pd.DataFrame, q: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Per-MTU scores; aggregate with .groupby('day').mean() for daily scores."""
    q = scores.sort_quantiles(q)
    y = mtus["y"].to_numpy()
    out = pd.DataFrame({"day": mtus["day"].to_numpy(), "slot": mtus["slot"].to_numpy(),
                        "y": y}, index=mtus.index)
    out["crps"] = scores.crps(y, q)
    out["ae"] = np.abs(y - scores.median(q))
    for c in scores.INTERVALS:
        out[f"in{c}"] = scores.inside(y, q, c)
        out[f"w{c}"] = scores.width(q, c)
    out["pit"] = scores.pit(y, q, np.random.default_rng(seed))
    return out


def save_forecast(path, days: pd.DatetimeIndex, q: np.ndarray, **meta) -> None:
    """Save slot quantiles (days, 24, 99) with their delivery days."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, days=np.asarray(days.strftime("%Y-%m-%d"), dtype="U10"),
                        q=q.astype(np.float32), meta=np.array(repr(meta)))


def load_forecast(path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return pd.DatetimeIndex(pd.to_datetime(z["days"])), z["q"].astype(float)


VALIDATION_END = pd.Timestamp("2020-12-31")


def summary_if_validation(prices: pd.DataFrame, days: pd.DatetimeIndex, q: np.ndarray):
    """Scores for validation runs only: test scores are computed once, by
    scripts/evaluate_models.py, after every setting is frozen (EVALUATION_PLAN section 2)."""
    if days[-1] > VALIDATION_END:
        return "test-period days: scores withheld until the final evaluation"
    return summary(prices, days, q)


def summary(prices: pd.DataFrame, days: pd.DatetimeIndex, q: np.ndarray) -> dict:
    """Mean daily scores over ``days`` (hourly MTUs; no quarter-hour layer)."""
    m = mtu_table(prices, days)
    s = daily_scores(m, to_mtu(q, days, m)).groupby("day").mean(numeric_only=True)
    out = {"days": int(len(s)), "crps": float(s["crps"].mean()), "mae": float(s["ae"].mean())}
    for c in scores.INTERVALS:
        out[f"cov{c}"] = float(s[f"in{c}"].mean())
        out[f"width{c}"] = float(s[f"w{c}"].mean())
    return out
