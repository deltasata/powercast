"""Data-quality checks: grid regularity, gaps, duplicates and daylight-saving days."""

from __future__ import annotations

import pandas as pd

from powercast.config import MARKET_TZ
from powercast.data.build import delivery_day


def expected_intervals(day: pd.Timestamp, mtu_min: int) -> int:
    """Number of MTUs in a Berlin delivery day (23, 24 or 25 hours long)."""
    start = pd.Timestamp(day).tz_localize(MARKET_TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(MARKET_TZ)
    return int((end - start) / pd.Timedelta(minutes=mtu_min))


def grid_report(s: pd.Series, freq: str) -> dict:
    """Missing grid points and NaN values of one series on a regular UTC grid."""
    s = s.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq=freq, tz="UTC")
    missing = full.difference(s.index)
    nan = s.index[s.isna()]
    holes = missing.union(nan)
    runs = []
    if len(holes):
        step = pd.Timedelta(freq)
        brk = (holes[1:] - holes[:-1]) != step
        starts = [holes[0], *holes[1:][brk]]
        ends = [*holes[:-1][brk], holes[-1]]
        runs = sorted(zip(starts, ends, strict=True), key=lambda r: r[0] - r[1])
    return {
        "first_utc": str(s.index.min()),
        "last_utc": str(s.index.max()),
        "n": int(s.size),
        "duplicates": int(s.index.duplicated().sum()),
        "missing_grid_points": int(len(missing)),
        "nan_values": int(len(nan)),
        "longest_gaps": [(str(a), str(b), int((b - a) / pd.Timedelta(freq)) + 1)
                         for a, b in runs[:5]],
    }


def price_day_counts(prices: pd.DataFrame) -> pd.DataFrame:
    """Per delivery day: observed vs expected number of MTUs."""
    day = delivery_day(prices.index)
    g = prices.groupby(day)
    out = pd.DataFrame({"n": g["price"].count(), "mtu_min": g["mtu_min"].first()})
    out["expected"] = [expected_intervals(d, m) for d, m in zip(out.index, out["mtu_min"],
                                                                 strict=True)]
    return out
