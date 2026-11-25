"""Build clean UTC tables from a raw SMARD snapshot.

Outputs (data/processed/<snapshot>/):

* ``prices.parquet``: one row per day-ahead market time unit (MTU), index = MTU start
  in UTC, columns ``price`` (EUR/MWh), ``mtu_min`` (60 or 15) and ``zone``.
  - 251  (DE-AT-LU, hourly)       for delivery up to 2018-09-30 (Berlin);
  - 4169 (DE-LU, hourly file)     for delivery 2018-10-01 to 2025-09-30;
  - 4169 (DE-LU, quarter-hour)    for delivery from 2025-10-01.
  SMARD's quarter-hour price file repeats the hourly price four times before
  2025-10-01, so it is not used there (see docs/LOG.md, step 2).
* ``fundamentals.parquet``: 15-minute grid, index = interval start in UTC, columns in
  MW. SMARD reports energy per interval (MWh per 15 min), so values are multiplied by 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from powercast.config import MARKET_TZ, QUARTER_HOUR_START_LOCAL
from powercast.data import smard

ZONE_SPLIT_LOCAL = pd.Timestamp("2018-10-01 00:00", tz=MARKET_TZ)

FUNDAMENTALS = {
    "load_actual": "410_DE_quarterhour",
    "load_forecast": "411_DE_quarterhour",
    "wind_onshore_actual": "4067_DE_quarterhour",
    "wind_offshore_actual": "1225_DE_quarterhour",
    "solar_actual": "4068_DE_quarterhour",
    "wind_onshore_forecast": "123_DE_quarterhour",
    "wind_offshore_forecast": "3791_DE_quarterhour",
    "solar_forecast": "125_DE_quarterhour",
}


def _dedup(s: pd.Series) -> pd.Series:
    """Sort and drop exact duplicate timestamps; raise if duplicates disagree."""
    s = s.sort_index()
    dup = s.index.duplicated(keep=False)
    if dup.any():
        g = s[dup].groupby(level=0).nunique(dropna=False)
        if (g > 1).any():
            raise ValueError(f"{s.name}: conflicting duplicate timestamps {g[g > 1].index[:5]}")
    return s[~s.index.duplicated()]


def build_prices(raw: Path) -> pd.DataFrame:
    split = ZONE_SPLIT_LOCAL.tz_convert("UTC")
    qh_start = QUARTER_HOUR_START_LOCAL.tz_convert("UTC")
    a = _dedup(smard.read(raw / "251_DE_hour"))
    b = _dedup(smard.read(raw / "4169_DE_hour"))
    c = _dedup(smard.read(raw / "4169_DE_quarterhour"))
    parts = [
        pd.DataFrame({"price": a[a.index < split], "mtu_min": 60, "zone": "DE-AT-LU"}),
        pd.DataFrame({"price": b[(b.index >= split) & (b.index < qh_start)],
                      "mtu_min": 60, "zone": "DE-LU"}),
        pd.DataFrame({"price": c[c.index >= qh_start], "mtu_min": 15, "zone": "DE-LU"}),
    ]
    df = pd.concat(parts)
    df.index.name = "start_utc"
    df["source"] = "smard"
    # trailing NaN rows are future intervals in the last weekly file, not gaps
    last = df["price"].last_valid_index()
    return df.loc[:last]


def fill_prices_from_ec(prices: pd.DataFrame, ec_price: pd.Series) -> pd.DataFrame:
    """Fill missing SMARD prices with Energy-Charts values at the same timestamps.

    Justified by the cross-check: the two sources agree exactly on all common prices
    (results/data_checks/crosscheck_<snapshot>.json). Filled rows get source="energy-charts".
    """
    out = prices.copy()
    miss = out["price"].isna() & out.index.isin(ec_price.dropna().index)
    out.loc[miss, "price"] = ec_price.reindex(out.index[miss]).to_numpy()
    out.loc[miss, "source"] = "energy-charts"
    return out


def build_fundamentals(raw: Path) -> pd.DataFrame:
    cols = {}
    for name, key in FUNDAMENTALS.items():
        s = _dedup(smard.read(raw / key))
        cols[name] = s * 4.0  # MWh per 15 min -> MW
    df = pd.DataFrame(cols)
    df.index.name = "start_utc"
    last = df.apply(pd.Series.last_valid_index).max()
    return df.loc[:last]


def delivery_day(index: pd.DatetimeIndex) -> pd.Index:
    """Berlin calendar day each interval is delivered on."""
    return index.tz_convert(MARKET_TZ).normalize().tz_localize(None)

def extend_from_ec(prices: pd.DataFrame, fund: pd.DataFrame,
                   ec: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Daily run: complete the newest SMARD values with Energy-Charts.

    SMARD posts the day-ahead price and the wind/solar forecasts hours later than
    Energy-Charts (LOG step 5, probe of 23 Nov 2026). Prices: missing values are filled
    and values after SMARD's last one appended (the sources agree exactly). Forecasts of
    load, wind and solar: missing values filled and appended (mean absolute differences
    0.15–36 MW, far below forecast errors). Measured values are not touched.
    """
    from powercast.config import QUARTER_HOUR_START_LOCAL

    n = {}
    if "price" in ec:
        p = fill_prices_from_ec(prices, ec["price"])
        extra = ec["price"].dropna()
        extra = extra[extra.index > p.index.max()]
        if len(extra):
            qh = QUARTER_HOUR_START_LOCAL.tz_convert("UTC")
            add = pd.DataFrame({"price": extra.to_numpy(),
                                "mtu_min": np.where(extra.index >= qh, 15, 60),
                                "zone": "DE-LU", "source": "energy-charts"}, index=extra.index)
            add = add.astype({c: prices[c].dtype for c in add.columns if c in prices.columns})
            p = pd.concat([p, add[prices.columns]])
        n["price"] = int((p["source"] == "energy-charts").sum()
                         - (prices["source"] == "energy-charts").sum())
        prices = p
    f = fund.copy()
    for col in ("load_forecast", "wind_onshore_forecast", "wind_offshore_forecast",
                "solar_forecast"):
        if col in ec:
            before = f[col].notna().sum() if col in f else 0
            f = f.combine_first(ec[col].rename(col).to_frame())
            n[col] = int(f[col].notna().sum() - before)
    return prices, f.sort_index(), n

