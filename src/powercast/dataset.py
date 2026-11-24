"""Load processed tables and reshape them to (delivery day × 24 local-hour slots).

Daylight-saving convention (docs/EVALUATION_PLAN.md, section 3): on 25-hour days the two
02:00 hours are averaged; on 23-hour days slot 2 is the mean of slots 1 and 3.
"""

from __future__ import annotations

import pandas as pd

from powercast.config import MARKET_TZ, PROCESSED_DIR

FORECAST_COLUMNS = ["load_forecast", "wind_onshore_forecast", "wind_offshore_forecast",
                    "solar_forecast"]


def load(snapshot: str) -> dict[str, pd.DataFrame]:
    d = PROCESSED_DIR / snapshot
    return {"prices": pd.read_parquet(d / "prices.parquet"),
            "fund": pd.read_parquet(d / "fundamentals.parquet")}


def to_slots(s: pd.Series) -> pd.DataFrame:
    """Hourly means by (Berlin delivery day, local hour 0–23)."""
    loc = s.index.tz_convert(MARKET_TZ)
    day = loc.normalize().tz_localize(None)
    g = s.groupby([day, loc.hour]).mean().unstack()
    g = g.reindex(columns=range(24))
    g.index.name = "day"
    gap = g[2].isna() & g[1].notna() & g[3].notna()  # 23-hour days
    g.loc[gap, 2] = (g.loc[gap, 1] + g.loc[gap, 3]) / 2
    return g


def fill_from_last_week(slots: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values by the value one week earlier (known at issue time)."""
    full = slots.reindex(pd.date_range(slots.index.min(), slots.index.max(), freq="D"))
    for _ in range(4):  # up to four consecutive missing weeks
        full = full.fillna(full.shift(7))
    return full
