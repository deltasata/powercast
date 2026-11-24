"""Daily feature matrix shared by LEAR, LightGBM and the neural network.

One row per delivery day D; columns (24 local-hour slots each):

* ``p{k}_{h}``     price of day D−k, k in PRICE_LAGS
* ``load{k}_{h}``  day-ahead load forecast for day D−k, k in exog_lags
* ``res{k}_{h}``   day-ahead wind (onshore + offshore) + solar forecast for day D−k
* ``dow_{i}``      day-of-week dummies of D (Monday = 0)

Lags are relative to the delivery day. Every lag must respect the publication rules in
powercast.nopeek; the automated check verifies it for each model run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from powercast import dataset

PRICE_LAGS = (1, 2, 3, 7)
EXOG_SOURCES = {"load": ["load_forecast"],
                "res": ["wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"]}


def first_allowed_lag(columns: list[str], probe_day: str = "2024-06-12") -> int:
    """Smallest k such that the forecasts for day D−k are public at the issue time of D.

    Derived from the publication rules in powercast.nopeek, so the features follow the
    measured publication times rather than a hand-written choice.
    """
    from powercast import nopeek

    day = pd.Timestamp(probe_day)
    for k in range(3):
        target = (day - pd.Timedelta(days=k)).tz_localize(nopeek.MARKET_TZ)
        idx = pd.date_range(target, periods=96, freq="15min").tz_convert("UTC")
        if all((nopeek.available_at(c, idx) <= nopeek.issue_time(day)).all() for c in columns):
            return k
    raise ValueError(f"{columns}: not published within 2 days")


def default_exog_lags() -> dict[str, tuple[int, int]]:
    """The two most recent publicly available days of each exogenous forecast."""
    out = {}
    for name, cols in EXOG_SOURCES.items():
        k = first_allowed_lag(cols)
        out[name] = (k, k + 1)
    return out


def exog_slots(fund: pd.DataFrame) -> dict[str, pd.DataFrame]:
    load = dataset.to_slots(fund["load_forecast"])
    res = dataset.to_slots(fund["wind_onshore_forecast"] + fund["wind_offshore_forecast"]
                           + fund["solar_forecast"])
    return {"load": dataset.fill_from_last_week(load), "res": dataset.fill_from_last_week(res)}


def build(tables: dict[str, pd.DataFrame], days: pd.DatetimeIndex,
          exog_lags: dict[str, tuple[int, ...]]) -> pd.DataFrame:
    price = dataset.fill_from_last_week(dataset.to_slots(tables["prices"]["price"]))
    ex = exog_slots(tables["fund"])
    blocks = []
    for k in PRICE_LAGS:
        blocks.append(_lagged(price, days, k, f"p{k}"))
    for name, lags in exog_lags.items():
        for k in lags:
            blocks.append(_lagged(ex[name], days, k, f"{name}{k}"))
    dow = pd.DataFrame(np.eye(7)[days.dayofweek], index=days,
                       columns=[f"dow_{i}" for i in range(7)])
    blocks.append(dow)
    return pd.concat(blocks, axis=1)


def _lagged(slots: pd.DataFrame, days: pd.DatetimeIndex, k: int, prefix: str) -> pd.DataFrame:
    v = slots.reindex(days - pd.Timedelta(days=k)).to_numpy()
    return pd.DataFrame(v, index=days, columns=[f"{prefix}_{h}" for h in range(24)])
