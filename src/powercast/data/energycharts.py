"""Parse raw Energy-Charts downloads (Energy-Charts.info, Fraunhofer ISE, CC BY 4.0)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EC_POWER = {"Load": "load_actual", "Wind onshore": "wind_onshore_actual",
            "Wind offshore": "wind_offshore_actual", "Solar": "solar_actual"}
EC_FC = {"load": "load_forecast", "wind_onshore": "wind_onshore_forecast",
         "wind_offshore": "wind_offshore_forecast", "solar": "solar_forecast"}


def _ts(seconds) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(seconds), unit="s", utc=True)


def load(root: Path) -> dict[str, pd.Series]:
    out: dict[str, list[pd.Series]] = {}
    for f in sorted(root.glob("price_*.json")):
        d = json.loads(f.read_text())
        out.setdefault("price", []).append(pd.Series(d["price"], index=_ts(d["unix_seconds"])))
    for f in sorted(root.glob("public_power_de_*.json")):
        d = json.loads(f.read_text())
        idx = _ts(d["unix_seconds"])
        for p in d["production_types"]:
            if p["name"] in EC_POWER:
                out.setdefault(EC_POWER[p["name"]], []).append(pd.Series(p["data"], index=idx))
    for f in sorted(root.glob("forecast_day-ahead_*_de_*.json")):
        pt = f.name.removeprefix("forecast_day-ahead_").rsplit("_de_", 1)[0]
        d = json.loads(f.read_text())
        out.setdefault(EC_FC[pt], []).append(
            pd.Series(d["forecast_values"], index=_ts(d["unix_seconds"])))
    res = {}
    for k, parts in out.items():
        s = pd.concat(parts).astype(float)
        res[k] = s[~s.index.duplicated()].sort_index()
    return res
