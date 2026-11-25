"""Download and parse SMARD chart data (Bundesnetzagentur | SMARD.de, CC BY 4.0).

SMARD serves each series as weekly JSON files:

    https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json
    https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json

where ``ts`` is the start of the week in milliseconds since the Unix epoch and each
file holds ``[[interval_start_ms, value], ...]``. Values are in MW (power) or EUR/MWh
(price); each timestamp is the *start* of its interval.

Raw files are saved unmodified under ``data/raw/smard/<download date>/``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"


@dataclass(frozen=True)
class Series:
    name: str
    filter_id: int
    region: str
    resolution: str  # "hour" or "quarterhour"
    unit: str

    @property
    def key(self) -> str:
        return f"{self.filter_id}_{self.region}_{self.resolution}"


# Germany-only region "DE" for the fundamentals: the only region covering 2015 onward
# (see docs/DECISIONS.md, D2.2). Prices: 251 = DE-AT-LU zone (to 2018-09-30),
# 4169 = DE-LU zone (from 2018-10-01); the region field is ignored by SMARD for prices.
SERIES = [
    Series("price_de_at_lu", 251, "DE", "hour", "EUR/MWh"),
    Series("price_de_lu", 4169, "DE", "hour", "EUR/MWh"),
    Series("price_de_lu", 4169, "DE", "quarterhour", "EUR/MWh"),
    Series("load_actual", 410, "DE", "quarterhour", "MW"),
    Series("load_forecast", 411, "DE", "quarterhour", "MW"),
    Series("wind_onshore_actual", 4067, "DE", "quarterhour", "MW"),
    Series("wind_offshore_actual", 1225, "DE", "quarterhour", "MW"),
    Series("solar_actual", 4068, "DE", "quarterhour", "MW"),
    Series("wind_onshore_forecast", 123, "DE", "quarterhour", "MW"),
    Series("wind_offshore_forecast", 3791, "DE", "quarterhour", "MW"),
    Series("solar_forecast", 125, "DE", "quarterhour", "MW"),
]


def _get(session: requests.Session, url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            json.loads(r.content)  # must be valid JSON before we save it
            return r.content
        except (requests.RequestException, ValueError):
            if k == tries - 1:
                raise
            time.sleep(2**k)
    raise RuntimeError("unreachable")


def download(series: Series, out_root: Path, workers: int = 4, last: int | None = None) -> Path:
    """Download the weekly files of ``series`` into ``out_root/<key>/``.

    ``last``: only the most recent ``last`` weekly files (daily update). Files already
    present are kept (the folder is one dated snapshot).
    """
    out = out_root / series.key
    out.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        idx_url = f"{BASE}/{series.filter_id}/{series.region}/index_{series.resolution}.json"
        idx = _get(s, idx_url)
        (out / "index.json").write_bytes(idx)
        stamps = json.loads(idx)["timestamps"]
        if last:
            stamps = stamps[-last:]

        def fetch(ts: int) -> None:
            f = out / f"{ts}.json"
            if f.exists():
                return
            url = (
                f"{BASE}/{series.filter_id}/{series.region}/"
                f"{series.filter_id}_{series.region}_{series.resolution}_{ts}.json"
            )
            f.write_bytes(_get(s, url))
            time.sleep(0.1)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(fetch, stamps))
    return out


def read(folder: Path) -> pd.Series:
    """Parse a downloaded series folder into a UTC-indexed float Series.

    Missing values (JSON null) are kept as NaN; they are reported by the data checks.
    Duplicated timestamps across weekly files are kept so the checks can see them.
    """
    stamps = json.loads((folder / "index.json").read_text())["timestamps"]
    frames = []
    for ts in stamps:
        if not (folder / f"{ts}.json").exists():  # partial snapshot (daily update)
            continue
        data = json.loads((folder / f"{ts}.json").read_text())["series"]
        frames.append(pd.DataFrame(data, columns=["t", "v"]))
    df = pd.concat(frames, ignore_index=True)
    idx = pd.to_datetime(df["t"], unit="ms", utc=True)
    return pd.Series(df["v"].astype(float).to_numpy(), index=idx, name=folder.name)
