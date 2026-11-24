"""Download Energy-Charts data (Fraunhofer ISE, CC BY 4.0) to cross-check SMARD.

Usage:
    python scripts/download_energycharts.py [--date YYYY-MM-DD] [--pause 10]

Saves one unmodified JSON per (endpoint, parameters, calendar year) under
data/raw/energy-charts/<date>/. The API answers HTTP 429 to fast bursts, so calls
are spaced by --pause seconds and retried with back-off.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import requests

from powercast.config import MARKET_TZ, RAW_DIR

API = "https://api.energy-charts.info"


def fetch(session, endpoint, params, dest, pause):
    if dest.exists():
        return "kept"
    for k in range(6):
        r = session.get(f"{API}/{endpoint}", params=params, timeout=300)
        if r.status_code == 429:
            time.sleep(pause * 2 ** (k + 1))
            continue
        if r.status_code == 404:  # no data for this period
            return "404"
        r.raise_for_status()
        json.loads(r.content)
        dest.write_bytes(r.content)
        time.sleep(pause)
        return f"{len(r.content) / 1e6:.1f} MB"
    raise RuntimeError(f"rate-limited too often: {endpoint} {params}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=dt.datetime.now(MARKET_TZ).date().isoformat())
    p.add_argument("--pause", type=float, default=10)
    a = p.parse_args()
    root = RAW_DIR / "energy-charts" / a.date
    root.mkdir(parents=True, exist_ok=True)
    this_year = dt.datetime.now(MARKET_TZ).date().year

    jobs = []
    for y in range(2015, this_year + 1):
        start, end = f"{y}-01-01", f"{y + 1}-01-01"
        if y <= 2018:
            jobs.append(("price", {"bzn": "DE-AT-LU", "start": start, "end": end},
                         f"price_DE-AT-LU_{y}.json"))
        if y >= 2018:
            jobs.append(("price", {"bzn": "DE-LU", "start": start, "end": end},
                         f"price_DE-LU_{y}.json"))
        jobs.append(("public_power", {"country": "de", "start": start, "end": end},
                     f"public_power_de_{y}.json"))
        for pt in ["load", "solar", "wind_onshore", "wind_offshore"]:
            jobs.append(("public_power_forecast",
                         {"country": "de", "production_type": pt,
                          "forecast_type": "day-ahead", "start": start, "end": end},
                         f"forecast_day-ahead_{pt}_de_{y}.json"))

    with requests.Session() as s:
        for endpoint, params, name in jobs:
            print(name, fetch(s, endpoint, params, root / name, a.pause), flush=True)
    (root / "manifest.json").write_text(json.dumps({
        "source": "Energy-Charts.info, Fraunhofer ISE (CC BY 4.0)", "api": API,
        "snapshot_date": a.date,
        "files": {f.name: f.stat().st_size for f in sorted(root.glob("*_*.json"))},
    }, indent=2))


if __name__ == "__main__":
    main()
