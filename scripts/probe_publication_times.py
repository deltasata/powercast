"""Measure when the day-ahead forecasts for tomorrow appear on SMARD and Energy-Charts.

Polls both sources every few minutes and appends one row per (poll, series) to
data/raw/publication_probe/<target-day>/log.csv:

    poll_utc, source, series, n_target, last_interval_utc, sha1_target_values

n_target is the number of non-missing values for the target delivery day (Berlin
time). A change of sha1 between polls means the published values were revised.
Every raw response is saved next to the log.

Usage:
    python scripts/probe_publication_times.py --until 19:30 --every 300
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import requests

from powercast.config import MARKET_TZ, RAW_DIR

SMARD = "https://www.smard.de/app/chart_data"
EC = "https://api.energy-charts.info/public_power_forecast"
SMARD_SERIES = {
    "price": 4169,
    "load_fc": 411,
    "wind_onshore_fc": 123,
    "wind_offshore_fc": 3791,
    "solar_fc": 125,
}
EC_SERIES = ["load", "wind_onshore", "wind_offshore", "solar"]


def _target_values(pairs, day_start_utc, day_end_utc):
    vals = [
        (t, v) for t, v in pairs if v is not None and day_start_utc <= t < day_end_utc
    ]
    digest = hashlib.sha1(json.dumps(vals).encode()).hexdigest()[:12]
    last = dt.datetime.fromtimestamp(vals[-1][0], dt.UTC).isoformat() if vals else ""
    return len(vals), last, digest


def poll_once(session, target: dt.date, outdir: Path, writer) -> None:
    start = dt.datetime.combine(target, dt.time(0), MARKET_TZ)
    end = dt.datetime.combine(target + dt.timedelta(days=1), dt.time(0), MARKET_TZ)
    s0, s1 = int(start.timestamp()), int(end.timestamp())
    now = dt.datetime.now(dt.UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    snap = outdir / stamp
    snap.mkdir(parents=True, exist_ok=True)

    for name, fid in SMARD_SERIES.items():
        try:
            idx = session.get(f"{SMARD}/{fid}/DE/index_quarterhour.json", timeout=30).json()
            # the delivery day can straddle two weekly files; fetch the last two
            pairs = []
            for ts in idx["timestamps"][-2:]:
                r = session.get(f"{SMARD}/{fid}/DE/{fid}_DE_quarterhour_{ts}.json", timeout=30)
                (snap / f"smard_{fid}_{ts}.json").write_bytes(r.content)
                pairs += [(t // 1000, v) for t, v in r.json()["series"]]
            row = _target_values(pairs, s0, s1)
        except (requests.RequestException, ValueError, KeyError) as exc:
            row = (-1, "", f"error:{type(exc).__name__}")
        writer.writerow([now.isoformat(), "smard", name, *row])

    for pt in EC_SERIES:
        params = {
            "country": "de",
            "production_type": pt,
            "forecast_type": "day-ahead",
            "start": target.isoformat(),
            "end": (target + dt.timedelta(days=1)).isoformat(),
        }
        try:
            r = session.get(EC, params=params, timeout=30)
            if r.status_code == 404:  # Energy-Charts: "no content available" = not published
                writer.writerow([now.isoformat(), "energy-charts", f"{pt}_fc", 0, "",
                                 "http404"])
                time.sleep(5)
                continue
            r.raise_for_status()
            (snap / f"ec_{pt}.json").write_bytes(r.content)
            d = r.json()
            pairs = list(zip(d["unix_seconds"], d["forecast_values"], strict=True))
            row = _target_values(pairs, s0, s1)
        except (requests.RequestException, ValueError, KeyError) as exc:
            row = (-1, "", f"error:{type(exc).__name__}")
        writer.writerow([now.isoformat(), "energy-charts", f"{pt}_fc", *row])
        time.sleep(5)  # Energy-Charts answers HTTP 429 to fast bursts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", help="delivery day YYYY-MM-DD (default: tomorrow, Berlin)")
    p.add_argument("--until", default="19:30", help="stop at this Berlin time HH:MM")
    p.add_argument("--every", type=int, default=300, help="seconds between polls")
    a = p.parse_args()

    today = dt.datetime.now(MARKET_TZ).date()
    target = dt.date.fromisoformat(a.target) if a.target else today + dt.timedelta(days=1)
    stop = dt.datetime.combine(today, dt.time.fromisoformat(a.until), MARKET_TZ)
    outdir = RAW_DIR / "publication_probe" / target.isoformat()
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "log.csv"
    new = not log.exists()
    with requests.Session() as session, log.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["poll_utc", "source", "series", "n_target", "last_interval_utc",
                        "sha1_target_values"])
        while dt.datetime.now(MARKET_TZ) < stop:
            poll_once(session, target, outdir, w)
            fh.flush()
            time.sleep(a.every)


if __name__ == "__main__":
    main()
