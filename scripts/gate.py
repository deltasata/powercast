"""First step of the daily GitHub job: forecast now, wait and forecast, or hand over.

GitHub starts few of the scheduled triggers, at unpredictable times (D12.1, D12.3). Any
run therefore decides, from the clock and from which forecasts exist on main, what to do:

- the next delivery day without a forecast is D; its issue time T is 10:15 Berlin on D−1;
- if T has passed: forecast now (flagged late by the forecast script if after 12:00);
- if T is more than WINDOW away: stop; a later trigger, inside the window, starts the
  run (D12.4: no runner sits idle outside the window);
- if T is at most MAX_WAIT away: wait until T, then forecast;
- otherwise hand over ("relay"): sleep, then start a fresh run of the workflow by
  workflow_dispatch (the built-in token may do this), which decides again. The sleep
  leaves the fresh run RELAY_TARGET before T. At most MAX_HOPS hand-overs per chain; a
  run that makes a forecast never hands over.

Usage in the workflow (writes mode, wait_s, sleep_s and next_hop to $GITHUB_OUTPUT):
    python scripts/gate.py --repo owner/name --hop 0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
ISSUE = dt.time(10, 15)
MAX_WAIT = dt.timedelta(minutes=230)     # wait + 110 min of work + margin < 355 min
RELAY_TARGET = dt.timedelta(minutes=180)  # a fresh run starts this long before T
MAX_SLEEP = dt.timedelta(minutes=330)
MAX_HOPS = 2                              # within WINDOW one hand-over suffices
WINDOW = dt.timedelta(hours=8)            # D12.4: twice the longest gap seen (4 h 05)


def decide(now: dt.datetime, exists: Callable[[dt.date], bool], hop: int) -> dict:
    """``now`` aware; ``exists(day)`` tells whether the forecast for ``day`` is on main."""
    n = now.astimezone(BERLIN)
    for k in (0, 1):
        day = n.date() + dt.timedelta(days=1 + k)
        if exists(day):
            continue
        t = dt.datetime.combine(n.date() + dt.timedelta(days=k), ISSUE, tzinfo=BERLIN)
        w = t - n
        base = {"day": day.isoformat(), "issue": t.isoformat(), "hop": hop}
        if w <= dt.timedelta(0):
            return {**base, "mode": "forecast", "wait_s": 0}
        if w > WINDOW:
            return {**base, "mode": "stop", "reason": "outside the window"}
        if w <= MAX_WAIT:
            return {**base, "mode": "forecast", "wait_s": int(w.total_seconds())}
        if hop >= MAX_HOPS:
            return {**base, "mode": "stop", "reason": "hop limit"}
        s = min(w - RELAY_TARGET, MAX_SLEEP)
        return {**base, "mode": "relay", "sleep_s": int(s.total_seconds()), "next_hop": hop + 1}
    return {"mode": "stop", "reason": "the next two forecasts exist", "hop": hop}


def github_exists(repo: str, token: str | None) -> Callable[[dt.date], bool]:
    def exists(day: dt.date) -> bool:
        url = f"https://api.github.com/repos/{repo}/contents/forecasts/live/{day}.csv?ref=main"
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"Accept": "application/vnd.github+json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            urllib.request.urlopen(req, timeout=30)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise
    return exists


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--hop", default="0")
    p.add_argument("--now", help="ISO time with offset, for tests")
    a = p.parse_args()
    now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now(dt.UTC)
    hop = int(a.hop or 0)
    d = decide(now, github_exists(a.repo, os.environ.get("GH_TOKEN")), hop)
    print(json.dumps({"now_berlin": now.astimezone(BERLIN).isoformat(), **d}))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            for k in ("mode", "wait_s", "sleep_s", "next_hop"):
                f.write(f"{k}={d.get(k, 0)}\n")


if __name__ == "__main__":
    main()
