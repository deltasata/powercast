"""The daily job's gate, replayed on the runs GitHub actually started (24-26 Nov 2026)."""

import datetime as dt
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gate", Path(__file__).resolve().parents[1] / "scripts" / "gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

UTC = dt.UTC


def have(*days):
    s = {dt.date.fromisoformat(d) for d in days}
    return lambda day: day in s


def follow(start, exists, hop=0, max_runs=10):
    """Run the chain as GitHub would; return the decisions in order."""
    now, out = start, []
    for _ in range(max_runs):
        d = gate.decide(now, exists, hop)
        out.append((now, d))
        if d["mode"] != "relay":
            return out
        now = now + dt.timedelta(seconds=d["sleep_s"])  # fresh run starts at once
        hop = d["next_hop"]
    raise AssertionError("chain did not end")


def test_26_nov_run_at_0610_berlin_hands_over_and_forecasts_on_time():
    chain = follow(dt.datetime(2026, 11, 26, 5, 10, tzinfo=UTC), have("2026-11-26"))
    now, d = chain[-1]
    assert d["mode"] == "forecast" and d["day"] == "2026-11-27"
    start = now + dt.timedelta(seconds=d["wait_s"])
    assert start.astimezone(gate.BERLIN).time() == dt.time(10, 15)
    assert len(chain) == 2


def test_evening_run_seeds_next_morning():
    # 24 Nov 19:41 UTC: forecast for 25 Nov existed, so the chain aims at 26 Nov
    chain = follow(dt.datetime(2026, 11, 24, 19, 41, tzinfo=UTC), have("2026-11-25"))
    now, d = chain[-1]
    assert d["mode"] == "forecast" and d["day"] == "2026-11-26"
    assert d["wait_s"] <= gate.MAX_WAIT.total_seconds()
    assert len(chain) - 1 <= gate.MAX_HOPS


def test_missing_forecast_after_issue_time_runs_now():
    d = gate.decide(dt.datetime(2026, 11, 26, 13, 0, tzinfo=UTC), have(), 0)
    assert d["mode"] == "forecast" and d["wait_s"] == 0 and d["day"] == "2026-11-27"


def test_both_forecasts_exist_stops():
    d = gate.decide(dt.datetime(2026, 11, 26, 13, 0, tzinfo=UTC),
                    have("2026-11-27", "2026-11-28"), 0)
    assert d["mode"] == "stop"


def test_hop_limit_and_worst_case():
    d = gate.decide(dt.datetime(2026, 11, 26, 11, 5, tzinfo=UTC), have("2026-11-27"),
                    gate.MAX_HOPS)
    assert d["mode"] == "stop"
    # worst case: a trigger just after the forecast was made, about 23 h before the next
    chain = follow(dt.datetime(2026, 11, 26, 11, 1, tzinfo=UTC), have("2026-11-27"))
    assert chain[-1][1]["mode"] == "forecast" and len(chain) - 1 <= gate.MAX_HOPS


def test_summer_time():
    chain = follow(dt.datetime(2026, 7, 1, 3, 0, tzinfo=UTC), have("2026-07-01"))
    now, d = chain[-1]
    start = now + dt.timedelta(seconds=d["wait_s"])
    assert start == dt.datetime(2026, 7, 1, 8, 15, tzinfo=UTC)
