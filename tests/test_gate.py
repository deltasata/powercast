"""The daily job's gate, replayed on the runs GitHub actually started (24-28 Nov 2026)."""

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


def test_evening_run_stops():
    # 24 Nov 19:41 UTC: forecast for 25 Nov existed; 10:15 on 25 Nov is 13.6 h away
    d = gate.decide(dt.datetime(2026, 11, 24, 19, 41, tzinfo=UTC), have("2026-11-25"), 0)
    assert d["mode"] == "stop" and d["day"] == "2026-11-26"


def test_run_after_the_forecast_stops_at_once():
    # 27 Nov 09:39 UTC (started 09:24 UTC after the 28 Nov forecast was published)
    d = gate.decide(dt.datetime(2026, 11, 27, 9, 24, tzinfo=UTC), have("2026-11-28"), 0)
    assert d["mode"] == "stop"


def test_missing_forecast_after_issue_time_runs_now():
    d = gate.decide(dt.datetime(2026, 11, 26, 13, 0, tzinfo=UTC), have(), 0)
    assert d["mode"] == "forecast" and d["wait_s"] == 0 and d["day"] == "2026-11-27"


def test_both_forecasts_exist_stops():
    d = gate.decide(dt.datetime(2026, 11, 26, 13, 0, tzinfo=UTC),
                    have("2026-11-27", "2026-11-28"), 0)
    assert d["mode"] == "stop"


def test_hop_limit_and_worst_case():
    d = gate.decide(dt.datetime(2026, 11, 26, 1, 20, tzinfo=UTC), have("2026-11-26"),
                    gate.MAX_HOPS)
    assert d["mode"] == "stop"
    # worst case: a trigger at the opening of the window, 8 h before 10:15
    start = dt.datetime(2026, 11, 26, 1, 15, tzinfo=UTC)
    chain = follow(start, have("2026-11-26"))
    now, d = chain[-1]
    assert d["mode"] == "forecast" and len(chain) - 1 <= gate.MAX_HOPS
    end = now + dt.timedelta(seconds=d["wait_s"])
    assert end - start <= gate.WINDOW
    assert d["wait_s"] <= gate.MAX_WAIT.total_seconds()


# start times (UTC) of the scheduled runs GitHub created, 26 Nov 05:10 - 28 Nov 07:53
FIRED = """26 05:10, 26 11:31, 26 17:52, 26 18:23, 26 18:59, 26 19:49, 26 22:16, 26 23:59,
27 01:47, 27 03:37, 27 08:39, 27 11:27, 27 14:04, 27 15:39, 27 17:38, 27 18:08,
27 18:57, 27 19:39, 27 21:55, 27 23:39, 28 01:29, 28 03:15, 28 07:53"""


def test_replay_observed_triggers():
    """Each morning is served on time, and no runner is busy outside the 8 h window."""
    fired = [dt.datetime(2026, 11, int(x[:2]), int(x[3:5]), int(x[6:8]), tzinfo=UTC)
             for x in FIRED.replace("\n", " ").split(", ")]
    made = {dt.date(2026, 11, 26)}                 # forecasts on main before the replay
    exists = lambda day: day in made               # noqa: E731
    busy_until = None
    for t in fired:
        if busy_until and t < busy_until:          # queued behind the running chain
            t = busy_until
        chain = follow(t, exists)
        now, d = chain[-1]
        if d["mode"] == "stop":
            assert d["reason"] in ("outside the window", "the next two forecasts exist")
            continue
        end = now + dt.timedelta(seconds=d["wait_s"])
        issue = dt.datetime.fromisoformat(d["issue"])
        assert end <= max(issue, t) + dt.timedelta(minutes=1)
        assert issue - t <= gate.WINDOW
        made.add(dt.date.fromisoformat(d["day"]))
        busy_until = end + dt.timedelta(minutes=10)
    assert {dt.date(2026, 11, 27), dt.date(2026, 11, 28), dt.date(2026, 11, 29)} <= made


def test_summer_time():
    chain = follow(dt.datetime(2026, 7, 1, 3, 0, tzinfo=UTC), have("2026-07-01"))
    now, d = chain[-1]
    start = now + dt.timedelta(seconds=d["wait_s"])
    assert start == dt.datetime(2026, 7, 1, 8, 15, tzinfo=UTC)
