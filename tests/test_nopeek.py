"""The no-peeking check must catch look-ahead and pass honest feature code."""

import numpy as np
import pandas as pd
import pytest

from powercast import nopeek


@pytest.fixture
def tables():
    idx = pd.date_range("2024-03-20", "2024-04-10", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({"price": rng.normal(80, 20, len(idx)), "mtu_min": 60}, index=idx)
    q = pd.date_range("2024-03-20", "2024-04-10", freq="15min", tz="UTC", inclusive="left")
    fund = pd.DataFrame({"load_forecast": rng.normal(5e4, 5e3, len(q)),
                         "load_actual": rng.normal(5e4, 5e3, len(q))}, index=q)
    return {"prices": prices, "fund": fund}


def _day_slice(df, day, offset_days):
    d = pd.Timestamp(day) + pd.Timedelta(days=offset_days)
    start = d.tz_localize("Europe/Berlin").tz_convert("UTC")
    end = (d + pd.Timedelta(days=1)).tz_localize("Europe/Berlin").tz_convert("UTC")
    return df[(df.index >= start) & (df.index < end)]


def honest(t, day):
    # yesterday's prices, today's load forecast for the target day, actual load 2 days ago
    return np.concatenate([
        _day_slice(t["prices"]["price"], day, -1).to_numpy(),
        _day_slice(t["fund"]["load_forecast"], day, 0).to_numpy(),
        _day_slice(t["fund"]["load_actual"], day, -2).to_numpy(),
    ])


def peeking_price(t, day):
    return _day_slice(t["prices"]["price"], day, 0).to_numpy()  # the target itself


def peeking_actual(t, day):
    return _day_slice(t["fund"]["load_actual"], day, -1).to_numpy()  # yesterday afternoon


DAYS = [pd.Timestamp("2024-03-30"), pd.Timestamp("2024-03-31"), pd.Timestamp("2024-04-02")]


def test_honest_passes(tables):
    nopeek.assert_no_peeking(honest, tables, DAYS)


@pytest.mark.parametrize("fn", [peeking_price, peeking_actual])
def test_peeking_is_caught(tables, fn):
    with pytest.raises(AssertionError, match="look-ahead"):
        nopeek.assert_no_peeking(fn, tables, DAYS)


def test_issue_time_handles_dst():
    # delivery 31 Mar 2024 (23-hour day): issued 30 Mar 11:00 CET = 10:00 UTC
    assert nopeek.issue_time(pd.Timestamp("2024-03-31")) == pd.Timestamp("2024-03-30 10:00",
                                                                          tz="UTC")
    # delivery 1 Apr 2024: issued 31 Mar 11:00 CEST = 09:00 UTC
    assert nopeek.issue_time(pd.Timestamp("2024-04-01")) == pd.Timestamp("2024-03-31 09:00",
                                                                         tz="UTC")


def test_price_of_delivery_day_is_unpublished_at_issue():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-04-01 22:00", tz="UTC")])  # 2 Apr 00:00 CEST
    assert nopeek.available_at("price", idx)[0] > nopeek.issue_time(pd.Timestamp("2024-04-02"))
    assert nopeek.available_at("price", idx)[0] < nopeek.issue_time(pd.Timestamp("2024-04-03"))
