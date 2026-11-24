"""Information-set rules and the automated no-peeking check.

Every input series has a *publication rule*: for each value it gives ``available_at``,
the earliest UTC time at which the value was public. A forecast for delivery day D
(Berlin) is issued at ``issue_time(D)`` and may only use values with
``available_at <= issue_time(D)``.

The check (:func:`assert_no_peeking`) is a perturbation test: it builds the inputs of a
forecast twice, once from the true tables and once from tables in which every value
published after the issue time is replaced by random numbers. Any difference means the
feature code used information it could not have had. The test does not trust the feature
code, so it catches off-by-one shifts, wrong time zones and centred rolling windows alike.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from powercast.config import GATE_CLOSURE_LOCAL, MARKET_TZ

# Forecasts are issued one hour before gate closure, leaving time to submit bids.
ISSUE_TIME_LOCAL = pd.Timedelta(hours=GATE_CLOSURE_LOCAL.hour - 1)


def _local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(MARKET_TZ).normalize()


def issue_time(delivery_day: pd.Timestamp) -> pd.Timestamp:
    """UTC issue time of the forecast for a Berlin delivery day."""
    d = pd.Timestamp(delivery_day)
    d = d.tz_localize(MARKET_TZ) if d.tzinfo is None else d.tz_convert(MARKET_TZ)
    prev = (d.normalize() - pd.Timedelta(days=1)).tz_localize(None).tz_localize(MARKET_TZ)
    return (prev + ISSUE_TIME_LOCAL).tz_convert("UTC")


def _day_before_at(hours: float) -> Callable[[pd.DatetimeIndex], pd.DatetimeIndex]:
    """Published on the day before delivery at a fixed Berlin clock time."""

    def rule(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        day = _local_day(index).tz_localize(None) - pd.Timedelta(days=1)
        t = (day + pd.Timedelta(hours=hours)).tz_localize(MARKET_TZ)
        return t.tz_convert("UTC")

    return rule


def _after_interval(lag: pd.Timedelta, width: pd.Timedelta):
    def rule(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return index + width + lag

    return rule


# Publication rules (see docs/DECISIONS.md D3.3 and the measurements in docs/LOG.md).
RULES: dict[str, Callable[[pd.DatetimeIndex], pd.DatetimeIndex]] = {
    # auction results are published around 12:45 on the day before delivery
    "price": _day_before_at(13.0),
    # TSO day-ahead forecasts (hour of publication on the day before delivery)
    "load_forecast": _day_before_at(10.0),
    "wind_onshore_forecast": _day_before_at(18.0),
    "wind_offshore_forecast": _day_before_at(18.0),
    "solar_forecast": _day_before_at(18.0),
    # measured values: published with a delay after the end of each 15-min interval
    "load_actual": _after_interval(pd.Timedelta(hours=2), pd.Timedelta(minutes=15)),
    "wind_onshore_actual": _after_interval(pd.Timedelta(hours=2), pd.Timedelta(minutes=15)),
    "wind_offshore_actual": _after_interval(pd.Timedelta(hours=2), pd.Timedelta(minutes=15)),
    "solar_actual": _after_interval(pd.Timedelta(hours=2), pd.Timedelta(minutes=15)),
}


def available_at(column: str, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if column not in RULES:
        raise KeyError(f"no publication rule for input '{column}'")
    return RULES[column](index)


def mask_unpublished(table: pd.DataFrame, cutoff: pd.Timestamp, fill: str = "nan",
                     rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Copy of ``table`` with values published after ``cutoff`` replaced.

    fill="nan" gives the honest information set; fill="noise" gives random values,
    used by the perturbation test.
    """
    out = table.copy()
    for c in out.columns:
        if c in ("mtu_min", "zone"):
            continue
        late = np.asarray(available_at(c, out.index) > cutoff)
        if not late.any():
            continue
        if fill == "nan":
            out.loc[late, c] = np.nan
        else:
            rng = rng or np.random.default_rng(0)
            out.loc[late, c] = rng.normal(1e4, 1e4, late.sum())
    return out


def assert_no_peeking(
    make_inputs: Callable[[dict[str, pd.DataFrame], pd.Timestamp], object],
    tables: dict[str, pd.DataFrame],
    delivery_days: Iterable[pd.Timestamp],
    seed: int = 0,
) -> None:
    """Raise AssertionError if ``make_inputs`` uses anything unpublished at issue time.

    ``make_inputs(tables, day)`` must return the model inputs (array, Series or
    DataFrame) for delivery ``day``. It receives the full tables; it is the function's
    job to look only at the past, and this check verifies that it does.
    """
    rng = np.random.default_rng(seed)
    for day in delivery_days:
        cut = issue_time(day)
        ref = make_inputs(tables, day)
        noisy = {k: mask_unpublished(v, cut, fill="noise", rng=rng) for k, v in tables.items()}
        got = make_inputs(noisy, day)
        a = np.asarray(ref, dtype=float)
        b = np.asarray(got, dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
            raise AssertionError(f"inputs for {pd.Timestamp(day).date()} change when "
                                 f"data published after {cut} is perturbed: look-ahead")
