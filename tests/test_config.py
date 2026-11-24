"""Sanity checks on the project constants."""

import pandas as pd

import powercast
from powercast import config


def test_package_imports():
    assert powercast.__version__


def test_quarter_hour_start_is_berlin_midnight_in_utc():
    # 1 October 2025 is in summer time (CEST = UTC+2),
    # so Berlin midnight is 22:00 UTC on the day before.
    utc = config.QUARTER_HOUR_START_LOCAL.tz_convert("UTC")
    assert utc == pd.Timestamp("2025-09-30 22:00", tz="UTC")
