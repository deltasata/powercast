"""Rolling LEAR point forecasts (input to QRA, model M1).

Usage:
    python scripts/run_lear.py --snapshot 2026-11-23 --start 2018-01-01 --end 2020-12-31

For each delivery day, fits LEAR on each calibration window in powercast.models.lear.WINDOWS
using only the days before it, and saves the 24 × len(WINDOWS) point forecasts to
data/forecasts/lear_points/<snapshot>_<start>_<end>.parquet. The no-peeking check runs
on the full forecast function first and stops the run if it fails.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from powercast import dataset, features, nopeek
from powercast.config import DATA_DIR
from powercast.models import lear

CHECK_DAYS = ["2018-03-25", "2018-10-28", "2019-06-05", "2020-01-31", "2020-02-01",
              "2021-09-15", "2025-10-26"]


def inputs(tables, lags):
    days = pd.date_range("2015-01-12", tables["prices"].index.max().tz_convert(None).normalize())
    X = features.build(tables, days, lags)
    Y = dataset.fill_from_last_week(dataset.to_slots(tables["prices"]["price"])).reindex(days)
    return X, Y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--jobs", type=int, default=60)
    a = p.parse_args()
    tables = dataset.load(a.snapshot)
    tables = {"prices": tables["prices"][["price"]], "fund": tables["fund"]}
    lags = features.default_exog_lags()
    print("exogenous lags:", lags, flush=True)

    def forecast_fn(tb, day):
        X, Y = inputs(tb, lags)
        return lear.lear_day(X, Y, day, lear.WINDOWS[0])

    check = [pd.Timestamp(d) for d in CHECK_DAYS if pd.Timestamp(d) <= pd.Timestamp(a.end)]
    t0 = time.time()
    nopeek.assert_no_peeking(forecast_fn, tables, check)
    print(f"no-peeking check passed on {len(check)} days ({time.time() - t0:.0f} s)",
          flush=True)

    X, Y = inputs(tables, lags)
    days = pd.date_range(a.start, a.end)

    def one(day):
        return np.column_stack([lear.lear_day(X, Y, day, w) for w in lear.WINDOWS])

    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(d) for d in days)
    print(f"{len(days)} days in {time.time() - t0:.0f} s", flush=True)
    arr = np.stack(res)  # (days, 24, windows)
    cols = pd.MultiIndex.from_product([range(24), lear.WINDOWS], names=["slot", "window"])
    df = pd.DataFrame(arr.reshape(len(days), -1), index=days, columns=cols)
    df.columns = [f"{s}_{w}" for s, w in df.columns]
    out = DATA_DIR / "forecasts" / "lear_points"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{a.snapshot}_{a.start}_{a.end}.parquet")


if __name__ == "__main__":
    main()
