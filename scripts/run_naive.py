"""Naive forecasts (model M0): same local hour one week earlier + past error quantiles.

Usage:
    python scripts/run_naive.py --snapshot 2026-11-23 --start 2019-01-01 --end 2020-12-31 \
        --window 364

The no-peeking check runs on the full forecast first. Output:
data/forecasts/m0_naive/w<window>_<start>_<end>.npz
"""

from __future__ import annotations

import argparse

import pandas as pd

from powercast import dataset, evaluate, nopeek
from powercast.config import DATA_DIR
from powercast.models import naive

CHECK_DAYS = ["2019-03-31", "2019-10-27", "2020-02-29", "2021-09-15", "2025-10-26"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window", type=int, default=364)
    a = p.parse_args()
    t = dataset.load(a.snapshot)
    tables = {"prices": t["prices"][["price"]]}

    def fn(tb, day):
        return naive.forecast(dataset.to_slots(tb["prices"]["price"]), pd.DatetimeIndex([day]),
                              a.window)

    check = [pd.Timestamp(d) for d in CHECK_DAYS if pd.Timestamp(d) <= pd.Timestamp(a.end)]
    nopeek.assert_no_peeking(fn, tables, check)
    print(f"no-peeking check passed on {len(check)} days")
    days = pd.date_range(a.start, a.end)
    q = naive.forecast(dataset.to_slots(t["prices"]["price"]), days, a.window)
    out = DATA_DIR / "forecasts" / "m0_naive" / f"w{a.window}_{a.start}_{a.end}.npz"
    evaluate.save_forecast(out, days, q, window=a.window)
    print(evaluate.summary_if_validation(t["prices"], days, q))


if __name__ == "__main__":
    main()
