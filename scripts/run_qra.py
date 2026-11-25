"""QRA on rolling LEAR point forecasts (model M1).

Usage:
    python scripts/run_qra.py --points data/forecasts/lear_points/<file>.parquet \
        --start 2019-01-01 --end 2020-12-31 --window 182 --snapshot 2026-11-23

``--points`` takes one or more files of LEAR point forecasts (from scripts/run_lear.py);
they are joined in time order, so the QRA window can reach back into an earlier run
(the test period starts in 2021 and its 364-day window needs the LEAR forecasts of 2020).

For each delivery day D, fits the quantile regressions on the ``window`` days before D
(LEAR forecasts and real prices of those days are known at issue time) and predicts D.
Saves data/forecasts/m1_lear_qra/w<window>_<start>_<end>.npz.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from powercast import dataset, evaluate
from powercast.config import DATA_DIR
from powercast.models import lear


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--points", nargs="+", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--jobs", type=int, default=60)
    a = p.parse_args()
    pts = pd.concat([pd.read_parquet(f) for f in a.points]).sort_index()
    if pts.index.duplicated().any():
        raise ValueError("the LEAR files overlap")
    nw = len(lear.WINDOWS)
    P = pts.to_numpy().reshape(len(pts), 24, nw)
    t = dataset.load(a.snapshot)
    Y = dataset.to_slots(t["prices"]["price"]).reindex(pts.index).to_numpy(float)
    days = pd.date_range(a.start, a.end)
    pos = pts.index.get_indexer(days)
    if (pos < a.window).any():
        raise ValueError("not enough LEAR forecasts before the first day")

    def one(i):
        return lear.qra_day(P[i - a.window:i], Y[i - a.window:i], P[i])

    q = np.stack(Parallel(n_jobs=a.jobs, verbose=2)(delayed(one)(i) for i in pos))
    out = DATA_DIR / "forecasts" / "m1_lear_qra" / f"w{a.window}_{a.start}_{a.end}.npz"
    evaluate.save_forecast(out, days, q, window=a.window, points=a.points)
    print(evaluate.summary_if_validation(t["prices"], days, q))


if __name__ == "__main__":
    main()
