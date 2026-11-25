"""Robustness checks on the test period, from saved forecasts and scores only.

Usage:
    python scripts/robustness.py --snapshot 2026-11-23

Nothing is refitted. Writes results/evaluation/robustness_test.json with
1. paired battery-profit differences between the learned models (block bootstrap, DM);
2. the memorisation check for Chronos-2 (M4): CRPS of M4, M1 and both M3 on the days
   after the release of the Chronos-2 weights (1 Nov 2025 – 31 Oct 2026), with DM tests;
3. the spread over the five seeds of each M3 network (EVALUATION_PLAN section 4): CRPS
   and 80 % coverage of every seed, scored exactly like the ensemble in
   scripts/evaluate_models.py (quarter-hour layer, clipping to the price limits).
Reads data/scores/test/*.parquet (written by evaluate_models.py and run_battery.py) and
the seed forecasts in data/forecasts/ddnn/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from powercast import dataset, evaluate, stats
from powercast.config import DATA_DIR, RESULTS_DIR

DAYS = ("2021-01-01", "2026-10-31")
RELEASE = ("2025-11-01", "2026-10-31")  # after the Chronos-2 weights were published
YEAR = 365.25


def ci(x, scale=1.0) -> dict:
    m, lo, hi = stats.bootstrap_mean_ci(np.asarray(x), block=14)
    return {"mean": m * scale, "ci95": [lo * scale, hi * scale]}


def seed_scores(t, days, path) -> dict:
    fdays, q = evaluate.load_forecast(path)
    q = q[fdays.get_indexer(days)]
    m, qm = evaluate.mtu_quantiles(t["prices"], t["fund"], days, q)
    daily = evaluate.daily_scores(m, qm).groupby("day").mean(numeric_only=True)
    return {"crps": float(daily["crps"].mean()), "cov80": float(daily["in80"].mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--jobs", type=int, default=10)
    a = p.parse_args()
    sc = DATA_DIR / "scores" / "test"
    out = {}

    b = pd.read_parquet(sc / "battery.parquet")
    for x, y in [("M4", "M1"), ("M3-JSU", "M1"), ("M4", "M3-JSU")]:
        d = ci((b[x] - b[y]).to_numpy(), YEAR)
        d["dm_p"] = stats.diebold_mariano((-b[x]).to_numpy(), (-b[y]).to_numpy())[1]
        out[f"profit_{x}_minus_{y}"] = d

    lo, hi = RELEASE
    s = {n: pd.read_parquet(sc / f"{n}.parquet").groupby("day")["crps"].mean().loc[lo:hi]
         for n in ["M1", "M3-JSU", "M3-normal", "M4"]}
    out["crps_from_release"] = {"days": [lo, hi], **{n: ci(v) for n, v in s.items()}}
    for n in ["M1", "M3-JSU", "M3-normal"]:
        out[f"dm_M4_lt_{n}_from_{lo}"] = stats.diebold_mariano(s["M4"].to_numpy(),
                                                               s[n].to_numpy())[1]

    t = dataset.load(a.snapshot)
    days = pd.date_range(*DAYS)
    ddnn = DATA_DIR / "forecasts" / "ddnn"
    jobs = [(dist, k) for dist in ("normal", "jsu") for k in range(5)]
    res = Parallel(n_jobs=a.jobs)(
        delayed(seed_scores)(t, days, ddnn / f"{dist}_seed{k}_{DAYS[0]}_{DAYS[1]}.npz")
        for dist, k in jobs)
    for dist in ("normal", "jsu"):
        r = [x for (d_, _), x in zip(jobs, res, strict=True) if d_ == dist]
        crps = np.array([x["crps"] for x in r])
        cov = np.array([x["cov80"] for x in r])
        out[f"M3-{dist}_seeds"] = {"crps": crps.tolist(), "cov80": cov.tolist(),
                                   "crps_mean": float(crps.mean()),
                                   "crps_sd": float(crps.std(ddof=1)),
                                   "cov80_mean": float(cov.mean()),
                                   "cov80_sd": float(cov.std(ddof=1))}

    dest = RESULTS_DIR / "evaluation" / "robustness_test.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
