"""Seed ensemble of model M3: average of the seeds' quantiles (EVALUATION_PLAN section 4).

Usage:
    python scripts/ensemble_seeds.py --snapshot 2026-11-23 --out <ensemble.npz> \
        <seed0.npz> <seed1.npz> ...

Averaging quantiles at fixed levels (Vincentization) keeps them sorted, since each seed's
quantiles are sorted. Prints validation scores of each seed, their mean and spread, and
of the ensemble (test-period scores are withheld).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from powercast import dataset, evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    days, qs = None, []
    for f in a.files:
        d, q = evaluate.load_forecast(f)
        if days is not None and not d.equals(days):
            raise ValueError(f"{f}: different delivery days")
        days = d
        qs.append(q)
    ens = np.mean(qs, axis=0)
    evaluate.save_forecast(a.out, days, ens, seeds=[str(f) for f in a.files])
    t = dataset.load(a.snapshot)
    if days[-1] <= evaluate.VALIDATION_END:
        per_seed = pd.DataFrame([evaluate.summary(t["prices"], days, q) for q in qs])
        cols = ["crps", "mae", "cov50", "cov80", "cov90", "cov98"]
        print("seeds: mean\n", per_seed[cols].mean().round(4).to_string())
        print("seeds: standard deviation\n", per_seed[cols].std().round(4).to_string())
    print("ensemble", evaluate.summary_if_validation(t["prices"], days, ens))


if __name__ == "__main__":
    main()
