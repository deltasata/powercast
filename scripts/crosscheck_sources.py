"""Cross-check SMARD against Energy-Charts on overlapping series.

Usage:
    python scripts/crosscheck_sources.py --smard 2026-11-23 --ec 2026-11-23

For every series: number of common timestamps, mean difference (SMARD − Energy-Charts),
mean and max absolute difference, share of timestamps with |difference| above a
tolerance, and the correlation. Writes results/data_checks/crosscheck_<smard>.json and
lists the largest differences.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from powercast.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from powercast.data import energycharts


def compare(a: pd.Series, b: pd.Series, tol: float) -> dict:
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) == 0:
        return {"n_common": 0}
    x, y = a[common], b[common]
    d = x - y
    top = d.abs().sort_values(ascending=False).head(5)
    return {
        "n_common": int(len(common)),
        "first": str(common.min()), "last": str(common.max()),
        "mean_diff": float(d.mean()),
        "mean_abs_diff": float(d.abs().mean()),
        "max_abs_diff": float(d.abs().max()),
        "tolerance": tol,
        "share_above_tol": float((d.abs() > tol).mean()),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "mean_smard": float(x.mean()),
        "largest": [(str(t), float(x[t]), float(y[t])) for t in top.index],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smard", required=True)
    p.add_argument("--ec", required=True)
    a = p.parse_args()
    prices = pd.read_parquet(PROCESSED_DIR / a.smard / "prices.parquet")
    prices = prices[prices["source"] == "smard"]  # never compare a source with itself
    fund = pd.read_parquet(PROCESSED_DIR / a.smard / "fundamentals.parquet")
    ec = energycharts.load(RAW_DIR / "energy-charts" / a.ec)

    rep = {"smard_snapshot": a.smard, "ec_snapshot": a.ec, "series": {}}
    rep["series"]["price"] = compare(prices["price"], ec["price"], tol=0.01)
    by_year = {}
    for y in range(2015, 2027):
        sl = slice(f"{y}-01-01", f"{y}-12-31")
        by_year[y] = compare(prices["price"].loc[sl], ec["price"].loc[sl], tol=0.01)
    rep["series"]["price_by_year"] = {y: {k: v[k] for k in ("n_common", "share_above_tol",
                                                            "max_abs_diff")}
                                      for y, v in by_year.items() if v["n_common"]}
    for col in fund.columns:
        if col in ec:
            rep["series"][col] = compare(fund[col], ec[col], tol=0.01 * fund[col].abs().mean())
    # prices on the two SMARD-missing days, if Energy-Charts has them
    gap = ec["price"].loc["2026-11-19 23:00":"2026-11-21 22:45"]
    rep["series"]["ec_price_on_smard_gap_days"] = int(gap.notna().sum())

    dest = RESULTS_DIR / "data_checks"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"crosscheck_{a.smard}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
