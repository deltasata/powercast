"""Build processed tables from a raw SMARD snapshot and run the data checks.

Usage:
    python scripts/build_dataset.py --snapshot 2026-11-23

Writes data/processed/<snapshot>/{prices,fundamentals}.parquet and
results/data_checks/<snapshot>.json.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from powercast.config import PROCESSED_DIR, QUARTER_HOUR_START_LOCAL, RAW_DIR, RESULTS_DIR
from powercast.data import build, checks, energycharts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--ec", help="Energy-Charts snapshot used to fill missing SMARD prices")
    a = p.parse_args()
    raw = RAW_DIR / "smard" / a.snapshot
    out = PROCESSED_DIR / a.snapshot
    out.mkdir(parents=True, exist_ok=True)

    prices = build.build_prices(raw)
    rep: dict = {"snapshot": a.snapshot, "prices": {}, "fundamentals": {}}
    if a.ec:
        ec = energycharts.load(RAW_DIR / "energy-charts" / a.ec)["price"]
        prices = build.fill_prices_from_ec(prices, ec)
        filled = prices.index[prices["source"] == "energy-charts"]
        rep["prices"]["filled_from_energy_charts"] = {
            "snapshot": a.ec, "n": int(len(filled)),
            "days": sorted({str(d.date()) for d in build.delivery_day(filled)})}
    fund = build.build_fundamentals(raw)
    prices.to_parquet(out / "prices.parquet")
    fund.to_parquet(out / "fundamentals.parquet")

    qh = QUARTER_HOUR_START_LOCAL.tz_convert("UTC")
    rep["prices"]["hourly_part"] = checks.grid_report(prices.loc[prices.index < qh, "price"], "1h")
    rep["prices"]["quarterhour_part"] = checks.grid_report(
        prices.loc[prices.index >= qh, "price"], "15min")

    days = checks.price_day_counts(prices)
    bad = days[days["n"] != days["expected"]]
    rep["prices"]["days"] = int(len(days))
    rep["prices"]["days_with_wrong_count"] = {str(d.date()): [int(r.n), int(r.expected)]
                                              for d, r in bad.iterrows()}
    dst = days[days["expected"].isin([23, 25, 92, 100])]
    rep["prices"]["dst_days_checked"] = int(len(dst))
    rep["prices"]["dst_days_ok"] = int((dst["n"] == dst["expected"]).sum())

    pr = prices["price"]
    rep["prices"]["min"] = float(pr.min())
    rep["prices"]["max"] = float(pr.max())
    rep["prices"]["n_negative"] = int((pr < 0).sum())
    rep["prices"]["n_above_500"] = int((pr > 500).sum())

    # SMARD's hourly file after the switch vs the mean of the quarter-hour prices
    h = build.smard.read(raw / "4169_DE_hour")
    h = h[(h.index >= qh)].dropna()
    q = prices.loc[prices.index >= qh, "price"]
    qmean = q.groupby(q.index.floor("h")).mean()
    common = h.index.intersection(qmean.index)
    d = (h[common] - qmean[common]).abs()
    rep["prices"]["hourly_file_vs_qh_mean_after_switch"] = {
        "n_hours": int(len(common)), "max_abs_diff": float(d.max()),
        "n_diff_gt_0.01": int((d > 0.01).sum())}

    for c in fund.columns:
        rep["fundamentals"][c] = checks.grid_report(fund[c], "15min")

    days_f = build.delivery_day(fund.index)
    nday = fund.groupby(days_f).size()
    exp = pd.Series([checks.expected_intervals(x, 15) for x in nday.index], index=nday.index)
    rep["fundamentals"]["days_with_wrong_count"] = {
        str(k.date()): [int(nday[k]), int(exp[k])] for k in nday.index[nday != exp]}

    dest = RESULTS_DIR / "data_checks"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{a.snapshot}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
