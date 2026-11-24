"""Battery arbitrage driven by each forecast (docs/EVALUATION_PLAN.md, section 7).

Usage:
    python scripts/run_battery.py --snapshot 2026-11-23 --period test \
        --forecast M0=data/forecasts/m0_naive/... --forecast M1=... [...]

For every delivery day and model: schedule a 1 MW / 2 MWh battery on the forecast mean
price of each MTU (mean of the 99 quantiles, after the quarter-hour layer), settle it at
the real prices. Also the perfect-foresight schedule (upper bound). Reports profit in
EUR per MW per year with 95 % block-bootstrap intervals, and the paired difference to M0,
overall and by market period. Writes results/battery/<period>.json and daily profits
to data/scores/<period>/battery.parquet.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from powercast import battery, dataset, evaluate, stats
from powercast.config import DATA_DIR, RESULTS_DIR
from powercast.quarterhour import QuarterHourLayer

PERIODS = {"validation": ("2019-01-01", "2020-12-31"), "test": ("2021-01-01", "2026-10-31")}
REGIMES = {"R0": ("2021-01-01", "2021-08-31"), "R1": ("2021-09-01", "2022-12-31"),
           "R2": ("2023-01-01", "2025-09-30"), "R3": ("2025-10-01", "2026-10-31")}
DAYS_PER_YEAR = 365.25


def daily_profit(m: pd.DataFrame, price_fc: np.ndarray, b: battery.Battery) -> pd.Series:
    """Profit (EUR) per delivery day of the schedule optimised on ``price_fc``."""
    out = {}
    y = m["y"].to_numpy()
    for d, idx in m.groupby("day").indices.items():
        dt_h = float(m["mtu_min"].iloc[idx[0]]) / 60
        net = battery.schedule(price_fc[idx], dt_h, b)
        out[d] = battery.profit(net, y[idx], dt_h)
    return pd.Series(out).sort_index()


def per_year(x: pd.Series, seed: int = 0) -> dict:
    mean, lo, hi = stats.bootstrap_mean_ci(x.to_numpy(), block=14, seed=seed)
    return {"mean": mean * DAYS_PER_YEAR, "ci95": [lo * DAYS_PER_YEAR, hi * DAYS_PER_YEAR]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--period", choices=list(PERIODS), required=True)
    p.add_argument("--forecast", action="append", required=True, help="NAME=path.npz")
    p.add_argument("--baseline", default="M0")
    a = p.parse_args()
    t = dataset.load(a.snapshot)
    days = pd.date_range(*PERIODS[a.period])
    b = battery.Battery()
    layer = QuarterHourLayer(t["prices"], t["fund"])

    m = evaluate.mtu_table(t["prices"], days)
    profits = {"perfect": daily_profit(m, m["y"].to_numpy(), b)}
    for spec in a.forecast:
        name, path = spec.split("=", 1)
        fdays, q = evaluate.load_forecast(path)
        if not days.isin(fdays).all():
            raise ValueError(f"{name}: forecast does not cover the {a.period} period")
        m, qm = evaluate.mtu_quantiles(t["prices"], t["fund"], days,
                                       q[fdays.get_indexer(days)], layer)
        profits[name] = daily_profit(m, qm.mean(axis=1), b)
        print(name, f"{profits[name].mean() * DAYS_PER_YEAR:.0f} EUR/MW/yr", flush=True)

    df = pd.DataFrame(profits)
    out = DATA_DIR / "scores" / a.period
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "battery.parquet")

    segments = {"all": (days[0], days[-1])}
    if a.period == "test":
        segments.update({k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in REGIMES.items()})
    report = {"snapshot": a.snapshot, "period": a.period,
              "battery": {"power_mw": b.power_mw, "energy_mwh": b.energy_mwh,
                          "round_trip": b.round_trip},
              "units": "EUR per MW per year", "segments": {}}
    for seg, (lo, hi) in segments.items():
        s = df.loc[lo:hi]
        r = {"days": int(len(s)), "profit": {n: per_year(s[n]) for n in s.columns},
             "share_of_perfect": {n: float(s[n].sum() / s["perfect"].sum()) for n in s.columns}}
        if a.baseline in s.columns:
            r["minus_" + a.baseline] = {n: per_year(s[n] - s[a.baseline]) for n in s.columns
                                        if n not in (a.baseline, "perfect")}
            r["dm_p_more_profit_than_" + a.baseline] = {
                n: stats.diebold_mariano((-s[n]).to_numpy(), (-s[a.baseline]).to_numpy())[1]
                for n in s.columns if n not in (a.baseline, "perfect")}
        report["segments"][seg] = r
    dest = RESULTS_DIR / "battery"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{a.period}.json").write_text(json.dumps(report, indent=1))
    for n, v in report["segments"]["all"]["profit"].items():
        print(f"{n:10s} {v['mean']:9.0f} [{v['ci95'][0]:9.0f}, {v['ci95'][1]:9.0f}] EUR/MW/yr")


if __name__ == "__main__":
    main()
