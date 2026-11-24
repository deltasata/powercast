"""Score saved forecasts exactly as in docs/EVALUATION_PLAN.md (sections 5 and 6).

Usage:
    python scripts/evaluate_models.py --snapshot 2026-11-23 --period test \
        --forecast M0=data/forecasts/m0_naive/... --forecast M1=... [...]

For every model: MTU-level quantiles (with the common quarter-hour layer on days from
1 Oct 2025), scores per MTU, daily means, and breakdowns by regime, local hour, negative
prices and spikes; block-bootstrap intervals; Diebold–Mariano p-values for all pairs.
Writes results/evaluation/<period>.json and per-MTU scores to data/scores/<period>/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from powercast import dataset, evaluate, scores, stats
from powercast.config import DATA_DIR, QUARTER_HOUR_START_LOCAL, RESULTS_DIR
from powercast.quarterhour import QuarterHourLayer

PERIODS = {"validation": ("2019-01-01", "2020-12-31"), "test": ("2021-01-01", "2026-10-31")}
REGIMES = {"R0": ("2021-01-01", "2021-08-31"), "R1": ("2021-09-01", "2022-12-31"),
           "R2": ("2023-01-01", "2025-09-30"), "R3": ("2025-10-01", "2026-10-31")}
QH_DAY = QUARTER_HOUR_START_LOCAL.tz_localize(None).normalize()
BLOCKS = (7, 14, 28)


def spike_threshold(prices: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """99th percentile of all prices delivered in the 365 days before each day."""
    daily = dataset.to_slots(prices["price"])  # hourly means; enough for a threshold
    s = daily.stack()
    out = {}
    for d in days:
        win = s.loc[d - pd.Timedelta(days=365):d - pd.Timedelta(days=1)]
        out[d] = float(np.quantile(win.to_numpy(), 0.99))
    return pd.Series(out)


def hourly_mean_scores(prices, q, days, lo, hi) -> dict:
    """R3 scored on hourly means: slot quantiles (before the layer) vs slot mean prices."""
    sel = (days >= lo) & (days <= hi)
    d = days[sel]
    y = dataset.to_slots(prices["price"]).reindex(d).to_numpy()
    qq = evaluate.clip_to_limits(q[sel], d).reshape(-1, q.shape[2])
    yy = y.reshape(-1)
    ok = np.isfinite(yy)
    c = scores.crps(yy[ok], scores.sort_quantiles(qq[ok]))
    day_of = np.repeat(np.arange(len(d)), 24)[ok]
    daily = pd.Series(c).groupby(day_of).mean().to_numpy()
    mean, lo_, hi_ = stats.bootstrap_mean_ci(daily, block=14)
    return {"crps": {"mean": mean, "ci95": [lo_, hi_]},
            "cov80": float(scores.inside(yy[ok], qq[ok], 80).mean())}


def summarise(per_mtu: pd.DataFrame, seed: int = 0) -> dict:
    daily = per_mtu.groupby("day").mean(numeric_only=True)
    res = {"days": int(len(daily)), "mtus": int(len(per_mtu))}
    for col, key in [("crps", "crps"), ("ae", "mae")] + [(f"in{c}", f"cov{c}")
                                                         for c in scores.INTERVALS]:
        mean, lo, hi = stats.bootstrap_mean_ci(daily[col].to_numpy(), block=14, seed=seed)
        res[key] = {"mean": mean, "ci95": [lo, hi]}
    for c in scores.INTERVALS:
        res[f"width{c}"] = float(daily[f"w{c}"].mean())
    res["pit_hist"] = np.histogram(per_mtu["pit"], bins=20, range=(0, 1))[0].tolist()
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--period", choices=list(PERIODS), required=True)
    p.add_argument("--forecast", action="append", required=True, help="NAME=path.npz")
    a = p.parse_args()
    t = dataset.load(a.snapshot)
    days = pd.date_range(*PERIODS[a.period])
    thr = spike_threshold(t["prices"], days)
    layer = QuarterHourLayer(t["prices"], t["fund"]) if days[-1] >= QH_DAY else None

    per_model, slot_q = {}, {}
    for spec in a.forecast:
        name, path = spec.split("=", 1)
        fdays, q = evaluate.load_forecast(path)
        if not days.isin(fdays).all():
            raise ValueError(f"{name}: forecast does not cover the {a.period} period")
        q = q[fdays.get_indexer(days)]
        m, qm = evaluate.mtu_quantiles(t["prices"], t["fund"], days, q, layer)
        s = evaluate.daily_scores(m, qm)
        s["spike"] = s["y"].to_numpy() > thr.reindex(s["day"]).to_numpy()
        s["negative"] = s["y"] < 0
        s["mtu_min"] = m["mtu_min"].to_numpy()
        per_model[name] = s
        slot_q[name] = q
        out = DATA_DIR / "scores" / a.period
        out.mkdir(parents=True, exist_ok=True)
        s.to_parquet(out / f"{name}.parquet")

    report = {"snapshot": a.snapshot, "period": a.period, "models": {}, "dm": {}}
    for name, s in per_model.items():
        r = {"all": summarise(s)}
        if a.period == "test":
            for reg, (lo, hi) in REGIMES.items():
                r[reg] = summarise(s[(s["day"] >= lo) & (s["day"] <= hi)])
            r["R3_hourly_means"] = hourly_mean_scores(t["prices"], slot_q[name],
                                                      days, *REGIMES["R3"])
        for sub in ("negative", "spike"):
            r[sub] = summarise(s[s[sub]]) if s[sub].any() else None
        r["by_hour_crps"] = s.groupby("slot")["crps"].mean().round(4).tolist()
        r["by_hour_cov80"] = s.groupby("slot")["in80"].mean().round(4).tolist()
        r["crps_block_sensitivity"] = {
            b: stats.bootstrap_mean_ci(s.groupby("day")["crps"].mean().to_numpy(), block=b)[1:]
            for b in BLOCKS}
        report["models"][name] = r

    names = list(per_model)
    daily = {n: per_model[n].groupby("day")["crps"].mean() for n in names}
    segments = {"all": (days[0], days[-1])}
    if a.period == "test":
        segments.update({k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in REGIMES.items()})
    for seg, (lo, hi) in segments.items():
        mat = {}
        for x in names:
            for y in names:
                if x == y:
                    continue
                dx, dy = daily[x].loc[lo:hi], daily[y].loc[lo:hi]
                mat[f"{x}<{y}"] = stats.diebold_mariano(dx.to_numpy(), dy.to_numpy())[1]
        report["dm"][seg] = mat
    dest = RESULTS_DIR / "evaluation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{a.period}.json").write_text(json.dumps(report, indent=1))
    for n in names:
        r = report["models"][n]["all"]
        print(f"{n:12s} CRPS {r['crps']['mean']:.3f} [{r['crps']['ci95'][0]:.3f}, "
              f"{r['crps']['ci95'][1]:.3f}]  MAE {r['mae']['mean']:.3f}  "
              f"cov50/80/90/98 " + " ".join(f"{r[f'cov{c}']['mean']:.3f}"
                                            for c in scores.INTERVALS))


if __name__ == "__main__":
    main()
