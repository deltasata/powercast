"""Random search of LightGBM hyperparameters on the validation period (model M2).

Usage:
    python scripts/tune_gbt.py --snapshot 2026-11-23 --n 30 --parallel 4 --threads 15

Proxy protocol to fit the compute budget (docs/DECISIONS.md D7.1): refit every 28 days,
5 quantile levels, score = mean pinball loss over those levels and all validation MTUs.
Writes results/tuning/gbt_<snapshot>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from powercast import dataset, evaluate, features, guard, scores
from powercast.config import RESULTS_DIR
from powercast.models.gbt import GBTQuantile, _long

VAL = ("2019-01-01", "2020-12-31")
LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({
            "num_leaves": int(rng.choice([15, 31, 63, 127])),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
            "n_estimators": int(rng.choice([200, 400, 800])),
            "min_child_samples": int(rng.choice([20, 50, 100, 200])),
            "feature_fraction": float(rng.choice([0.5, 0.8, 1.0])),
            "lambda_l2": float(rng.choice([0.0, 1.0, 10.0])),
            "window": int(rng.choice([364, 728, 1092])),
        })
    return out


def evaluate_config(cfg, X, Y, prices, threads, refit=28):
    days = pd.date_range(*VAL)
    params = {k: v for k, v in cfg.items() if k != "window"}
    qs = []
    for i in range(0, len(days), refit):
        block = days[i:i + refit]
        d0 = block[0]
        train = pd.date_range(d0 - pd.Timedelta(days=cfg["window"]), d0 - pd.Timedelta(days=1))
        m = GBTQuantile(params, levels=LEVELS, threads=threads).fit(X, Y, train)
        z = np.column_stack([mm.predict(_long(X, block)) for mm in m.models])
        qs.append(m.ty.inverse(np.sort(z, axis=1)).reshape(len(block), 24, -1))
    q = np.concatenate(qs)
    mt = evaluate.mtu_table(prices, days)
    qm = evaluate.to_mtu(q, days, mt)
    loss = scores.pinball(mt["y"].to_numpy(), qm, LEVELS)
    daily = pd.Series(loss.mean(axis=1), index=mt["day"].to_numpy()).groupby(level=0).mean()
    return {**cfg, "pinball5": float(daily.mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--threads", type=int, default=15)
    a = p.parse_args()
    t = dataset.load(a.snapshot)
    days = pd.date_range("2015-01-12", t["prices"].index.max().tz_convert(None).normalize())
    X = features.build(t, days, features.default_exog_lags())
    Y = dataset.fill_from_last_week(dataset.to_slots(t["prices"]["price"])).reindex(days)
    cfgs = sample_configs(a.n)
    for w in sorted({c["window"] for c in cfgs}):
        guard.check_tabular(t, w, VAL[1])
    res = Parallel(n_jobs=a.parallel, verbose=10)(
        delayed(evaluate_config)(c, X, Y, t["prices"], a.threads) for c in cfgs)
    res = sorted(res, key=lambda r: r["pinball5"])
    out = RESULTS_DIR / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gbt_{a.snapshot}.json").write_text(json.dumps(res, indent=2))
    for r in res[:5]:
        print(r)


if __name__ == "__main__":
    main()
