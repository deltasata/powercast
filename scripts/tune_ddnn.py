"""Random search of distributional-network hyperparameters on validation (model M3).

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/tune_ddnn.py --snapshot 2026-11-23 --dist jsu

Proxy protocol (docs/DECISIONS.md D8.1): refit every 28 days, one seed, score = mean
daily CRPS over the validation period. Writes results/tuning/ddnn_<dist>_<snapshot>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from powercast import dataset, evaluate, features, guard
from powercast.config import RESULTS_DIR
from powercast.models.ddnn import DDNN, Config

VAL = ("2019-01-01", "2020-12-31")


def sample_configs(n: int, dist: str, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    sizes = [32, 64, 128, 256, 512]
    return [{
        "dist": dist,
        "hidden": (int(rng.choice(sizes)), int(rng.choice(sizes))),
        "activation": str(rng.choice(["relu", "elu", "softplus", "tanh"])),
        "dropout": float(rng.choice([0.0, 0.1, 0.2, 0.3])),
        "lr": float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-2)))),
        "l1": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))),
        "window": int(rng.choice([728, 1092])),
    } for _ in range(n)]


def run(cfg: dict, X, Y, prices, refit: int = 28, seed: int = 0):
    torch.set_num_threads(1)
    days = pd.date_range(*VAL)
    c = Config(**{k: v for k, v in cfg.items() if k != "window"})
    qs = []
    for i in range(0, len(days), refit):
        block = days[i:i + refit]
        d0 = block[0]
        train = pd.date_range(d0 - pd.Timedelta(days=cfg["window"]), d0 - pd.Timedelta(days=1))
        qs.append(DDNN(c, seed=seed).fit(X, Y, train).predict(X, block))
    q = np.concatenate(qs)
    return {**cfg, "hidden": list(cfg["hidden"]), **evaluate.summary(prices, days, q)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dist", choices=["normal", "jsu"], required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--jobs", type=int, default=30)
    a = p.parse_args()
    t = dataset.load(a.snapshot)
    days = pd.date_range("2015-01-12", t["prices"].index.max().tz_convert(None).normalize())
    X = features.build(t, days, features.default_exog_lags())
    Y = dataset.fill_from_last_week(dataset.to_slots(t["prices"]["price"])).reindex(days)
    cfgs = sample_configs(a.n, a.dist)
    for w in sorted({c["window"] for c in cfgs}):
        guard.check_tabular(t, w, VAL[1])
    res = Parallel(n_jobs=a.jobs, verbose=10)(
        delayed(run)(c, X, Y, t["prices"]) for c in cfgs)
    res = sorted(res, key=lambda r: r["crps"])
    out = RESULTS_DIR / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ddnn_{a.dist}_{a.snapshot}.json").write_text(json.dumps(res, indent=2))
    for r in res[:5]:
        print(r)


if __name__ == "__main__":
    main()
