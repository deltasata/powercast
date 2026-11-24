"""Rolling forecasts for models M2 (LightGBM), M3 (distributional NN), M4 (Chronos-2).

Usage:
    python scripts/run_model.py --model gbt    --config results/tuning/gbt_final.json \
        --snapshot 2026-11-23 --start 2019-01-01 --end 2020-12-31
    python scripts/run_model.py --model ddnn   --config results/tuning/ddnn_jsu_final.json \
        --seeds 0 1 2 3 4 ...
    python scripts/run_model.py --model chronos --variant covariates ...

Refit every 7 days (EVALUATION_PLAN section 3) on the configured window ending the day
before the first day of each block. The no-peeking check runs first, on exactly the data
handed to the model (training rows, training targets and target-day inputs): a
deterministic model given identical data gives an identical forecast.
Output: data/forecasts/<model>/<tag>_<start>_<end>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from powercast import dataset, evaluate, features, guard, nopeek
from powercast.config import DATA_DIR

REFIT = 7


def chronos_inputs(tables, day):
    from powercast.models.chronos2 import CONTEXT
    P = dataset.fill_from_last_week(dataset.to_slots(tables["prices"]["price"]))
    ex = features.exog_slots(tables["fund"])
    first = day - pd.Timedelta(days=CONTEXT // 24 + 1)
    hist_days = pd.date_range(first, day - pd.Timedelta(days=1))
    return np.concatenate([P.reindex(hist_days).to_numpy().ravel(),
                           ex["res"].reindex(hist_days).to_numpy().ravel(),
                           ex["load"].reindex(hist_days.append(pd.DatetimeIndex([day])))
                           .to_numpy().ravel()])


def run_block_gbt(cfg, X, Y, block, threads):
    from powercast.models.gbt import GBTQuantile
    params = {k: v for k, v in cfg.items() if k not in ("window", "pinball5")}
    d0 = block[0]
    train = pd.date_range(d0 - pd.Timedelta(days=cfg["window"]), d0 - pd.Timedelta(days=1))
    return GBTQuantile(params, threads=threads).fit(X, Y, train).predict(X, block)


def run_block_ddnn(cfg, X, Y, block, seed):
    import torch

    from powercast.models.ddnn import DDNN, Config
    torch.set_num_threads(1)
    keep = {k: v for k, v in cfg.items() if k in Config.__dataclass_fields__}
    keep["hidden"] = tuple(keep["hidden"])
    d0 = block[0]
    train = pd.date_range(d0 - pd.Timedelta(days=cfg["window"]), d0 - pd.Timedelta(days=1))
    return DDNN(Config(**keep), seed=seed).fit(X, Y, train).predict(X, block)


def run_chunk_chronos(variant, P, ex, chunk, threads):
    import torch

    from powercast.models.chronos2 import Chronos2
    torch.set_num_threads(threads)
    return Chronos2(variant).forecast(P, ex, chunk)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gbt", "ddnn", "chronos"], required=True)
    p.add_argument("--config")
    p.add_argument("--variant", default="covariates")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    t = dataset.load(a.snapshot)
    tables = {"prices": t["prices"][["price"]], "fund": t["fund"]}
    lags = features.default_exog_lags()
    cfg = json.loads(open(a.config).read()) if a.config else {}
    if a.model == "chronos":
        check_days = guard.check_days(a.end)
        t0 = time.time()
        nopeek.assert_no_peeking(chronos_inputs, tables, check_days)
        print(f"no-peeking check passed on {len(check_days)} days ({time.time() - t0:.0f} s)",
              flush=True)
    else:
        guard.check_tabular(t, cfg["window"], a.end)

    days = pd.date_range(a.start, a.end)
    blocks = [days[i:i + REFIT] for i in range(0, len(days), REFIT)]
    all_days = pd.date_range("2015-01-12", days[-1])
    X = features.build(t, all_days, lags)
    Y = dataset.fill_from_last_week(dataset.to_slots(t["prices"]["price"])).reindex(all_days)
    out_dir = DATA_DIR / "forecasts" / a.model
    t0 = time.time()
    if a.model == "gbt":
        res = Parallel(n_jobs=a.jobs, verbose=5)(
            delayed(run_block_gbt)(cfg, X, Y, b, a.threads) for b in blocks)
        q = np.concatenate(res)
        name = f"{a.tag or 'gbt'}_{a.start}_{a.end}.npz"
        evaluate.save_forecast(out_dir / name, days, q, config=cfg)
        print(evaluate.summary_if_validation(t["prices"], days, q), flush=True)
    elif a.model == "ddnn":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        tasks = [(s, b) for s in a.seeds for b in blocks]
        res = Parallel(n_jobs=a.jobs, verbose=5)(
            delayed(run_block_ddnn)(cfg, X, Y, b, s) for s, b in tasks)
        per_seed = {s: [] for s in a.seeds}
        for (s, _), r in zip(tasks, res, strict=True):
            per_seed[s].append(r)
        for s in a.seeds:
            q = np.concatenate(per_seed[s])
            name = f"{a.tag or cfg['dist']}_seed{s}_{a.start}_{a.end}.npz"
            evaluate.save_forecast(out_dir / name, days, q, config=cfg, seed=s)
            print("seed", s, evaluate.summary_if_validation(t["prices"], days, q), flush=True)
    else:
        P = dataset.fill_from_last_week(dataset.to_slots(t["prices"]["price"]))
        ex = features.exog_slots(t["fund"])
        chunks = [days[i:i + 64] for i in range(0, len(days), 64)]
        res = Parallel(n_jobs=a.jobs, verbose=5)(
            delayed(run_chunk_chronos)(a.variant, P, ex, c, a.threads) for c in chunks)
        q = np.concatenate(res)
        evaluate.save_forecast(out_dir / f"{a.tag or a.variant}_{a.start}_{a.end}.npz", days, q,
                               variant=a.variant)
        print(evaluate.summary_if_validation(t["prices"], days, q), flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
