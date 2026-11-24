"""Daily run: forecast tomorrow's DE-LU day-ahead prices before the 12:00 auction.

Usage (run each morning, e.g. by .github/workflows/daily.yml):
    python scripts/daily_forecast.py [--state data/live] [--out forecasts/live]

1. Update the processed tables in --state (full download on the first run, otherwise
   only the last three weekly SMARD files; missing prices filled from Energy-Charts).
2. Apply the publication rules for the issue time (D−1 11:00 Berlin): the live forecast
   sees exactly the information set of the backtest.
3. Forecast with the live models (config/live.json) and the common quarter-hour layer.
4. Write forecasts/live/<D>.csv (selected quantiles) and <D>.npz (all 99 levels), then
   score every earlier live forecast whose prices are known: forecasts/live/track_record.csv.

A run after 12:00 Berlin is marked late=true in the output and in the track record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from powercast import dataset, evaluate, features, nopeek, scores
from powercast.config import MARKET_TZ, RAW_DIR, ROOT
from powercast.data import build, energycharts, smard, update
from powercast.models import lear, naive
from powercast.quarterhour import QuarterHourLayer

SEED_POINTS = ROOT / "config" / "seed" / "lear_points.parquet"
SHOW = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def fetch_ec(ec_dir: Path, today: str) -> dict[str, pd.Series]:
    """Last three weeks of Energy-Charts prices and day-ahead forecasts (CC BY 4.0)."""
    import time

    import requests

    start = (pd.Timestamp(today) - pd.Timedelta(days=21)).date().isoformat()
    end = (pd.Timestamp(today) + pd.Timedelta(days=2)).date().isoformat()
    jobs = [("price", {"bzn": "DE-LU"}, f"price_DE-LU_{today}.json")]
    jobs += [("public_power_forecast", {"country": "de", "production_type": pt,
                                        "forecast_type": "day-ahead"},
              f"forecast_day-ahead_{pt}_de_{today}.json")
             for pt in ("load", "solar", "wind_onshore", "wind_offshore")]
    for endpoint, params, name in jobs:
        for k in range(4):
            r = requests.get(f"https://api.energy-charts.info/{endpoint}",
                             params={**params, "start": start, "end": end}, timeout=120)
            if r.status_code != 429:
                break
            time.sleep(5 * 2 ** k)
        if r.ok:
            (ec_dir / name).write_bytes(r.content)
        else:
            print(f"Energy-Charts {name}: HTTP {r.status_code}", flush=True)
        time.sleep(2)
    return energycharts.load(ec_dir)


def refresh(state: Path, today: str) -> dict[str, pd.DataFrame]:
    proc = state / "processed"
    raw = RAW_DIR / "smard" / today
    ec_dir = RAW_DIR / "energy-charts" / today
    ec_dir.mkdir(parents=True, exist_ok=True)
    ec = fetch_ec(ec_dir, today)
    if not (proc / "prices.parquet").exists():
        for s in smard.SERIES:
            smard.download(s, raw)
        prices = build.build_prices(raw)
        fund = build.build_fundamentals(raw)
        proc.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(proc / "prices.parquet")
        fund.to_parquet(proc / "fundamentals.parquet")
        tables = {"prices": prices, "fund": fund}
    else:
        tables = update.update(proc, raw, proc, weeks=3)
    # the stored tables keep SMARD values only; Energy-Charts values are added for this
    # run, so later SMARD postings are always used once they exist
    prices, fund, n = build.extend_from_ec(tables["prices"], tables["fund"], ec)
    print("filled from Energy-Charts:", n, flush=True)
    return {"prices": prices, "fund": fund}


def info_set(tables, day):
    cut = nopeek.issue_time(day)
    return {"prices": nopeek.mask_unpublished(tables["prices"][["price", "mtu_min"]], cut),
            "fund": nopeek.mask_unpublished(tables["fund"], cut)}


def forecast_m0(tb, day, cfg):
    P = dataset.to_slots(tb["prices"]["price"])
    return naive.forecast(P, pd.DatetimeIndex([day]), cfg["window"])[0]


def forecast_m1(tb, day, cfg, state: Path):
    """LEAR-QRA; LEAR point forecasts of past days are cached in the state folder."""
    lags = features.default_exog_lags()
    days = pd.date_range("2015-01-12", day)
    X = features.build(tb, days, lags)
    Y = dataset.fill_from_last_week(dataset.to_slots(tb["prices"]["price"])).reindex(days)
    f = state / "lear_points.parquet"
    if not f.exists() and SEED_POINTS.exists():
        # first run: LEAR forecasts of past days from the backtest (same model, same rules)
        pd.read_parquet(SEED_POINTS).to_parquet(f)
    pts = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    need = pd.date_range(day - pd.Timedelta(days=cfg["window"]), day)
    todo = need.difference(pts.index)
    # a past day's LEAR forecast uses only data before that day (same as the backtest)
    new = Parallel(n_jobs=-1)(
        delayed(lambda d: np.column_stack([lear.lear_day(X, Y, d, w)
                                           for w in lear.WINDOWS]).ravel())(d) for d in todo)
    if len(todo):
        cols = pts.columns if len(pts.columns) else None
        pts = pd.concat([pts, pd.DataFrame(np.array(new), index=todo, columns=cols)])
    pts = pts.sort_index()
    pts.columns = [str(c) for c in pts.columns]
    pts.to_parquet(f)
    P = pts.loc[need].to_numpy().reshape(len(need), 24, len(lear.WINDOWS))
    return lear.qra_day(P[:-1], Y.loc[need[:-1]].to_numpy(), P[-1])


def forecast_m4(tb, day, cfg):
    """Chronos-2, zero-shot: price history to D−1, load forecast for D, wind + solar
    forecasts to D−1 (the same inputs as the backtest, D9.1–D9.2)."""
    from powercast.models.chronos2 import Chronos2

    P = dataset.fill_from_last_week(dataset.to_slots(tb["prices"]["price"]))
    ex = features.exog_slots(tb["fund"])
    model = Chronos2(cfg.get("variant", "covariates"))
    return model.forecast(P, ex, pd.DatetimeIndex([day]))[0]


def score_track(out: Path, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    known_until = prices["price"].last_valid_index()
    for f in sorted(out.glob("20*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        day = pd.Timestamp(meta["delivery_day"])
        end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
        if end - pd.Timedelta(minutes=15) > known_until:
            continue
        y = prices["price"].reindex(pd.DatetimeIndex(z["mtu_start_utc"].astype("datetime64[ns]"),
                                                     tz="UTC")).to_numpy()
        for name in meta["models"]:
            q = z[name]
            ok = np.isfinite(y)
            rows.append({"delivery_day": day.date().isoformat(), "model": name,
                         "late": meta["late"], "backfill": meta.get("backfill", False),
                         "crps": float(scores.crps(y[ok], q[ok]).mean()),
                         "mae": float(np.abs(y[ok] - scores.median(q[ok])).mean()),
                         "cov80": float(scores.inside(y[ok], q[ok], 80).mean()),
                         "cov90": float(scores.inside(y[ok], q[ok], 90).mean())})
    tr = pd.DataFrame(rows)
    if len(tr):
        tr.to_csv(out / "track_record.csv", index=False)
    return tr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", default=str(ROOT / "data" / "live"))
    p.add_argument("--out", default=str(ROOT / "forecasts" / "live"))
    p.add_argument("--config", default=str(ROOT / "config" / "live.json"))
    p.add_argument("--force", action="store_true", help="overwrite an existing forecast")
    p.add_argument("--day", help="delivery day to forecast instead of tomorrow (tests and "
                                 "backfills; marked backfill=true)")
    a = p.parse_args()
    state, out = Path(a.state), Path(a.out)
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())

    now = dt.datetime.now(MARKET_TZ)
    day = pd.Timestamp(a.day) if a.day else pd.Timestamp(now.date() + dt.timedelta(days=1))
    if (out / f"{day.date()}.npz").exists() and not a.force:
        print(f"forecast for {day.date()} exists; nothing to do")
        return
    if now < (day - pd.Timedelta(hours=13, minutes=45)).tz_localize(MARKET_TZ) and not a.force:
        print("before 10:15 Berlin: the load forecast may not be published yet; exiting")
        return
    late = now >= (day - pd.Timedelta(hours=12)).tz_localize(MARKET_TZ)

    tables = refresh(state, now.date().isoformat())
    tb = info_set(tables, day)
    slot_q = {}
    if "M0" in cfg["models"]:
        slot_q["M0"] = forecast_m0(tb, day, cfg["models"]["M0"])
    if "M1" in cfg["models"]:
        slot_q["M1"] = forecast_m1(tb, day, cfg["models"]["M1"], state)
    if "M4" in cfg["models"]:
        slot_q["M4"] = forecast_m4(tb, day, cfg["models"]["M4"])

    # MTUs of the delivery day (quarter-hours), mapped to slots, then the common layer
    start = day.tz_localize(MARKET_TZ).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
    idx = pd.date_range(start, end, freq="15min", inclusive="left")
    target = pd.DataFrame({"price": np.nan, "mtu_min": 15}, index=idx)
    prices_ext = pd.concat([tb["prices"].loc[tb["prices"].index < start], target])
    loc = idx.tz_convert(MARKET_TZ)
    mtus = pd.DataFrame({"day": loc.normalize().tz_localize(None), "slot": loc.hour}, index=idx)
    layer = QuarterHourLayer(prices_ext, tb["fund"])
    mtu_q = {}
    for name, q in slot_q.items():
        qm = evaluate.to_mtu(q[None], pd.DatetimeIndex([day]), mtus)
        mtu_q[name] = scores.sort_quantiles(evaluate.clip_to_limits(
            layer.apply(day, mtus, qm, seed=int(day.strftime("%Y%m%d"))), mtus["day"]))

    meta = {"delivery_day": day.date().isoformat(), "issued_berlin": now.isoformat(),
            "late": bool(late), "backfill": bool(a.day), "models": list(mtu_q),
            "primary": cfg["primary"],
            "information_set": f"published by {nopeek.issue_time(day).isoformat()}"}
    np.savez_compressed(out / f"{day.date()}.npz", meta=np.array(json.dumps(meta)),
                        mtu_start_utc=np.asarray(idx.tz_convert(None).astype("datetime64[s]")),
                        **mtu_q)
    j = [list(scores.TAUS).index(t) for t in [round(x, 2) for x in SHOW]]
    table = pd.DataFrame({"start_utc": idx.strftime("%Y-%m-%dT%H:%MZ"),
                          "start_berlin": loc.strftime("%Y-%m-%d %H:%M")})
    for t, k in zip(SHOW, j, strict=True):
        table[f"q{int(round(t * 100)):02d}"] = mtu_q[cfg["primary"]][:, k].round(2)
    table.to_csv(out / f"{day.date()}.csv", index=False)
    tr = score_track(out, tables["prices"])
    print(json.dumps(meta))
    if len(tr):
        print(tr.groupby("model")[["crps", "mae", "cov80", "cov90"]].mean().round(3))


if __name__ == "__main__":
    main()
