"""Daily incremental update of the processed tables (used by the daily run).

Downloads only the last few weekly SMARD files of every series into a dated partial
snapshot, rebuilds those weeks and overlays them on the previous processed tables:
newer values replace older ones (SMARD revises recent measured values).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from powercast.data import build, smard


def update(prev_dir: Path, raw_root: Path, out_dir: Path, weeks: int = 3,
           ec_price: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    for s in smard.SERIES:
        smard.download(s, raw_root, last=weeks)
    new_p = build.build_prices(raw_root)
    if ec_price is not None:
        new_p = build.fill_prices_from_ec(new_p, ec_price)
    new_f = build.build_fundamentals(raw_root)
    old_p = pd.read_parquet(prev_dir / "prices.parquet")
    old_f = pd.read_parquet(prev_dir / "fundamentals.parquet")
    prices = new_p.dropna(subset=["price"]).combine_first(old_p).sort_index()
    fund = new_f.combine_first(old_f).sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_dir / "prices.parquet")
    fund.to_parquet(out_dir / "fundamentals.parquet")
    return {"prices": prices, "fund": fund}
