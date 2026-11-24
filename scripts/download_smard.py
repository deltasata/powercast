"""Download all SMARD series used by the project into a dated raw snapshot.

Usage:
    python scripts/download_smard.py [--date YYYY-MM-DD]

Writes data/raw/smard/<date>/<filter>_<region>_<resolution>/*.json and a
manifest.json with file counts, bytes and covered period per series.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from powercast.config import MARKET_TZ, RAW_DIR
from powercast.data import smard


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=dt.datetime.now(MARKET_TZ).date().isoformat())
    a = p.parse_args()
    root = RAW_DIR / "smard" / a.date
    manifest = {"source": "Bundesnetzagentur | SMARD.de (CC BY 4.0)",
                "base_url": smard.BASE, "snapshot_date": a.date, "series": {}}
    for s in smard.SERIES:
        folder = smard.download(s, root)
        files = list(folder.glob("*.json"))
        ser = smard.read(folder).dropna()
        manifest["series"][s.key] = {
            "name": s.name,
            "unit": s.unit,
            "files": len(files),
            "bytes": sum(f.stat().st_size for f in files),
            "first_utc": str(ser.index.min()),
            "last_utc": str(ser.index.max()),
            "n_values": int(ser.size),
        }
        print(s.key, manifest["series"][s.key], flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
