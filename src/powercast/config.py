"""Project-wide constants: paths, time zones and market facts.

All timestamps inside the project are stored in UTC. Europe/Berlin is used
only to define market rules (the 12:00 auction deadline, delivery days).
"""

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # dated, unmodified copies of every download
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned tables built from raw/
RESULTS_DIR = ROOT / "results"  # saved forecasts and scores (small, committed)
FIGURES_DIR = ROOT / "figures"

# Time
UTC = ZoneInfo("UTC")
MARKET_TZ = ZoneInfo("Europe/Berlin")

# Market facts
BIDDING_ZONE = "DE-LU"
# Day-ahead gate closure: bids for delivery day D+1 close at 12:00 Berlin time on day D.
GATE_CLOSURE_LOCAL = time(12, 0)
# Day-ahead market time unit: 60 min before, 15 min from this delivery day on (Berlin time).
QUARTER_HOUR_START_LOCAL = pd.Timestamp("2025-10-01 00:00", tz=MARKET_TZ)
