"""No-peeking check on exactly the data a model sees, run before every model run.

The inputs of one refit for delivery day D are the feature rows of the training days and
of D, plus the training targets. If data published after the issue time of D is replaced
by noise and these inputs do not change, a model given them cannot have peeked.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from powercast import dataset, features, nopeek

# both daylight-saving days of 2019, the missing load-forecast day (31 Jan 2020), the gas
# crisis, and the 25-hour day after the switch to quarter-hour prices
CHECK_DAYS = ["2019-03-31", "2019-10-27", "2020-02-01", "2021-09-15", "2025-10-26"]


def check_days(end) -> list[pd.Timestamp]:
    """Fixed check days up to ``end``; the last day of the run if there are none."""
    days = [pd.Timestamp(d) for d in CHECK_DAYS if pd.Timestamp(d) <= pd.Timestamp(end)]
    return days or [pd.Timestamp(end)]


def model_inputs(tables, day, window, lags):
    """Everything a refit for ``day`` sees: X of train days + day, Y of train days."""
    days = pd.date_range("2015-01-12", day)
    X = features.build(tables, days, lags)
    Y = dataset.fill_from_last_week(dataset.to_slots(tables["prices"]["price"])).reindex(days)
    train = pd.date_range(day - pd.Timedelta(days=window), day - pd.Timedelta(days=1))
    return np.concatenate([X.reindex(train.append(pd.DatetimeIndex([day]))).to_numpy().ravel(),
                           Y.reindex(train).to_numpy().ravel()])


def check_tabular(t: dict, window: int, end: str) -> None:
    """Run the check for a feature-based model (LightGBM, network) with this window."""
    tables = {"prices": t["prices"][["price"]], "fund": t["fund"]}
    lags = features.default_exog_lags()
    days = check_days(end)
    t0 = time.time()
    nopeek.assert_no_peeking(lambda tb, d: model_inputs(tb, d, window, lags), tables, days)
    print(f"no-peeking check passed on {len(days)} days, window {window} "
          f"({time.time() - t0:.0f} s)", flush=True)
