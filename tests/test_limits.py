import numpy as np
import pandas as pd

from powercast import evaluate, scores


def test_clip_to_limits_by_day():
    days = pd.DatetimeIndex(["2022-05-09", "2022-05-10"])
    q = np.array([[-np.inf, 0.0, 3500.0], [-600.0, 10.0, np.inf]])
    c = evaluate.clip_to_limits(q, days)
    assert c.tolist() == [[-500.0, 0.0, 3000.0], [-500.0, 10.0, 4000.0]]


def test_clipping_into_the_support_never_raises_pinball_loss():
    rng = np.random.default_rng(0)
    y = rng.uniform(-500, 3000, 500)
    q = np.sort(rng.normal(0, 3000, (500, 99)), axis=1)
    days = pd.DatetimeIndex(["2021-01-01"] * 500)
    before = scores.crps(y, q)
    after = scores.crps(y, evaluate.clip_to_limits(q, days))
    assert np.all(after <= before + 1e-9)
