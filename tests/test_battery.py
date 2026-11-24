"""Battery LP: hand-checkable cases."""

import numpy as np

from powercast import battery


def test_two_price_day_one_cycle():
    # 12 cheap hours at 0, 12 expensive at 100: store E = 2 MWh, deliver 2η MWh at 100
    p = np.array([0.0] * 12 + [100.0] * 12)
    net = battery.schedule(p, 1.0)
    assert abs(battery.profit(net, p, 1.0) - 200.0 * np.sqrt(0.9)) < 1e-6
    assert np.all(net[:12] <= 1e-9) and np.all(net[12:] >= -1e-9)


def test_flat_prices_do_nothing():
    p = np.full(24, 50.0)
    net = battery.schedule(p, 1.0)
    assert np.allclose(net, 0, atol=1e-9)


def test_efficiency_loss():
    # buy at 40, sell at 50: store E = 2 MWh; buy E/η = 2.108 MWh, deliver Eη = 1.897 MWh
    p = np.array([40.0, 40.0, 40.0, 50.0, 50.0, 50.0])
    net = battery.schedule(p, 1.0)
    eta = np.sqrt(0.9)
    assert abs(battery.profit(net, p, 1.0) - (2 * eta * 50 - 2 / eta * 40)) < 1e-6


def test_at_most_one_cycle():
    # two price valleys and peaks: without the cycle limit it would cycle twice
    p = np.array([0.0] * 6 + [100.0] * 6 + [0.0] * 6 + [100.0] * 6)
    net = battery.schedule(p, 1.0)
    assert abs(battery.profit(net, p, 1.0) - 200.0 * np.sqrt(0.9)) < 1e-6


def test_quarter_hours():
    p = np.array([0.0] * 48 + [100.0] * 48)
    net = battery.schedule(p, 0.25)
    assert abs(battery.profit(net, p, 0.25) - 200.0 * np.sqrt(0.9)) < 1e-6
