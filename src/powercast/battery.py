"""Battery arbitrage on the day-ahead market (docs/EVALUATION_PLAN.md, section 7).

For one delivery day with n MTUs of length Δt hours and prices p_t (EUR/MWh):

    maximise   Σ_t p_t (d_t − c_t) Δt
    subject to 0 ≤ c_t, d_t ≤ P                          (charge / discharge power, MW)
               s_t = Σ_{u≤t} (η c_u − d_u / η) Δt         (stored energy, MWh)
               0 ≤ s_t ≤ E,  s_n = 0                     (empty at start and end of day)
               Σ_t d_t Δt / η ≤ E                        (at most one full cycle per day)

η = √(round-trip efficiency). The schedule is optimised on forecast prices and paid at
the real prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    energy_mwh: float = 2.0
    round_trip: float = 0.90

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.round_trip))


def schedule(prices: np.ndarray, dt_h: float, b: Battery = Battery()) -> np.ndarray:
    """Optimal net discharge (MW, negative = charging) for the given prices."""
    p = np.asarray(prices, float)
    n = len(p)
    eta = b.eta
    # x = [c_1..c_n, d_1..d_n]; minimise −profit
    cost = np.concatenate([p * dt_h, -p * dt_h])
    tri = np.tril(np.ones((n, n)))
    soc = np.hstack([eta * dt_h * tri, -dt_h / eta * tri])  # s_t as a linear map of x
    a_ub = np.vstack([soc, -soc, np.concatenate([np.zeros(n), np.full(n, dt_h / eta)])[None, :]])
    b_ub = np.concatenate([np.full(n, b.energy_mwh), np.zeros(n), [b.energy_mwh]])
    a_eq = soc[-1:, :]
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[0.0],
                  bounds=[(0, b.power_mw)] * (2 * n), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    c, d = res.x[:n], res.x[n:]
    return d - c


def profit(net_mw: np.ndarray, prices: np.ndarray, dt_h: float) -> float:
    """Revenue in EUR of a schedule settled at ``prices``."""
    return float(np.sum(np.asarray(net_mw) * np.asarray(prices) * dt_h))
