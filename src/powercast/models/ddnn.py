"""M3: distributional neural network (Marcjasz, Narajewski, Weron, Ziel 2023).

A multilayer perceptron maps the daily feature vector (powercast.features) to the
parameters of 24 hourly predictive distributions of the transformed price:

* normal:          N(μ, σ),                        σ = softplus(·)
* Johnson's SU:    X = ξ + λ sinh((Z − γ) / δ),    Z ~ N(0, 1); δ, λ = softplus(·)

Training minimises the mean negative log-likelihood with Adam; early stopping on a
random 20 % of the training days. Inputs and target use the asinh transform fitted on
the training window (powercast.transforms); quantiles are back-transformed exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from powercast.scores import TAUS
from powercast.transforms import Asinh

ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU, "softplus": nn.Softplus, "tanh": nn.Tanh}
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


@dataclass
class Config:
    dist: str = "jsu"  # "normal" or "jsu"
    hidden: tuple[int, int] = (256, 128)
    activation: str = "elu"
    dropout: float = 0.1
    lr: float = 1e-3
    l1: float = 1e-4  # L1 penalty on the first layer's weights
    batch: int = 32
    max_epochs: int = 1500
    patience: int = 50
    extra: dict = field(default_factory=dict)

    @property
    def n_par(self) -> int:
        return 2 if self.dist == "normal" else 4


class Net(nn.Module):
    def __init__(self, n_in: int, cfg: Config):
        super().__init__()
        act = ACTIVATIONS[cfg.activation]
        self.first = nn.Linear(n_in, cfg.hidden[0])
        self.body = nn.Sequential(act(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden[0], cfg.hidden[1]), act(),
                                  nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden[1], 24 * cfg.n_par))
        self.n_par = cfg.n_par

    def forward(self, x):
        return self.body(self.first(x)).view(-1, 24, self.n_par)


def nll(out: torch.Tensor, y: torch.Tensor, dist: str) -> torch.Tensor:
    if dist == "normal":
        mu, sigma = out[..., 0], F.softplus(out[..., 1]) + 1e-3
        z = (y - mu) / sigma
        return (torch.log(sigma) + LOG_SQRT_2PI + 0.5 * z**2).mean()
    gamma, delta = out[..., 0], F.softplus(out[..., 1]) + 1e-3
    xi, lam = out[..., 2], F.softplus(out[..., 3]) + 1e-3
    u = (y - xi) / lam
    z = gamma + delta * torch.asinh(u)
    logpdf = (torch.log(delta) - torch.log(lam) - LOG_SQRT_2PI
              - 0.5 * torch.log1p(u**2) - 0.5 * z**2)
    return -logpdf.mean()


def quantiles(out: np.ndarray, dist: str, taus: np.ndarray = TAUS) -> np.ndarray:
    """(n, 24, n_par) raw outputs → quantiles (n, 24, K) of the transformed price."""
    from scipy.stats import norm

    zt = norm.ppf(taus)
    sp = lambda v: np.logaddexp(0, v) + 1e-3  # noqa: E731  softplus
    if dist == "normal":
        mu, sigma = out[..., 0], sp(out[..., 1])
        return mu[..., None] + sigma[..., None] * zt
    gamma, delta, xi, lam = out[..., 0], sp(out[..., 1]), out[..., 2], sp(out[..., 3])
    return xi[..., None] + lam[..., None] * np.sinh((zt - gamma[..., None]) / delta[..., None])


class DDNN:
    def __init__(self, cfg: Config, seed: int = 0, device: str = "cpu"):
        self.cfg, self.seed, self.device = cfg, seed, device

    def _prep_x(self, Xd: np.ndarray) -> np.ndarray:
        return np.hstack([self.tx.transform(Xd[:, ~self.dummies]), Xd[:, self.dummies]])

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, train_days: pd.DatetimeIndex) -> DDNN:
        Xt, Yt = X.reindex(train_days), Y.reindex(train_days)
        ok = (Xt.notna().all(axis=1) & Yt.notna().all(axis=1)).to_numpy()
        Xd, Yd = Xt.to_numpy(float)[ok], Yt.to_numpy(float)[ok]
        self.dummies = np.array([c.startswith("dow_") for c in X.columns])
        self.tx = Asinh().fit(Xd[:, ~self.dummies])
        self.ty = Asinh().fit(Yd.reshape(-1))
        x = torch.tensor(self._prep_x(Xd), dtype=torch.float32, device=self.device)
        y = torch.tensor(self.ty.transform(Yd), dtype=torch.float32, device=self.device)

        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        n = len(x)
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(0.2 * n))
        iv, it = perm[:n_val], perm[n_val:]
        self.net = Net(x.shape[1], self.cfg).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        best, best_state, wait = float("inf"), None, 0
        for _ in range(self.cfg.max_epochs):
            self.net.train()
            order = it[torch.randperm(len(it), generator=g)]
            for k in range(0, len(order), self.cfg.batch):
                b = order[k:k + self.cfg.batch]
                loss = nll(self.net(x[b]), y[b], self.cfg.dist)
                loss = loss + self.cfg.l1 * self.net.first.weight.abs().sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(nll(self.net(x[iv]), y[iv], self.cfg.dist))
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break
        self.net.load_state_dict(best_state)
        self.val_nll = best
        return self

    def predict(self, X: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
        """Quantiles (len(days), 24, 99) in EUR/MWh."""
        x = torch.tensor(self._prep_x(X.reindex(days).to_numpy(float)), dtype=torch.float32,
                         device=self.device)
        self.net.eval()
        with torch.no_grad():
            out = self.net(x).cpu().numpy()
        z = quantiles(out, self.cfg.dist)
        return np.sort(self.ty.inverse(z), axis=2)
