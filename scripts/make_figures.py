"""Figures for the README from saved results (no model is run here).

Usage:
    python scripts/make_figures.py --period test

Reads results/evaluation/<period>.json, results/battery/<period>.json and
writes figures/<period>_*.png:
  crps_by_period   CRPS with 95 % block-bootstrap interval, one panel per market period
  coverage         empirical vs nominal coverage of the central intervals
  pit              PIT histograms (flat = calibrated)
  crps_by_hour     CRPS by local hour of delivery
  battery          battery profit per forecast, with the perfect-foresight bound
"""

from __future__ import annotations

import argparse
import json
import os

from powercast.config import FIGURES_DIR, RESULTS_DIR, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# fixed categorical order (one colour per model, never reassigned)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEGMENT_NAMES = {"all": "Whole period", "R0": "R0 Jan–Aug 2021", "R1": "R1 gas crisis 2021–22",
                 "R2": "R2 2023–Sep 2025", "R3": "R3 quarter-hour, from Oct 2025"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False,
    "text.color": INK, "axes.titlesize": 9, "axes.titlecolor": INK})


def colour_map(names):
    return {n: COLOURS[i % len(COLOURS)] for i, n in enumerate(names)}


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES_DIR / name)


def crps_by_period(ev, period, col):
    models = list(ev["models"])
    segs = [s for s in ["all", "R0", "R1", "R2", "R3"] if s in ev["models"][models[0]]]
    fig, axes = plt.subplots(1, len(segs), figsize=(2.6 * len(segs), 0.45 * len(models) + 1.2),
                             sharey=True, squeeze=False)
    for ax, seg in zip(axes[0], segs, strict=True):
        for i, m in enumerate(models):
            r = ev["models"][m][seg]["crps"]
            lo, hi = r["ci95"]
            ax.errorbar(r["mean"], i, xerr=[[r["mean"] - lo], [hi - r["mean"]]], fmt="o",
                        ms=5, color=col[m], ecolor=col[m], elinewidth=2, capsize=0)
        ax.set_title(SEGMENT_NAMES[seg])
        ax.set_xlabel("CRPS (EUR/MWh)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(models)), models)
    axes[0][0].set_ylim(len(models) - 0.5, -0.5)
    fig.suptitle("CRPS per model, 95 % block-bootstrap interval (lower is better)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if len(models) > 3 else 0.9))
    save(fig, f"{period}_crps_by_period.png")


def coverage(ev, period, col):
    models = list(ev["models"])
    levels = [50, 80, 90, 98]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    off = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else [0.0]
    for k, m in enumerate(models):
        r = ev["models"][m]["all"]
        mean = [100 * r[f"cov{c}"]["mean"] for c in levels]
        lo = [100 * r[f"cov{c}"]["ci95"][0] for c in levels]
        hi = [100 * r[f"cov{c}"]["ci95"][1] for c in levels]
        x = np.arange(len(levels)) + off[k]
        ax.errorbar(x, np.array(mean) - levels,
                    yerr=[np.array(mean) - lo, np.array(hi) - mean], fmt="o", ms=5,
                    color=col[m], elinewidth=2, capsize=0, label=m)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(levels)), [f"{c} %" for c in levels])
    ax.set_xlabel("Nominal coverage of the central interval")
    ax.set_ylabel("Empirical − nominal (percentage points)")
    ax.set_title("Interval coverage: 0 = calibrated, below 0 = intervals too narrow",
                 loc="left")
    ax.legend(ncol=min(6, len(models)), loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8)
    save(fig, f"{period}_coverage.png")


def pit(ev, period, col):
    models = list(ev["models"])
    n = len(models)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 1.9 * rows), sharey=True,
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, m in zip(axes.ravel(), models, strict=False):
        h = np.array(ev["models"][m]["all"]["pit_hist"], float)
        dens = h / h.sum() * len(h)
        ax.bar((np.arange(len(h)) + 0.5) / len(h), dens, width=1 / len(h) - 0.008,
               color=col[m])
        ax.axhline(1, color=INK2, lw=1)
        ax.set_title(m)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.suptitle("PIT histograms (flat at 1 = calibrated; U shape = intervals too narrow)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, f"{period}_pit.png")


def crps_by_hour(ev, period, col):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m, r in ev["models"].items():
        y = r["by_hour_crps"]
        ax.plot(range(len(y)), y, color=col[m], lw=2, label=m)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Local hour of delivery (Europe/Berlin)")
    ax.set_ylabel("CRPS (EUR/MWh)")
    ax.set_title("CRPS by hour of the day", loc="left")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    save(fig, f"{period}_crps_by_hour.png")


def battery(bt, period, col):
    """Paired profit difference to the naive forecast M0 (the absolute profits share most
    of their day-to-day variation, so their own intervals overlap)."""
    seg = bt["segments"]["all"]
    diff = seg.get("minus_M0", {})
    names = list(diff)
    fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(names) + 1.5))
    for i, n in enumerate(names):
        r = diff[n]
        lo, hi = r["ci95"]
        ax.errorbar(r["mean"] / 1e3, i, xerr=[[(r["mean"] - lo) / 1e3], [(hi - r["mean"]) / 1e3]],
                    fmt="o", ms=5, color=col.get(n, INK2), elinewidth=2, capsize=0)
        share = 100 * seg["share_of_perfect"][n]
        ax.annotate(f"{share:.1f} % of perfect", (hi / 1e3, i), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax.axvline(0, color=INK2, lw=1)
    m0 = seg["profit"]["M0"]["mean"] / 1e3
    pf = seg["profit"]["perfect"]["mean"] / 1e3
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Extra profit over the naive forecast M0 (thousand EUR per MW per year)")
    ax.set_title(f"Battery 1 MW / 2 MWh: M0 earns {m0:.1f}, perfect foresight {pf:.1f} "
                 "k EUR/MW/yr", loc="left")
    save(fig, f"{period}_battery.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="test")
    a = p.parse_args()
    ev = json.loads((RESULTS_DIR / "evaluation" / f"{a.period}.json").read_text())
    col = colour_map(list(ev["models"]))
    crps_by_period(ev, a.period, col)
    coverage(ev, a.period, col)
    pit(ev, a.period, col)
    crps_by_hour(ev, a.period, col)
    bf = RESULTS_DIR / "battery" / f"{a.period}.json"
    if bf.exists():
        battery(json.loads(bf.read_text()), a.period, col)


if __name__ == "__main__":
    main()
