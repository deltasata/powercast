"""Markdown tables for the README, made only from saved results.

Usage:
    python scripts/report.py --period test > results/report_test.md

Reads results/evaluation/<period>.json and results/battery/<period>.json.
"""

from __future__ import annotations

import argparse
import json

from powercast.config import RESULTS_DIR

SEGMENTS = {"all": "whole period", "R0": "R0 Jan–Aug 2021", "R1": "R1 Sep 2021–Dec 2022",
            "R2": "R2 2023–Sep 2025", "R3": "R3 Oct 2025–Oct 2026 (15 min)"}


def ci(r, scale=1.0, fmt="{:.2f}"):
    return (f"{fmt.format(r['mean'] * scale)} "
            f"[{fmt.format(r['ci95'][0] * scale)}, {fmt.format(r['ci95'][1] * scale)}]")


def scores_table(ev) -> str:
    rows = ["| Model | CRPS (EUR/MWh) | MAE (EUR/MWh) | Coverage 50 / 80 / 90 / 98 % |",
            "|---|---|---|---|"]
    for m, r in ev["models"].items():
        a = r["all"]
        cov = " / ".join(f"{100 * a[f'cov{c}']['mean']:.1f}" for c in (50, 80, 90, 98))
        rows.append(f"| {m} | {ci(a['crps'])} | {ci(a['mae'])} | {cov} |")
    return "\n".join(rows)


def regime_table(ev) -> str:
    segs = [s for s in SEGMENTS if s in next(iter(ev["models"].values()))]
    rows = ["| Model | " + " | ".join(SEGMENTS[s] for s in segs) + " |",
            "|---|" + "---|" * len(segs)]
    for m, r in ev["models"].items():
        rows.append(f"| {m} | " + " | ".join(ci(r[s]["crps"]) for s in segs) + " |")
    if "R3_hourly_means" in next(iter(ev["models"].values())):
        rows += ["", "R3 scored on hourly mean prices (comparable with R0–R2):", "",
                 "| Model | CRPS (EUR/MWh) | Coverage 80 % |", "|---|---|---|"]
        for m, r in ev["models"].items():
            h = r["R3_hourly_means"]
            rows.append(f"| {m} | {ci(h['crps'])} | {100 * h['cov80']:.1f} |")
    return "\n".join(rows)


def subset_table(ev) -> str:
    rows = ["| Model | CRPS, negative prices | Coverage 80 %, negative | CRPS, spikes | "
            "Coverage 80 %, spikes |", "|---|---|---|---|---|"]
    for m, r in ev["models"].items():
        cells = []
        for sub in ("negative", "spike"):
            s = r.get(sub)
            cells += ([ci(s["crps"]), f"{100 * s['cov80']['mean']:.1f}"] if s else ["–", "–"])
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def dm_table(ev, seg="all") -> str:
    names = list(ev["models"])
    mat = ev["dm"][seg]
    rows = ["| A \\ B | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("–")
            else:
                p = mat[f"{a}<{b}"]
                txt = "< 0.001" if p < 0.001 else f"{p:.3f}"
                cells.append(f"**{txt}**" if p < 0.05 else txt)
        rows.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def battery_table(bt) -> str:
    seg = bt["segments"]["all"]
    rows = ["| Schedule from | Profit (EUR/MW/yr) | Share of perfect foresight | "
            "Difference to M0 (EUR/MW/yr) | DM p (more profit than M0) |",
            "|---|---|---|---|---|"]
    for n, r in seg["profit"].items():
        diff = seg.get("minus_M0", {}).get(n)
        p = seg.get("dm_p_more_profit_than_M0", {}).get(n)
        rows.append(f"| {n} | {ci(r, fmt='{:,.0f}')} | {100 * seg['share_of_perfect'][n]:.1f} % | "
                    f"{ci(diff, fmt='{:+,.0f}') if diff else '–'} | "
                    f"{f'{p:.3g}' if p is not None else '–'} |")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="test")
    a = p.parse_args()
    ev = json.loads((RESULTS_DIR / "evaluation" / f"{a.period}.json").read_text())
    print(f"### Scores, {a.period} period (95 % block-bootstrap intervals)\n")
    print(scores_table(ev) + "\n")
    if a.period == "test":
        print("### CRPS by market period\n")
        print(regime_table(ev) + "\n")
    print("### Negative prices and spikes\n")
    print(subset_table(ev) + "\n")
    print("### Diebold–Mariano p-values (H1: row model A has lower CRPS than column B)\n")
    print(dm_table(ev) + "\n")
    bf = RESULTS_DIR / "battery" / f"{a.period}.json"
    if bf.exists():
        print("### Battery (1 MW / 2 MWh, 90 % round trip, one cycle per day)\n")
        print(battery_table(json.loads(bf.read_text())) + "\n")


if __name__ == "__main__":
    main()
