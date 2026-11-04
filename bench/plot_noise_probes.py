#!/usr/bin/env python3
"""Plot the fixed-state noise-cost and archived-state timestep probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_in = root / "docs" / "performance" / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, default=default_in / "noise_cost_probe.json")
    parser.add_argument("--timesteps", type=Path, default=default_in / "noise_timestep_probe.json")
    parser.add_argument("--out-dir", type=Path, default=default_in)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost = json.loads(args.cost.read_text(encoding="utf-8"))
    timesteps = json.loads(args.timesteps.read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 120})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    fig.suptitle(
        "Noise-path diagnostics — fixed-state microbenchmark + 100τ archived-state simplified probe\n"
        "No production speed claim; no noise-on/off physical-equivalence claim",
        fontsize=12,
    )

    # Panel A: use receipt medians and show the seven raw repetitions as dots.
    preps = ["configured_down", "cool"]
    labels = ["configured\nhot start", "random\ncool state"]
    summary = {row["preparation"]: row for row in cost["summary"]}
    x = np.arange(len(preps))
    width = 0.34
    off = np.array([summary[p]["noise_off_rhs_seconds"] for p in preps]) * 1e6
    on = np.array([summary[p]["noise_on_rhs_seconds"] for p in preps]) * 1e6
    ax_a.bar(x - width / 2, off, width, color="#9aa7b2", label="noise 0")
    ax_a.bar(x + width / 2, on, width, color="#d95f02", label="noise 0.001")
    raw = cost["rows"]
    rng = np.random.default_rng(7)
    for xpos, prep in zip(x, preps):
        for noise, color, dx in ((0.0, "#425466", -width / 2), (0.001, "#8c2d04", width / 2)):
            vals = [r["rhs"]["seconds_per_call"] * 1e6 for r in raw
                    if r["preparation"] == prep and r["noise"] == noise]
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax_a.scatter(np.full(len(vals), xpos + dx) + jitter, vals, s=17,
                         color=color, edgecolor="white", linewidth=0.35, zorder=3,
                         label="7 repetitions (not CI)" if xpos == 0 and noise == 0.001 else None)
    ax_a.set_xticks(x, labels)
    ax_a.set_ylabel("RHS time per call (μs)")
    ax_a.set_title("A. Fixed-state RHS cost")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel B: accepted steps against tolerance; line style encodes noise.
    rows = timesteps["rows"]
    currents = sorted({float(r["Je"]) for r in rows})
    colors = ["#1b9e77", "#7570b3", "#e7298a"]
    for current, color in zip(currents, colors):
        for noise, style, label_noise in ((0.0, "--", "noise 0"), (0.001, "-", "noise 0.001")):
            selected = sorted((r for r in rows if float(r["Je"]) == current and r["noise"] == noise),
                              key=lambda r: float(r["tolerance"]))
            tolerances = np.array([float(r["tolerance"]) for r in selected])
            accepts = np.array([int(r["naccept"]) for r in selected])
            ax_b.plot(tolerances, accepts, linestyle=style, marker="o", markersize=4,
                      color=color, label=f"Je={current:.3f}, {label_noise}")
    ax_b.axvline(1e-4, color="black", linewidth=0.9, alpha=0.65)
    ax_b.text(1.07e-4, ax_b.get_ylim()[1] * 0.98, "production\n1e-4", fontsize=8,
              va="top", ha="left")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("relative/absolute tolerance")
    ax_b.set_ylabel("accepted steps over 100τ")
    ax_b.set_title("B. Adaptive-step response")
    ax_b.grid(alpha=0.25)
    ax_b.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "noise_probes.png"
    pdf = args.out_dir / "noise_probes.pdf"
    fig.savefig(png, dpi=220, metadata={"Title": "TDGL noise probes"})
    fig.savefig(pdf, metadata={"Title": "TDGL noise probes"})
    plt.close(fig)
    print(json.dumps({"png": str(png), "pdf": str(pdf),
                      "cost_sha256": sha256(args.cost),
                      "timesteps_sha256": sha256(args.timesteps),
                      "currents": currents}))


if __name__ == "__main__":
    main()
