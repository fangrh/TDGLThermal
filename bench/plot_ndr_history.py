"""Plot verified ordered-current histories and their finite-difference slopes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if not data["complete"] or data["errors"] or not data["artifact_validation"]:
        raise ValueError("a complete, artifact-verified history receipt is required")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    slopes = []
    for direction, color, style in (("ascending", "#0072B2", "o-"),
                                     ("descending", "#D55E00", "s--")):
        seeds = sorted({r["seed"] for r in data["rows"] if r["direction"] == direction})
        for index, seed in enumerate(seeds):
            rows = sorted((r for r in data["rows"] if r["direction"] == direction and r["seed"] == seed), key=lambda r: r["step"])
            current = np.array([r["Je"] for r in rows])
            voltage = np.array([r["V"] for r in rows])
            slope = np.diff(voltage) / np.diff(current)
            label = direction.capitalize() if index == 0 else None
            axes[0].plot(current, voltage, style, color=color, alpha=.75, label=label)
            axes[1].plot((current[1:] + current[:-1]) / 2, slope, style, color=color, alpha=.75)
            slopes.extend({"direction": direction, "seed": seed, "edge": i, "dV_dJe": float(v)} for i, v in enumerate(slope))
    axes[0].set_title("Ordered-current responses")
    axes[0].set_ylabel("V (model units)"); axes[0].legend(frameon=False)
    axes[1].set_title("All observed adjacent slopes are positive" if all(r["dV_dJe"] > 0 for r in slopes) else "Adjacent finite-difference slopes")
    axes[1].set_ylabel("dV/dJe (model units)"); axes[1].axhline(0, color="gray", lw=1)
    midpoint = sorted((current[1:] + current[:-1]) / 2)
    axes[1].set_xticks(midpoint, [f"{j:.5f}" for j in midpoint])
    for axis in axes:
        axis.set_xlabel("Je (model units)"); axis.grid(alpha=.2)
    fig.suptitle("Full curl · 3 seeds/direction · 5000τ dwell · final 2000τ means", fontsize=12)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"history.{ext}", dpi=190)
    plt.close(fig)
    receipt = {"source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "slopes": slopes, "all_adjacent_slopes_positive": all(r["dV_dJe"] > 0 for r in slopes),
               "scope": "descriptive three-seed ordered histories, not an eight-seed stationarity test"}
    (args.out / "history_slopes.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
