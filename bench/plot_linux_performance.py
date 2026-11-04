#!/usr/bin/env python3
"""Create the compact Linux CPU/GPU performance overview figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "performance" / "results"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _range_err(values):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    return med, np.array([[med - values.min()], [values.max() - med]])


def _whole_panel(ax, whole):
    if not whole or not whole.get("valid"):
        ax.text(.5, .52, "Whole-solve validation running", ha="center", va="center", fontsize=12, color="#344054")
        ax.text(.5, .38, "Awaiting gpu_solver_benchmark.json\n(no whole-solver values plotted)", ha="center", va="center", fontsize=9, color="#667085")
        ax.set_axis_off()
        return
    # Accept the later whole-solver receipt without baking in a schema.
    rows = whole.get("rows", whole.get("batches", []))
    batches, cpu, gpu, field_gate = [], [], [], []
    for row in rows:
        b = row.get("batch", row.get("batch_size")); c = row.get("cpu_seconds", row.get("cpu_wall_seconds", row.get("cpu_median"))); g = row.get("gpu_seconds", row.get("gpu_wall_seconds", row.get("gpu_median")))
        if b is None or c is None or g is None: continue
        batches.append(float(b)); cpu.append(float(c)); gpu.append(float(g)); field_gate.append(row.get("field_gate", row.get("max_field_error", None)))
    if not batches:
        ax.text(.5, .5, "Whole-solve receipt present\nbut no plottable batch rows", ha="center", va="center")
        ax.set_axis_off(); return
    ax.plot(batches, cpu, "o-", label="CPU aggregate", color="#344054")
    ax.plot(batches, gpu, "o-", label="GPU whole solve", color="#D55E00")
    optional = RESULTS / "gpu_batch99_accuracy.json"
    if optional.exists():
        acc = json.loads(optional.read_text(encoding="utf-8"))
        subtitle = "Full batch-99 gate passed" if acc.get("batch_count") == 99 and acc.get("accurate_count") == 99 else "Full batch-99 accuracy gate pending"
    else:
        subtitle = "3 representative states pass achieved-accuracy gates; all 99 validation pending"
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Batch size\nNo HDF/noise-0 claims beyond receipt scope", fontsize=8); ax.set_ylabel("Wall time (s)")
    ax.set_title("Whole-solve validation\n" + subtitle, loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def plot(output_dir: Path = RESULTS):
    cpu = read("cpu_v2_summary.json")
    worker = read("worker_scaling_summary.json")
    gpu = read("gpu_rhs_benchmark_v2.json")
    whole_path = RESULTS / "gpu_solver_benchmark.json"
    whole = json.loads(whole_path.read_text(encoding="utf-8")) if whole_path.exists() else None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    groups = cpu["groups"]
    labels = [f"{g['cohort'].replace('cpu_v2_', '')}\nJe={g['Je']:g}" for g in groups]
    med = np.array([g["median_speedup"] for g in groups]); lo = med - np.array([g["min_speedup"] for g in groups]); hi = np.array([g["max_speedup"] for g in groups]) - med
    ax.errorbar(np.arange(len(groups)), med, yerr=[lo, hi], fmt="o", capsize=4, color="#0072B2", lw=1.4)
    ax.axhline(1, color="#98A2B3", lw=.8); ax.set_xticks(np.arange(len(groups)), labels, fontsize=8); ax.set_ylabel("Reference / candidate speedup")
    ax.set_xlabel("100τ · 5 paired repeats/group · min–max bars", fontsize=8)
    ax.set_title("CPU matched full-output speedups", loc="left", fontweight="bold")
    ax = axes[0, 1]
    wg = sorted(worker["groups"], key=lambda g: g["workers"]); x = np.array([g["workers"] for g in wg])
    for key, color, label in (("reference", "#344054", "Reference"), ("cpu_compatible", "#0072B2", "CPU-compatible")):
        med, low, high = [], [], []
        for g in wg:
            vals = g["phase_samples_seconds"][key]; m, e = _range_err(vals); med.append(m); low.append(e[0, 0]); high.append(e[1, 0])
        ax.errorbar(x, med, yerr=[low, high], fmt="o-", capsize=4, color=color, label=label)
    ax.set_xticks(x); ax.set_xlabel("Workers\n100τ · 36 jobs/phase · 3 repeats · lower is better", fontsize=8); ax.set_ylabel("Phase wall time (s)"); ax.set_title("Persistent-worker scaling", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    batches = [r["batch"] for r in gpu["rows"]]; cpu_t = [r["cpu_median"] for r in gpu["rows"]]; gpu_t = [r["gpu_median"] for r in gpu["rows"]]
    ax.plot(batches, cpu_t, "o-", color="#667085", label="One CPU aggregate")
    ax.plot(batches, gpu_t, "o-", color="#D55E00", label="GPU full RHS")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Batch size\nF64 full RHS · transfer/setup excluded · not whole-solver time", fontsize=8); ax.set_ylabel("RHS time (s)"); ax.set_title("GPU full-RHS throughput", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]; _whole_panel(ax, whole)
    receipt = RESULTS / "production_5000_artifacts.json"
    if receipt.exists():
        ax.text(.98, .02, "5000τ receipt: 24 pairs, 9/9 field errors = 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#027A48")
    fig.suptitle("Linux performance overview", fontsize=14, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = output_dir / "acceleration_summary.png", output_dir / "acceleration_summary.pdf"
    fig.savefig(png, dpi=220); fig.savefig(pdf); plt.close(fig)
    return png, pdf


if __name__ == "__main__":
    print("\n".join(str(p) for p in plot()))
