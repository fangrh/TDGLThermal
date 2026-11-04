#!/usr/bin/env python3
"""Strict receipt, accuracy, throughput, and IV summary for a full worker scan."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

from compare_results import field_errors, meta_errors, validate_artifact

FIELDS = ("times", "V", "psi_avg", "T_avg", "psi_real", "psi_imag", "T", "Ax", "Ay")
BACKENDS = ("reference", "cpu_compatible")
T95_DF2 = 4.30265


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contract_failures(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    rows = data.get("rows")
    if not isinstance(rows, list):
        return ["rows missing or not a list"]
    if len(rows) != 594:
        failures.append(f"expected 594 rows, got {len(rows)}")
    if float(data.get("horizon", float("nan"))) != 5000.0:
        failures.append(f"worker horizon is {data.get('horizon')}, expected 5000")
    if int(data.get("repetitions", -1)) != 3 or int(data.get("samples_per_current", -1)) != 1:
        failures.append("expected repetitions=3 and samples_per_current=1")
    if float(data.get("noise", float("nan"))) != 0.001 or data.get("preparation") != "configured_down":
        failures.append("noise/preparation protocol mismatch")
    if data.get("full_scan") is not True or int(data.get("actual_nworkers", -1)) != 24:
        failures.append("expected full_scan=true with 24 workers")
    grid = data.get("current_grid", [])
    if len(grid) != 99 or len({float(x) for x in grid}) != 99 or any(not math.isfinite(float(x)) for x in grid) or data.get("selected_indices") != list(range(1, 100)):
        failures.append("expected 99-point current grid and selected indices 1..99")
    if int(data.get("seed_base", -1)) != 2026110401:
        failures.append("unexpected seed base")
    keys = defaultdict(set)
    tuple_counts = defaultdict(int)
    expected_keys = {(rep, idx, 1) for rep in (1, 2, 3) for idx in range(1, 100)}
    for row in rows:
        key = (row.get("repetition"), row.get("current_index"), row.get("sample"))
        keys[key].add(row.get("backend"))
        tuple_counts[(key, row.get("backend"))] += 1
        if row.get("backend") not in BACKENDS:
            failures.append(f"unsupported backend {row.get('backend')}")
        expected_seed = 2026110401 + 100 * (int(row.get("repetition", -1)) - 1)
        if row.get("seed") != expected_seed:
            failures.append(f"unexpected seed for {key}: {row.get('seed')}")
        index = int(row.get("current_index", -1))
        if not (1 <= index <= 99) or not math.isclose(float(row.get("Je", float("nan"))), float(grid[index - 1]), rel_tol=0, abs_tol=1e-15):
            failures.append(f"current-grid mismatch for {key}")
        if not row.get("complete") or row.get("status") not in ("success", "Success", "ReturnCode.Success"):
            failures.append(f"incomplete row {key}/{row.get('backend')}")
    if set(keys) != expected_keys:
        failures.append("expected exactly repetitions 1..3 × current indices 1..99 × sample 1")
    if len(keys) != 297:
        failures.append(f"expected 297 paired trajectories, got {len(keys)}")
    if any(v != set(BACKENDS) for v in keys.values()):
        failures.append("missing backend in one or more paired trajectories")
    if any(n != 1 for n in tuple_counts.values()):
        failures.append("duplicate trajectory/backend tuple")
    phases = data.get("phases", [])
    if len(phases) != 6 or any(int(p.get("jobs", -1)) != 99 or int(p.get("jobs_per_phase", -1)) != 99 or
                               p.get("backend") not in BACKENDS or not math.isfinite(float(p.get("total_phase_wall_seconds", float("nan")))) or float(p.get("total_phase_wall_seconds", 0)) <= 0
                               for p in phases):
        failures.append("expected six positive 99-job phase receipts")
    if {(p.get("repetition"), p.get("phase"), p.get("backend")) for p in phases} != {
            (rep, phase, "reference" if phase == 1 and rep % 2 == 1 or phase == 2 and rep % 2 == 0 else "cpu_compatible")
            for rep in (1, 2, 3) for phase in (1, 2)}:
        failures.append("phase repetition/backend matrix mismatch")
    return failures


def ci_log(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    half = T95_DF2 * sem
    return {"mean_log": mean, "ci95_low": mean - half, "ci95_high": mean + half,
            "ratio": math.exp(mean), "ratio_ci95_low": math.exp(mean - half),
            "ratio_ci95_high": math.exp(mean + half), "n": len(values), "df": len(values) - 1}


def phase_timings(data: dict[str, Any]) -> dict[str, list[float]]:
    out = {backend: [] for backend in BACKENDS}
    phases = sorted(data.get("phases", []), key=lambda p: (int(p.get("repetition", -1)), int(p.get("phase", -1))))
    for phase in phases:
        backend = phase.get("backend")
        if backend in out:
            out[backend].append(float(phase.get("total_phase_wall_seconds", float("nan"))))
    return out


def summarize(runroot: pathlib.Path, plot_path: pathlib.Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = runroot / "worker_scaling.json"
    data = load(source)
    failures = contract_failures(data)
    rows = data.get("rows", [])
    pairs: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[(int(row.get("repetition", -1)), int(row.get("current_index", -1)), int(row.get("sample", -1)))][row.get("backend")] = row
    pair_receipts: list[dict[str, Any]] = []
    maximum_field_error = 0.0
    iv: dict[float, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, pair in sorted(pairs.items()):
        if set(pair) != set(BACKENDS):
            continue
        ref, cand = pair["reference"], pair["cpu_compatible"]
        if ref.get("Je") != cand.get("Je") or ref.get("seed") != cand.get("seed"):
            failures.append(f"pair metadata mismatch {key}")
        manifests: dict[str, dict[str, Any]] = {}
        artifacts: dict[str, Any] = {}
        h5s: dict[str, pathlib.Path] = {}
        for backend, row in pair.items():
            mp = pathlib.Path(str(row.get("output_path", ""))).with_suffix(".json")
            try:
                m = load(mp)
            except (OSError, ValueError, TypeError) as exc:
                failures.append(f"missing manifest {key}/{backend}: {exc}"); continue
            manifests[backend] = m
            expected_role = "corrected_reference" if backend == "reference" else "compatible_candidate"
            for field, value in (("backend", backend), ("role", expected_role), ("horizon", 5000.0),
                                 ("final_time", 5000.0), ("noise", 0.001), ("Nx", 60), ("Ny", 30),
                                 ("tolerance", 1e-4), ("snapshot_interval", 1.0), ("skip_ratio", 0.9),
                                 ("solver", "Tsit5"), ("save_fields", True), ("complete", True)):
                if m.get(field) != value: failures.append(f"{backend} protocol {field} mismatch {key}")
            if m.get("status") not in ("success", "Success", "ReturnCode.Success"):
                failures.append(f"{backend} failed status {key}")
            initial_path = pathlib.Path(str(m.get("initial_vector_path", "")))
            if not initial_path.is_file():
                failures.append(f"missing initial vector {key}/{backend}: {initial_path}")
            elif sha256(initial_path) != m.get("initial_vector_hash"):
                failures.append(f"initial vector hash mismatch {key}/{backend}")
            hp = pathlib.Path(str(m.get("output_path", "")))
            if not hp.is_file(): failures.append(f"missing HDF5 {key}/{backend}: {hp}"); continue
            h5s[backend] = hp
            artifact_errors, resolved = validate_artifact(mp, m)
            if artifact_errors:
                failures.extend(f"{backend} artifact contract {key}: {error}" for error in artifact_errors)
            actual_hash = sha256(hp)
            if actual_hash != m.get("output_hash"): failures.append(f"output hash mismatch {key}/{backend}")
            artifacts[backend] = {"manifest": str(mp), "hdf5": str(hp), "output_hash": actual_hash,
                                  "initial_hash": m.get("initial_hash"), "initial_vector_path": str(initial_path),
                                  "initial_vector_hash": m.get("initial_vector_hash"),
                                  "initial_vector_actual_hash": sha256(initial_path) if initial_path.is_file() else None,
                                  "config_hash": m.get("config_hash"), "model_hash": m.get("model_hash"),
                                  "rhs_source_hash": m.get("rhs_source_hash")}
        if set(manifests) != set(BACKENDS) or set(h5s) != set(BACKENDS): continue
        metadata_errors = meta_errors(manifests["reference"], manifests["cpu_compatible"])
        if metadata_errors:
            failures.extend(f"paired manifest contract {key}: {error}" for error in metadata_errors)
        for field in ("config_hash", "model_hash", "rhs_source_hash", "julia_version", "machine", "initial_hash", "initial_vector_hash", "seed", "Je", "noise", "horizon", "tolerance", "snapshot_interval", "skip_ratio", "solver", "precision", "output_schema"):
            if manifests["reference"].get(field) != manifests["cpu_compatible"].get(field):
                failures.append(f"paired provenance mismatch {field} {key}")
        for field in ("rhs_calls", "accepted_steps", "rejected_steps"):
            if manifests["reference"].get(field) != manifests["cpu_compatible"].get(field): failures.append(f"paired solver stats mismatch {field} {key}")
        try:
            import h5py
            import numpy as np
            with h5py.File(h5s["reference"], "r") as rh, h5py.File(h5s["cpu_compatible"], "r") as ch:
                common_errors, common_values = field_errors(h5s["reference"], h5s["cpu_compatible"])
                failures.extend(f"field contract {key}: {error}" for error in common_errors)
                if set(common_values) != set(FIELDS):
                    failures.append(f"field contract missing observables {key}")
                for name, error in common_values.items():
                    maximum_field_error = max(maximum_field_error, error)
                    if error > 1e-11:
                        failures.append(f"field contract error {name}={error} {key}")
                for name in FIELDS:
                    if name not in rh or name not in ch: failures.append(f"missing dataset {name} {key}"); continue
                    a, b = np.asarray(rh[name]), np.asarray(ch[name])
                    if a.shape != b.shape: failures.append(f"shape mismatch {name} {key}"); continue
                    if not np.isfinite(a).all() or not np.isfinite(b).all(): failures.append(f"nonfinite dataset {name} {key}"); continue
                    err = float(np.max(np.abs(a - b)) / (1 + np.max(np.abs(a))))
                    if err > 1e-11: failures.append(f"field error {name}={err} {key}")
                times = np.asarray(rh["times"])
                if times.size == 0 or not math.isclose(float(times[-1]), 5000.0, abs_tol=1e-10): failures.append(f"shortened horizon {key}")
                start = int(math.ceil(5000 * 0.9))
                for backend, handle in (("reference", rh), ("cpu_compatible", ch)):
                    mask = np.asarray(handle["times"]) >= 4500.0
                    for field in ("V", "T_avg", "psi_avg"):
                        iv[float(ref["Je"])][backend][field].append(float(np.mean(np.asarray(handle[field])[mask])))
        except ImportError:
            failures.append("h5py/numpy unavailable for artifact validation")
        pair_receipts.append({"repetition": key[0], "current_index": key[1], "sample": key[2], "current": ref.get("Je"), "seed": ref.get("seed"), "reference_wall_seconds": manifests["reference"].get("wall_seconds"), "candidate_wall_seconds": manifests["cpu_compatible"].get("wall_seconds"), "speedup_reference_over_candidate": float(manifests["reference"].get("wall_seconds", 0)) / float(manifests["cpu_compatible"].get("wall_seconds", 1)), "field_errors": common_values, "artifacts": artifacts})
    phase_times = phase_timings(data)
    rep_logs = [math.log(r / c) for r, c in zip(phase_times["reference"], phase_times["cpu_compatible"])
                if math.isfinite(r) and math.isfinite(c) and r > 0 and c > 0]
    result: dict[str, Any] = {"schema": "tdgl-worker-full-scan-v1", "valid": not failures,
        "source": str(source), "source_sha256": sha256(source), "rows": len(rows), "paired_trajectories": len(pair_receipts),
        "protocol": {k: data.get(k) for k in ("horizon", "repetitions", "samples_per_current", "noise", "preparation", "tolerance", "skip_ratio", "snapshot_interval", "actual_nworkers", "seed_base")},
        "source_settings": {k: data.get(k) for k in ("config_source_hash", "benchmark_source_hash", "julia_version", "created_utc", "worker_info")},
        "phase_performance": {backend: {"n": len(values), "wall_seconds": values,
                                         "median_wall_seconds": statistics.median(values) if values else None,
                                         "min": min(values) if values else None,
                                         "max": max(values) if values else None}
                              for backend, values in phase_times.items()},
        "paired_speedup_log_ci95_df2": ci_log(rep_logs) if len(rep_logs) == 3 else {"error": "expected three repetition means"},
        "field_error_summary": {"maximum": maximum_field_error, "all_exact_zero": maximum_field_error == 0.0, "all_within_threshold": maximum_field_error <= 1e-11},
        "iv_window": {"start": 4500.0, "end": 5000.0, "observable": "retained HDF5 samples"},
        "iv": iv, "pairs": pair_receipts, "failures": failures}
    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            currents = sorted(iv)
            fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            for ax, field, label in zip(axes, ("V", "T_avg", "psi_avg"), ("V", "T", "mean |psi|²")):
                for backend, style in (("reference", "-"), ("cpu_compatible", "--")):
                    means = [statistics.mean(iv[c][backend][field]) for c in currents]
                    lows = [min(iv[c][backend][field]) for c in currents]
                    highs = [max(iv[c][backend][field]) for c in currents]
                    ax.plot(currents, means, style, label=backend)
                    ax.fill_between(currents, lows, highs, alpha=.15)
                ax.set_ylabel(label); ax.grid(alpha=.25)
            fig.suptitle("Full scan IV curves: horizon 5000, noise 0.001; three-seed mean and min/max", fontsize=11)
            axes[0].legend(); axes[-1].set_xlabel("Je"); fig.tight_layout(); fig.savefig(plot_path, dpi=180)
            fig.savefig(plot_path.with_suffix(".pdf")); plt.close(fig)
        except ImportError:
            result["failures"].append("matplotlib unavailable for IV plot"); result["valid"] = False
    return result, pair_receipts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("runroot", type=pathlib.Path); p.add_argument("--output", type=pathlib.Path); p.add_argument("--csv", type=pathlib.Path); p.add_argument("--plot", type=pathlib.Path)
    a = p.parse_args(argv)
    try: result, pairs = summarize(a.runroot, a.plot)
    except Exception as exc: result, pairs = {"schema": "tdgl-worker-full-scan-v1", "valid": False, "failures": [f"exception: {exc}"], "pairs": []}, []
    out = a.output or a.runroot / "full_scan_24_5000_summary.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if a.csv:
        with a.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=("repetition", "current_index", "sample", "current", "seed", "reference_wall_seconds", "candidate_wall_seconds", "speedup_reference_over_candidate")); w.writeheader(); w.writerows([{k: x.get(k) for k in w.fieldnames} for x in pairs])
    print(json.dumps({"valid": result.get("valid"), "pairs": len(pairs), "failures": len(result.get("failures", [])), "output": str(out)}))
    return 0 if result.get("valid") else 2


if __name__ == "__main__": raise SystemExit(main())
