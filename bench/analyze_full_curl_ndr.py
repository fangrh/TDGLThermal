#!/usr/bin/env python3
"""Receipt-aware analysis for the full-curl NDR endpoint campaign.

The analysis deliberately treats each seed's trajectory as one observation.  HDF5
frames are useful for plotting and time-window means, but are never treated as
independent replicates.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from compare_results import validate_artifact

T_ONE_7, T_TWO_7 = 1.8945786, 2.364624
T_ONE_2, T_TWO_2 = 2.919985, 4.302653
EXPECTED_SEEDS = tuple(range(2026110411, 2026110419))
EXPECTED_SEGMENTS, SEGMENT_HORIZON, TOTAL_HORIZON = 10, 5000.0, 50000.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chain_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/segment*/chain.json"))


def read_chains(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Return chain entries grouped by job id, preserving segment order."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in chain_files(root):
        try:
            entry = _json(path)
            entry["_chain_path"] = str(path)
            entry["_segment_dir"] = str(path.parent)
            grouped[str(entry.get("id", path.parent.parent.name))].append(entry)
        except (OSError, ValueError, TypeError):
            continue
    for rows in grouped.values():
        rows.sort(key=lambda x: int(x.get("segment", 0)))
    return dict(grouped)


def read_chains_with_errors(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for path in chain_files(root):
        try:
            entry = _json(path)
            if not isinstance(entry, dict): raise ValueError("receipt is not an object")
            entry["_chain_path"], entry["_segment_dir"] = str(path), str(path.parent)
            grouped[str(entry.get("id", path.parent.parent.name))].append(entry)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"malformed chain receipt {path}: {exc}")
    for rows in grouped.values(): rows.sort(key=lambda x: int(x.get("segment", 0)))
    return dict(grouped), errors


def _manifest(entry: dict[str, Any]) -> dict[str, Any]:
    m = entry.get("manifest", {})
    return m if isinstance(m, dict) else {}


def _output_path(entry: dict[str, Any]) -> Path:
    m = _manifest(entry)
    p = Path(str(m.get("output_path", "")))
    if p.is_file():
        return p
    return Path(entry["_segment_dir"]) / p.name


def _mean_window(times: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    mask = (times >= lo) & (times <= hi)
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def read_segment(entry: dict[str, Any]) -> dict[str, Any]:
    """Read scalar observables and the Ax-integrated voltage for one segment."""
    path = _output_path(entry)
    with h5py.File(path, "r") as h:
        times = np.asarray(h["times"], dtype=float)
        out: dict[str, Any] = {k: np.asarray(h[k], dtype=float) for k in ("times", "V", "T_avg", "psi_avg")}
        axds = h["Ax"]
        if axds.ndim != 3 or axds.shape[0] != len(times) or axds.shape[1] < 3 or axds.shape[2] < 3:
            raise ValueError(f"invalid Ax shape in {path}: {axds.shape}")
        # Ax dimensions are (time, y, x); this is intentionally the same
        # interior averaging convention as observable compute_voltage.
        out["V_integrated"] = compute_integrated_voltage(np.asarray(axds[[0, -1]], dtype=float), np.asarray([times[0], times[-1]]))
        out["path"] = str(path)
        out["cumulative_start"] = float(entry.get("cumulative_start", 0.0))
        out["cumulative_end"] = float(entry.get("cumulative_end", times[-1]))
    return out


def _seed_and_endpoint(groups: dict[str, list[dict[str, Any]]]) -> dict[tuple[float, int], list[dict[str, Any]]]:
    out: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for entries in groups.values():
        if not entries:
            continue
        first = entries[0]
        out[(float(first.get("Je", math.nan)), int(first.get("seed", -1)))].extend(entries)
    return out


def _complete_chain(entries: list[dict[str, Any]], total: float = TOTAL_HORIZON, expected_je: float | None = None) -> bool:
    if len(entries) != EXPECTED_SEGMENTS:
        return False
    entries = sorted(entries, key=lambda x: int(x.get("segment", 0)))
    if expected_je is not None and any(not math.isclose(float(e.get("Je", math.nan)), expected_je, abs_tol=1e-15) for e in entries): return False
    for e in entries:
        m = _manifest(e)
        if (m.get("horizon"), m.get("skip_ratio"), m.get("noise"), m.get("tolerance")) != (5000.0, 0.6, 0.001, 1e-4): return False
        if not m.get("complete") or m.get("status") not in ("success", "Success", "ReturnCode.Success"): return False
    provenance = [{k: _manifest(e).get(k) for k in ("config_hash", "model_hash", "rhs_source_hash", "output_schema")} for e in entries]
    if any(x != provenance[0] for x in provenance[1:]): return False
    return all(int(e.get("segment", 0)) == i + 1 and
               math.isclose(float(e.get("cumulative_start", math.nan)), i * SEGMENT_HORIZON, abs_tol=1e-8) and
               math.isclose(float(e.get("cumulative_end", math.nan)), (i + 1) * SEGMENT_HORIZON, abs_tol=1e-8) and
               bool(_manifest(e).get("complete", False)) for i, e in enumerate(entries)) and math.isclose(float(entries[-1].get("cumulative_end")), total, abs_tol=1e-8)


def _paired_stats(values: list[float], threshold: float = -0.01) -> dict[str, Any]:
    n = len(values); mean = float(np.mean(values)) if n else float("nan")
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    t = {8: T_ONE_7, 3: T_ONE_2}.get(n, float("nan"))
    upper = mean + t * sem if math.isfinite(sem) else float("nan")
    return {"n": n, "values": values, "mean": mean, "sd": sd, "sem": sem,
            "ci95_upper_one_sided": upper, "threshold": threshold,
            "pass": n == 8 and math.isfinite(upper) and upper < threshold,
            "df": n - 1}


paired_endpoint_stats = _paired_stats


def _stationarity(values: list[float]) -> dict[str, Any]:
    n = len(values); mean = float(np.mean(values)) if n else float("nan")
    sem = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    t = {8: T_TWO_7, 3: T_TWO_2}.get(n, float("nan"))
    half = t * sem if math.isfinite(sem) else float("nan")
    lo, hi = mean - half, mean + half
    return {"n": n, "values": values, "mean": mean, "ci95_low": lo, "ci95_high": hi,
            "tolerance": 0.0025, "pass": n in (3, 8) and lo >= -0.0025 and hi <= 0.0025}


stationarity_stats = _stationarity


def compute_integrated_voltage(ax: np.ndarray, times: np.ndarray) -> float:
    """Compute endpoint Ax voltage in native model units."""
    ax = np.asarray(ax); times = np.asarray(times, dtype=float)
    if ax.ndim != 3 or times.ndim != 1 or ax.shape[0] != times.size or times[-1] == times[0]:
        return float("nan")
    return float(-30.0 * (np.mean(ax[-1, 1:-1, 1:-1]) - np.mean(ax[0, 1:-1, 1:-1])) / (times[-1] - times[0]))


def validate_chain_artifacts(entries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    previous_output: Path | None = None
    previous_rng: Path | None = None
    previous_entry: dict[str, Any] | None = None
    for e in sorted(entries, key=lambda x: int(x.get("segment", 0))):
        m = _manifest(e); mp = Path(e["_chain_path"]).with_name(Path(m.get("output_path", "x.h5")).with_suffix(".json").name)
        # compare_results resolves relative paths against the manifest path.
        ae, resolved = validate_artifact(mp, m)
        errors.extend(ae)
        out = _output_path(e)
        if resolved is not None and out.is_file() and _sha256(out) != m.get("output_hash"):
            errors.append(f"output SHA mismatch segment {e.get('segment')}")
        iv = Path(str(m.get("initial_vector_path", "")))
        if not iv.is_file(): iv = Path(e["_segment_dir"]) / iv.name
        if not iv.is_file(): errors.append(f"missing initial binary segment {e.get('segment')}")
        elif m.get("initial_vector_hash") and _sha256(iv) != m.get("initial_vector_hash"): errors.append(f"initial binary SHA mismatch segment {e.get('segment')}")
        rng = Path(str(e.get("rng_after_path", "")))
        if not rng.is_file(): rng = Path(e["_segment_dir"]) / rng.name
        if not rng.is_file(): errors.append(f"missing RNG checkpoint segment {e.get('segment')}")
        elif e.get("rng_after_sha256") and _sha256(rng) != e.get("rng_after_sha256"): errors.append(f"RNG SHA mismatch segment {e.get('segment')}")
        if previous_output is not None and e.get("preceding_sha256") != _sha256(previous_output): errors.append(f"HDF5 continuity mismatch segment {e.get('segment')}")
        if previous_rng is not None:
            before = str(e.get("rng_before_sha256", ""))
            if before != str(previous_entry.get("rng_after_sha256", "")) or before != _sha256(previous_rng): errors.append(f"RNG continuity mismatch segment {e.get('segment')}")
            recorded_rng = Path(str(e.get("preceding_rng_path", "")))
            if not recorded_rng.is_file() or _sha256(recorded_rng) != before: errors.append(f"RNG path continuity mismatch segment {e.get('segment')}")
        if previous_output is not None and m.get("initial_source_hash") != _sha256(previous_output):
            errors.append(f"initial-state continuity mismatch segment {e.get('segment')}")
        previous_output, previous_rng = out, rng
        previous_entry = e
    return errors


def analyze(endpoint_root: Path, screen_root: Path | None = None, validate_artifacts: bool = False) -> dict[str, Any]:
    endpoint, errors = read_chains_with_errors(endpoint_root)
    by_seed: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for job, rows in endpoint.items():
        if rows: by_seed[int(rows[0].get("seed", -1))][job.split("_seed")[0]] = rows
    expected_jes = {"endpoint_a": 0.20816326530612245, "endpoint_b": 0.21122448979591835}
    complete_seeds = [s for s in EXPECTED_SEEDS if set(by_seed.get(s, {})) == set(expected_jes) and all(_complete_chain(by_seed[s][k], expected_je=v) for k, v in expected_jes.items())]
    if validate_artifacts:
        for rows in endpoint.values(): errors.extend(validate_chain_artifacts(rows))
    pairs = []
    traces = {}
    for seed in complete_seeds:
        a, b = by_seed[seed].get("endpoint_a"), by_seed[seed].get("endpoint_b")
        if not a or not b: continue
        try:
            sa = [read_segment(x) for x in a]; sb = [read_segment(x) for x in b]
            ma = _mean_window(sa[-1]["times"] + 45000, sa[-1]["V"], 48000, 50000)
            mb = _mean_window(sb[-1]["times"] + 45000, sb[-1]["V"], 48000, 50000)
            pairs.append({"seed": seed, "V_a": ma, "V_b": mb, "delta_V": mb - ma,
                          "T_avg_a": _mean_window(sa[-1]["times"] + 45000, sa[-1]["T_avg"], 48000, 50000), "T_avg_b": _mean_window(sb[-1]["times"] + 45000, sb[-1]["T_avg"], 48000, 50000),
                          "psi_avg_a": _mean_window(sa[-1]["times"] + 45000, sa[-1]["psi_avg"], 48000, 50000), "psi_avg_b": _mean_window(sb[-1]["times"] + 45000, sb[-1]["psi_avg"], 48000, 50000),
                          "V_integrated_a": sa[-1]["V_integrated"], "V_integrated_b": sb[-1]["V_integrated"],
                          "alias_difference_a": ma - sa[-1]["V_integrated"], "alias_difference_b": mb - sb[-1]["V_integrated"]})
            traces[seed] = {"a": sa, "b": sb}
        except (OSError, KeyError, ValueError, IndexError) as exc: errors.append(f"seed {seed}: {exc}")
    stats = _paired_stats([x["delta_V"] for x in pairs])
    stats["pass"] = stats["pass"] and len(complete_seeds) == 8 and not errors
    drift_values: dict[str, list[float]] = {"endpoint_a": [], "endpoint_b": []}
    for seed, tr in traces.items():
        for key, label in (("a", "endpoint_a"), ("b", "endpoint_b")):
            late = _mean_window(tr[key][-1]["times"] + 45000, tr[key][-1]["V"], 48000, 50000)
            prev = _mean_window(tr[key][-2]["times"] + 40000, tr[key][-2]["V"], 43000, 45000) if len(tr[key]) > 1 else float("nan")
            if math.isfinite(late) and math.isfinite(prev): drift_values[label].append(late - prev)
    stationarity = {label: _stationarity(vals) for label, vals in drift_values.items()}
    stationarity["pass"] = len(pairs) == 8 and not errors and all(x["pass"] for x in stationarity.values())
    endpoint_iv = []
    trajectories = []
    for seed, tr in traces.items():
        for key, label in (("a", "endpoint_a"), ("b", "endpoint_b")):
            first = tr[key][0]
            endpoint_iv.append({"id": label, "seed": seed, "Je": expected_jes[label], "V": _mean_window(first["times"], first["V"], 2000, 5000), "T_avg": _mean_window(first["times"], first["T_avg"], 2000, 5000), "psi_avg": _mean_window(first["times"], first["psi_avg"], 2000, 5000)})
            for i, seg in enumerate(tr[key]):
                trajectories.append({"seed": seed, "id": label, "segment": i + 1, "time_midpoint": seg["cumulative_start"] + float(np.mean(seg["times"])), "V": float(np.mean(seg["V"])), "T_avg": float(np.mean(seg["T_avg"])), "psi_avg": float(np.mean(seg["psi_avg"]))})
    result = {"schema": "tdgl-full-curl-ndr-analysis-v1", "model_units": "native model units",
              "protocol": {"segments": EXPECTED_SEGMENTS, "segment_horizon": SEGMENT_HORIZON, "total_horizon": TOTAL_HORIZON, "final_window": [48000, 50000], "seeds_expected": list(EXPECTED_SEEDS)},
              "complete_paired_seeds": complete_seeds, "incomplete": len(complete_seeds) < 8 or bool(errors),
              "endpoint_drop": stats, "stationarity": stationarity,
              "sustained_NDR": {"status": "unresolved", "reason": "endpoint drop alone does not establish sustained NDR; interior/history controls required"},
              "artifact_errors": errors, "artifact_validation": validate_artifacts,
              "endpoint_segment1_iv": endpoint_iv, "late_trajectory": trajectories,
              "screen": analyze_screen(screen_root) if screen_root else None, "pairs": pairs}
    return result


def analyze_screen(root: Path) -> dict[str, Any]:
    groups = read_chains(root); rows = []
    for job, entries in groups.items():
        if not entries: continue
        try:
            s = read_segment(sorted(entries, key=lambda e: int(e.get("segment", 0)))[0]); m = _manifest(entries[0])
            rows.append({"id": job, "seed": entries[0].get("seed"), "Je": entries[0].get("Je"), "V": _mean_window(s["times"], s["V"], 2000, 5000), "T_avg": _mean_window(s["times"], s["T_avg"], 2000, 5000), "psi_avg": _mean_window(s["times"], s["psi_avg"], 2000, 5000), "complete": bool(m.get("complete"))})
        except (OSError, KeyError, ValueError): continue
    return {"horizon": 5000.0, "rows": rows, "scope": "screening only; no stochastic convergence claim"}


def write_outputs(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    def finite(x: Any) -> Any:
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, dict): return {k: finite(v) for k, v in x.items()}
        if isinstance(x, list): return [finite(v) for v in x]
        return x
    result = finite(result)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "endpoint_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        rows = result["pairs"]; w = csv.DictWriter(f, fieldnames=["seed", "V_a", "V_b", "delta_V", "T_avg_a", "T_avg_b", "psi_avg_a", "psi_avg_b", "V_integrated_a", "V_integrated_b", "alias_difference_a", "alias_difference_b"]); w.writeheader(); w.writerows(rows)
    screen = result.get("screen")
    if screen:
        with (out / "screen_iv.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "seed", "Je", "V", "T_avg", "psi_avg", "complete"]); w.writeheader(); w.writerows(screen["rows"])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    screening = list(result["endpoint_segment1_iv"])
    screening.extend((result.get("screen") or {}).get("rows", []))
    currents = sorted({r["Je"] for r in screening})
    if currents:
        samples = [[r["V"] for r in screening if r["Je"] == j] for j in currents]
        means = np.array([np.mean(v) for v in samples])
        ax.errorbar(currents, means, yerr=[means - [min(v) for v in samples],
                                         [max(v) for v in samples] - means], fmt="o-", capsize=3)
        for j, values in zip(currents, samples):
            ax.scatter([j] * len(values), values, s=10, alpha=.35, color="gray")
    ax.set_title("5000τ screening: mean and seed range")
    ax.set_xlabel("Je (model units)"); ax.set_ylabel("V (model units)")
    ax = axes[0, 1]; ps = result["pairs"]
    endpoint_currents = [0.20816326530612245, 0.21122448979591835]
    if ps:
        values = [[p[k] for p in ps] for k in ("V_a", "V_b")]
        means = np.array([np.mean(v) for v in values])
        ax.errorbar(endpoint_currents, means, yerr=[means - [min(v) for v in values],
                                                   [max(v) for v in values] - means], fmt="o-", capsize=4)
    else:
        ax.text(.5, .5, "Awaiting complete 50000τ pairs", ha="center", transform=ax.transAxes)
    ax.set_title("50000τ endpoint contrast: mean and range")
    ax.set_xlabel("Je (model units)"); ax.set_ylabel("V (model units)")
    ax = axes[1, 0]
    ax.axhline(-.01, color="k", lw=1, ls=":", label="Declared drop resolution")
    ax.scatter([p["seed"] - 2026110400 for p in ps], [p["delta_V"] for p in ps])
    upper = result["endpoint_drop"]["ci95_upper_one_sided"]
    if upper is not None:
        ax.axhline(upper, color="r", ls="--", label=f"One-sided 95% upper: {upper:.4g}")
    ax.set_title("Paired-seed ΔV = Vb − Va")
    ax.set_xlabel("Seed suffix"); ax.set_ylabel("ΔV (model units)"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for label, color in (("endpoint_a", "#0072B2"), ("endpoint_b", "#D55E00")):
        rows = [r for r in result["late_trajectory"] if r["id"] == label]
        seeds = sorted({r["seed"] for r in rows})
        for seed in seeds:
            rr = [r for r in rows if r["seed"] == seed]
            ax.plot([r["time_midpoint"] for r in rr], [r["V"] for r in rr], color=color, alpha=.25, lw=.8)
        times = sorted({r["time_midpoint"] for r in rows})
        if times:
            ax.plot(times, [np.mean([r["V"] for r in rows if r["time_midpoint"] == t]) for t in times],
                    "o-", color=color, label=label.replace("_", " "))
    ax.set_title("Retained-window means (2000τ per marker)")
    ax.set_xlabel("Cumulative time (τ)"); ax.set_ylabel("V (model units)")
    if result["late_trajectory"]: ax.legend(fontsize=8)
    for axis in axes.flat: axis.grid(alpha=.2)
    fig.suptitle("Corrected full-curl TDGL: NDR validation", fontsize=14)
    fig.savefig(out / "ndr_summary.png", dpi=180); fig.savefig(out / "ndr_summary.pdf"); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--endpoint-root", required=True, type=Path); p.add_argument("--screen-root", type=Path); p.add_argument("--out", required=True, type=Path); p.add_argument("--validate-artifacts", action="store_true"); a = p.parse_args(argv)
    result = analyze(a.endpoint_root, a.screen_root, a.validate_artifacts); write_outputs(result, a.out); print(json.dumps({"incomplete": result["incomplete"], "complete_paired_seeds": result["complete_paired_seeds"], "out": str(a.out)})); return 0


if __name__ == "__main__": raise SystemExit(main())
