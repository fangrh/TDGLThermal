#!/usr/bin/env python3
"""Receipt-aware analysis of the independent full I-V campaign.

This module intentionally treats every current/seed pair as an independent
hot-start experiment.  It does not infer an ascending or descending sweep.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from analyze_full_curl_ndr import (read_segment, validate_chain_artifacts,
    _complete_chain, _mean_window, _output_path)

SEED = 2026110411
N_BASE = 99
BASE_CURRENTS = tuple(float(x) for x in np.linspace(0.0, 0.3, N_BASE))
_A, _B = .3 * 68 / 98, .3 * 69 / 98
REFINEMENT_CURRENTS = (_A + (_B - _A) / 3, _A + 2 * (_B - _A) / 3)
ALL_CURRENTS = BASE_CURRENTS + REFINEMENT_CURRENTS
TOTAL_HORIZON, SEGMENT_HORIZON, N_SEGMENTS = 50000.0, 5000.0, 10
SAVED_DATASETS = ("times", "V", "psi_avg", "T_avg", "psi_real", "psi_imag", "T", "Ax", "Ay")
PROTOCOL = {"temperature": .18, "noise": .001, "horizon": TOTAL_HORIZON,
            "segment_horizon": SEGMENT_HORIZON, "snapshot_interval": 1.0,
            "skip_ratio": .6, "tolerance": 1e-4, "seeds": [SEED],
            "base_currents": list(BASE_CURRENTS),
            "refinement_currents": list(REFINEMENT_CURRENTS)}


def _finite(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list): return [_finite(v) for v in x]
    return x


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_files(path: Path) -> list[dict[str, Any]]:
    """Read either one chain receipt or a JSON array of receipts."""
    raw = _read_json(path)
    rows = raw if isinstance(raw, list) else [raw]
    out = []
    for row in rows:
        if not isinstance(row, dict): raise ValueError(f"receipt is not an object: {path}")
        x = dict(row); x["_chain_path"] = str(path); x["_segment_dir"] = str(path.parent)
        out.append(x)
    return out


def _manifest(e: dict[str, Any]) -> dict[str, Any]:
    m = e.get("manifest", {})
    return m if isinstance(m, dict) else {}


def _protocol_ok(e: dict[str, Any], je: float) -> bool:
    m = _manifest(e)
    def close(k, v):
        try: return math.isclose(float(m.get(k)), v, abs_tol=1e-12)
        except (TypeError, ValueError): return False
    return (math.isclose(float(e.get("Je", math.nan)), je, abs_tol=1e-12) and
            int(e.get("seed", -1)) == SEED and int(m.get("seed", -1)) == SEED and
            m.get("complete") is True and m.get("status") in ("success", "Success", "ReturnCode.Success") and
            all(close(k, v) for k, v in (("horizon", 5000), ("skip_ratio", .6), ("noise", .001), ("tolerance", 1e-4))) and
            int(m.get("Nx", -1)) == 60 and int(m.get("Ny", -1)) == 30 and
            bool(m.get("save_fields", True)) and m.get("output_schema") == "streamed_hdf5_fields" and
            m.get("precision") in ("ComplexF64/Float64", "Float64") and
            close("final_time", 5000.0) and close("snapshot_interval", 1.0) and
            close("hx", .5) and close("hy", .5) and m.get("solver") == "Tsit5" and
            all(m.get(k) for k in ("config_hash", "model_hash", "rhs_source_hash")))


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    errors = []
    if catalog.get("schema") != "tdgl-full-iv-catalog-v1": errors.append("wrong catalog schema")
    p = catalog.get("protocol", {})
    for key, value in PROTOCOL.items():
        got = p.get(key)
        if key in ("base_currents", "refinement_currents"):
            if not isinstance(got, list) or len(got) != len(value) or any(not math.isclose(float(a), float(b), abs_tol=1e-12) for a, b in zip(got, value)): errors.append(f"protocol {key} mismatch")
        elif key == "seeds":
            if got != value: errors.append(f"protocol {key} mismatch")
        elif got is None or not math.isclose(float(got), float(value), abs_tol=1e-12): errors.append(f"protocol {key} mismatch")
    points = catalog.get("points")
    if not isinstance(points, list) or len(points) != 101: return errors + ["catalog must contain 101 points"]
    seen = set(); ids = set()
    expected = [(x, "base") for x in BASE_CURRENTS] + [(x, "refinement") for x in REFINEMENT_CURRENTS]
    for point, (je, kind) in zip(points, expected):
        try:
            key = (float(point["Je"]), int(point["seed"]))
            if point.get("id") in ids: errors.append(f"duplicate point id {point.get('id')}")
            ids.add(point.get("id"))
            if key in seen: errors.append(f"duplicate point {key}")
            seen.add(key)
            if not math.isclose(key[0], je, abs_tol=1e-12) or point.get("kind") != kind: errors.append(f"grid point mismatch {point}")
            if key[1] != SEED: errors.append(f"seed mismatch {point}")
            if not isinstance(point.get("segments_to_run"), int) or not 0 <= point["segments_to_run"] <= 10: errors.append(f"invalid segments_to_run {point}")
        except (KeyError, TypeError, ValueError): errors.append(f"malformed point {point}")
    return errors


def _discover(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    for path in root.glob("**/chain.json"):
        try: rows.extend(_entry_files(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc: errors.append(f"malformed chain receipt {path}: {exc}")
    return rows, errors


def _merge_entries(entries: list[dict[str, Any]], je: float) -> tuple[list[dict[str, Any]], list[str]]:
    errors, by_end = [], {}
    for e in entries:
        try: end, start = float(e.get("cumulative_end")), float(e.get("cumulative_start"))
        except (TypeError, ValueError): errors.append("missing cumulative_end"); continue
        if not math.isclose(end, round(end / SEGMENT_HORIZON) * SEGMENT_HORIZON, rel_tol=0, abs_tol=1e-8) or not math.isclose(start, end - SEGMENT_HORIZON, rel_tol=0, abs_tol=1e-8): errors.append(f"invalid cumulative interval {start},{end}"); continue
        if not _protocol_ok(e, je): errors.append(f"protocol mismatch {e.get('_chain_path')}")
        x = dict(e); x["segment"] = int(round(end / SEGMENT_HORIZON)); x["cumulative_end"] = end; x["cumulative_start"] = start
        old = by_end.get(end)
        if old is not None:
            # Same logical segment is acceptable only when it is byte-identical in receipt identity.
            def canon(z): return {k: v for k, v in z.items() if k not in ("_chain_path", "_segment_dir")}
            if json.dumps(canon(old), sort_keys=True, default=str) != json.dumps(canon(x), sort_keys=True, default=str): errors.append(f"conflicting duplicate segment {end}")
        else: by_end[end] = x
    rows = [by_end[k] for k in sorted(by_end)]
    if len(rows) == 10 and [x["segment"] for x in rows] == list(range(1, 11)) and not _complete_chain(rows, expected_je=je): errors.append(f"invalid chain {je}")
    return rows, errors


def _finite_fields_errors(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for entry in rows:
        try:
            with h5py.File(_output_path(entry), "r") as h:
                for name in SAVED_DATASETS:
                    dataset = h[name]
                    for start in range(0, dataset.shape[0], 64):
                        if not np.isfinite(dataset[start:start + 64]).all():
                            errors.append(f"nonfinite saved field {name} segment {entry.get('segment')}")
                            break
        except (OSError, KeyError, ValueError, IndexError) as exc:
            errors.append(f"saved-field scan failed: {exc}")
    return errors


def _point_result(je: float, rows: list[dict[str, Any]], errors: list[str], validate: bool) -> dict[str, Any]:
    row = {"Je": je, "seed": SEED, "complete": False, "errors": list(errors)}
    if len(rows) == 10 and not errors and _complete_chain(rows, expected_je=je):
        try:
            segs = [read_segment(x) for x in rows]; final = segs[-1]
            v = _mean_window(final["times"] + 45000, final["V"], 48000, 50000)
            t = _mean_window(final["times"] + 45000, final["T_avg"], 48000, 50000)
            psi = _mean_window(final["times"] + 45000, final["psi_avg"], 48000, 50000)
            prev = _mean_window(segs[-2]["times"] + 40000, segs[-2]["V"], 43000, 45000)
            drift = v - prev
            row.update(complete=all(math.isfinite(x) for x in (v, t, psi, final["V_integrated"], drift)),
                       V=v, T_avg=t, psi_avg=psi, V_integrated=final["V_integrated"],
                       stationarity={"late_minus_previous": drift, "windows": [[43000,45000],[48000,50000]]})
            if validate:
                row["artifact_errors"] = validate_chain_artifacts(rows)
                if not row["artifact_errors"]: row["artifact_errors"] += _finite_fields_errors(rows)
            if row.get("artifact_errors"): row["complete"] = False; row["errors"] += row["artifact_errors"]
        except (OSError, KeyError, ValueError, IndexError) as exc: row["errors"].append(str(exc))
    return row


def analyze(catalog_path: Path, campaign_root: Path, focused_analysis: Path | None = None, validate: bool = False) -> dict[str, Any]:
    catalog = _read_json(catalog_path); errors = validate_catalog(catalog)
    imported: list[dict[str, Any]] = []
    for p in catalog.get("points", []) if isinstance(catalog.get("points"), list) else []:
        for raw in p.get("existing_chain_paths", []) or []:
            try: imported.extend(_entry_files(Path(raw)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc: errors.append(f"import {raw}: {exc}")
    discovered, discover_errors = _discover(campaign_root); errors.extend(discover_errors)
    by_point: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for e in imported + discovered:
        try:
            raw_je = float(e.get("Je")); matches = [j for j in ALL_CURRENTS if abs(j - raw_je) <= 1e-12]
            if len(matches) != 1: errors.append(f"entry Je not in declared grid: {raw_je}")
            else: by_point[matches[0]].append(e)
        except (TypeError, ValueError): errors.append("entry missing Je")
    expected_hashes = {k: catalog.get(k, catalog.get("protocol", {}).get(k)) for k in ("config_hash", "model_hash")}
    provenance = set()
    for e in imported + discovered:
        m = _manifest(e); provenance.add(tuple(m.get(k) for k in ("config_hash", "model_hash", "rhs_source_hash", "precision")))
        for k, expected in expected_hashes.items():
            if expected and m.get(k) != expected: errors.append(f"{k} mismatch catalog for {e.get('_chain_path')}")
    if len(provenance) > 1: errors.append("inconsistent source provenance across points")
    points = []
    for je in ALL_CURRENTS:
        rows, pe = _merge_entries(by_point.get(je, []), je)
        points.append(_point_result(je, rows, pe, validate))
    complete = len(points) == 101 and all(p["complete"] for p in points) and not errors
    focused = None
    if focused_analysis:
        try: focused = _finite(_read_json(focused_analysis))
        except (OSError, ValueError, TypeError) as exc: errors.append(f"focused analysis: {exc}")
    complete = len(points) == 101 and all(p["complete"] for p in points) and not errors
    analysis_files = [Path(__file__), Path(__file__).with_name("analyze_full_curl_ndr.py"),
                      Path(__file__).with_name("compare_results.py")]
    analysis_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in analysis_files}
    return _finite({"schema": "tdgl-full-iv-analysis-v1", "complete": complete, "pending": not complete,
                    "artifact_validation": validate, "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                    "finite_saved_fields_validation": validate, "saved_datasets": list(SAVED_DATASETS),
                    "provenance": {"catalog_path": str(catalog_path.resolve()),
                        "campaign_root": str(campaign_root.resolve()), "expected_physics_hashes": expected_hashes,
                        "observed_physics_sources": [dict(zip(("config_hash", "model_hash", "rhs_source_hash", "precision"), p))
                            for p in sorted(provenance, key=lambda p: tuple(map(str, p)))],
                        "analysis_source_sha256": analysis_sources, "python": sys.version,
                        "numpy": np.__version__,
                        "focused_input_path": str(focused_analysis.resolve()) if focused_analysis else None,
                        "focused_input_sha256": hashlib.sha256(focused_analysis.read_bytes()).hexdigest()
                            if focused_analysis and focused_analysis.is_file() else None},
                    "source_count": {"base": 99, "refinement": 2, "total": 101}, "protocol": PROTOCOL,
                    "complete_point_count": sum(p["complete"] for p in points),
                    "points": points, "errors": errors, "focused_analysis": focused,
                    "focused_overlay_note": "Local n=8 includes the global seed 2026110411; it is not n=9 and does not provide eight-seed full-range coverage.",
                    "claim": "independent hot-start I-V points; no sweep direction inferred"})


def write_outputs(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_iv_analysis.json").write_text(json.dumps(_finite(result), indent=2, allow_nan=False), encoding="utf-8")
    with (out / "full_iv_seed_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Je", "seed", "complete", "V", "T_avg", "psi_avg", "V_integrated", "late_minus_previous"])
        for p in result["points"]: w.writerow([p.get(k) for k in ("Je","seed","complete","V","T_avg","psi_avg","V_integrated")] + [p.get("stationarity",{}).get("late_minus_previous")])
    with (out / "full_iv_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Je", "n", "V_mean", "V_min", "V_max"])
        for p in result["points"]:
            vals = [p["V"]] if p.get("complete") else []
            w.writerow([p["Je"], len(vals), np.mean(vals) if vals else None, np.min(vals) if vals else None, np.max(vals) if vals else None])
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pts = sorted(result["points"], key=lambda p: p["Je"])
        def series(rows, key):
            return np.array([p.get(key, np.nan) if p.get("complete") else np.nan for p in rows], dtype=float)
        xb = np.array([p["Je"] for p in pts]); yb = series(pts, "V")
        complete_label = f"{sum(bool(p.get('complete')) for p in pts)}/101 complete; final 48–50kτ"
        fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True); ax = ax.ravel()
        # NaNs intentionally break the line across pending points.
        ax[0].plot(xb, yb, "o-", ms=2, lw=.8, color="#0072B2", label="Seed 2026110411 · 99 base + 2 refined")
        ax[0].set(xlabel="Je (native model units)", ylabel="V (native model units)", title=f"Full I–V · {complete_label}"); ax[0].legend(fontsize=8)
        ax[0].set_xlim(-.005, .305)
        zoom = [p for p in pts if .2 <= p["Je"] <= .22]
        ax[1].plot([p["Je"] for p in zoom], series(zoom, "V"), "o-", ms=3, color="#0072B2", label="n=1 global")
        ax[1].set(xlabel="Je (native model units)", ylabel="V", title="NDR zoom · Je = 0.200–0.220"); ax[1].legend(fontsize=8)
        ax[1].set_xlim(.2,.22)
        ax[2].plot(xb, series(pts, "T_avg"), color="#009E73"); ax[2].set(xlabel="Je", ylabel="T_avg", title="Final 48–50kτ temperature")
        ax[3].plot(xb, series(pts, "psi_avg"), color="#CC79A7"); ax[3].set(xlabel="Je", ylabel="mean |ψ|²", title="Final 48–50kτ condensate density")
        focused = result.get("focused_analysis")
        if isinstance(focused, dict):
            # Focused analysis schema stores values and stats separately.
            for p in focused.get("points", []):
                st = p.get("stats", {}); vals = p.get("values", [])
                mean = st.get("mean")
                if mean is None and vals: mean = float(np.mean(vals))
                if p.get("Je") is not None and mean is not None:
                    lo, hi = (min(vals), max(vals)) if vals else (mean, mean)
                    ax[1].errorbar(p["Je"], mean, yerr=[[mean-lo], [hi-mean]], fmt="o", color="#D55E00", capsize=3)
            ax[1].plot([], [], "o", color="#D55E00", label="local n=8 mean/range (includes blue seed)"); ax[1].legend(fontsize=8)
        for a in ax: a.grid(alpha=.2)
        fig.suptitle("Corrected full curl · T0 = 0.18 · independent hot preparations", fontsize=13)
        fig.savefig(out / "full_iv.png", dpi=180); fig.savefig(out / "full_iv.pdf"); plt.close(fig)
    except ImportError: pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--catalog", "--catalogJSON", dest="catalog", required=True, type=Path); p.add_argument("--campaign-root", required=True, type=Path); p.add_argument("--focused-analysis", type=Path); p.add_argument("--out", "--outDIR", dest="out", required=True, type=Path); p.add_argument("--validate-artifacts", action="store_true")
    a = p.parse_args(argv); result = analyze(a.catalog, a.campaign_root, a.focused_analysis, a.validate_artifacts); write_outputs(result, a.out); print(json.dumps({"complete": result["complete"], "out": str(a.out)})); return 0 if result["complete"] else 1

if __name__ == "__main__": raise SystemExit(main())
