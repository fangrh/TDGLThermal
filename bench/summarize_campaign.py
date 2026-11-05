#!/usr/bin/env python3
"""Summarize and gate a completed ``run_cpu_campaign.py`` output root.

This is stdlib-only and reads manifests plus compare stdout logs; it never
opens the raw HDF5 files.  A nonzero exit means the receipt is not valid.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any

FIELDS = ("times", "V", "psi_avg", "T_avg", "psi_real", "psi_imag", "T", "Ax", "Ay")
REQUIRED_MANIFEST = ("schema_version", "role", "backend", "config_hash", "model_hash",
                     "rhs_source_hash", "julia_version", "machine", "initial_hash",
                     "seed", "Je", "noise", "Nx", "Ny", "hx", "hy", "horizon",
                     "tolerance", "snapshot_interval", "skip_ratio", "solver", "precision",
                     "save_fields", "status", "complete", "final_time", "wall_seconds",
                     "output_path", "output_hash", "initial_vector_path", "initial_vector_hash",
                     "output_schema")


def _load(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _manifest_path(cmd: list[str], flag: str) -> pathlib.Path:
    try:
        return pathlib.Path(cmd[cmd.index(flag) + 1])
    except (ValueError, IndexError):
        raise ValueError(f"comparison command lacks {flag}")


def _local_manifest(runroot: pathlib.Path, recorded: pathlib.Path) -> pathlib.Path:
    if recorded.is_file():
        return recorded
    candidate = runroot / recorded.relative_to(runroot) if recorded.is_absolute() and str(recorded).startswith(str(runroot)) else None
    if candidate and candidate.is_file():
        return candidate
    return runroot / recorded.name


def _safe_command(cmd: list[str]) -> list[str]:
    # Campaign metadata is retained, but any accidental user@host token is redacted.
    return [re.sub(r"(?<=//)[^/\s]+@[^/\s]+", "[redacted-host]", str(x)) for x in cmd]


def summarize(runroot: pathlib.Path) -> dict[str, Any]:
    summary_path = runroot / "campaign_summary.json"
    summary = _load(summary_path)
    failures: list[str] = []
    args = summary.get("arguments", {})
    jobs = summary.get("jobs")
    if not isinstance(jobs, list):
        failures.append("campaign_summary.jobs is missing or not a list")
        jobs = []
    if len(jobs) != 24:
        failures.append(f"unexpected job count: {len(jobs)} (expected 24)")
    counts = summary.get("counts", {})
    if counts.get("complete") != 24 or any(counts.get(k, 0) for k in ("failed", "incomplete", "deferred")):
        failures.append(f"campaign counts are not 24 complete and zero failed/incomplete/deferred: {counts}")
    expected = {(float(j) , 2026110401 + i) for j in (0.06, 0.18, 0.27) for i in range(8)}
    seen: set[tuple[float, int]] = set()
    pairs: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        current, seed = float(job.get("current", float("nan"))), int(job.get("seed", -1))
        key = (current, seed)
        if key in seen:
            failures.append(f"duplicate job pair {key}")
        seen.add(key)
        if job.get("status") != "complete":
            failures.append(f"job {key} status={job.get('status')}")
        segments = job.get("segments")
        if not isinstance(segments, list) or len(segments) != 1:
            failures.append(f"job {key} does not have exactly one complete segment")
            continue
        seg = segments[0]
        if seg.get("returncode") != 0 or seg.get("timed_out"):
            failures.append(f"job {key} benchmark segment failed")
        if seg.get("compare_returncode") != 0 or seg.get("compare_timed_out"):
            failures.append(f"job {key} compare process failed")
        cmd = seg.get("compare_command")
        if not isinstance(cmd, list):
            failures.append(f"job {key} missing compare command")
            continue
        log_path = runroot / f"Je{current:g}" / f"seed{seed}" / "segment1" / "compare.stdout.log"
        if not log_path.is_file():
            failures.append(f"missing comparison log for {key}: {log_path}")
            continue
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        try:
            report = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:
            report = None
        if not isinstance(report, dict) or report.get("passed") is not True or report.get("errors"):
            failures.append(f"comparison failure for {key}: {report}")
        ferr = report.get("field_errors", {}) if isinstance(report, dict) else {}
        if set(ferr) != set(FIELDS) or any(float(ferr.get(k, float("nan"))) != 0.0 for k in FIELDS):
            failures.append(f"field errors are not exactly nine zeros for {key}")
        rp = _local_manifest(runroot, _manifest_path(cmd, "--reference"))
        cp = _local_manifest(runroot, _manifest_path(cmd, "--candidate"))
        manifests = []
        for label, path, role, backend in (("reference", rp, "corrected_reference", "reference"), ("candidate", cp, "compatible_candidate", "cpu_compatible")):
            if not path.is_file():
                failures.append(f"missing {label} manifest for {key}: {path}")
                continue
            m = _load(path)
            missing = [k for k in REQUIRED_MANIFEST if k not in m]
            if missing:
                failures.append(f"{label} manifest missing fields for {key}: {missing}")
            if m.get("backend") != backend or m.get("role") != role or m.get("status") not in ("success", "Success", "ReturnCode.Success") or not m.get("complete"):
                failures.append(f"invalid {label} manifest status/role for {key}")
            protocol = {"horizon": 5000.0, "final_time": 5000.0, "noise": 0.001,
                        "Nx": 60, "Ny": 30, "hx": 0.5, "hy": 0.5,
                        "tolerance": 1e-4, "skip_ratio": 0.9, "snapshot_interval": 1.0,
                        "seed": seed, "Je": current, "solver": "Tsit5",
                        "precision": "ComplexF64/Float64", "save_fields": True}
            for field, expected_value in protocol.items():
                if m.get(field) != expected_value:
                    failures.append(f"{label} production protocol mismatch {field}: {m.get(field)} != {expected_value} for {key}")
            if not math.isfinite(float(m.get("wall_seconds", float("nan")))) or float(m.get("wall_seconds", 0)) <= 0:
                failures.append(f"invalid {label} wall time for {key}")
            manifests.append({"label": label, "path": str(path), "metadata": {k: m.get(k) for k in REQUIRED_MANIFEST}, "settings": {k: m.get(k) for k in ("schema_version", "config_hash", "model_hash", "rhs_source_hash", "julia_version", "machine", "seed", "Je", "noise", "Nx", "Ny", "hx", "hy", "horizon", "tolerance", "snapshot_interval", "skip_ratio", "solver", "precision", "save_fields", "output_schema")}, "inputs": {"initial_hash": m.get("initial_hash"), "initial_vector_hash": m.get("initial_vector_hash"), "initial_vector_path": m.get("initial_vector_path")}, "output": {"output_path": m.get("output_path"), "output_hash": m.get("output_hash")}, "performance": {"wall_seconds": m.get("wall_seconds"), "rhs_seconds": m.get("rhs_seconds"), "accepted_steps": m.get("accepted_steps"), "rejected_steps": m.get("rejected_steps"), "rhs_calls": m.get("rhs_calls")}})
        if len(manifests) == 2:
            a,b = (x["metadata"] for x in manifests)
            for field in ("config_hash","model_hash","rhs_source_hash","julia_version","machine","initial_hash","initial_vector_hash","output_schema"):
                if a.get(field) != b.get(field):
                    failures.append(f"paired provenance mismatch {field} for {key}")
            pa,pb = (x["performance"] for x in manifests)
            for field in ("rhs_calls","accepted_steps","rejected_steps"):
                if pa.get(field) != pb.get(field):
                    failures.append(f"paired solver statistics mismatch {field} for {key}")
            pairs.append({"current": current, "seed": seed, "segment": seg.get("segment"), "comparison_log": str(log_path), "comparison": {"field_errors": ferr, "speedup_reference_over_candidate": report.get("speedup"), "errors": report.get("errors", [])}, "timing": {"job_elapsed_seconds": job.get("elapsed_seconds"), "segment_elapsed_seconds": seg.get("elapsed_seconds"), "compare_elapsed_seconds": seg.get("compare_elapsed_seconds")}, "manifests": manifests, "commands": {"benchmark": _safe_command(seg.get("command", [])), "compare": _safe_command(cmd)}})
    if seen != expected:
        failures.append(f"pair matrix mismatch: observed {len(seen)}, expected 24 combinations")
    if len(pairs) != 24:
        failures.append(f"matched pair receipt count is {len(pairs)}, expected 24")
    return {"schema": "tdgl-cpu-campaign-artifacts-v1", "valid": not failures, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "campaign_root": str(runroot), "campaign_summary": {"started_at": summary.get("started_at"), "ended_at": summary.get("ended_at"), "elapsed_seconds": summary.get("elapsed_seconds"), "counts": counts, "arguments": args, "performance_phase": summary.get("performance_phase"), "environment": {k: v for k, v in summary.get("environment", {}).items() if k != "ssh_host"}}, "expected": {"pair_count": 24, "currents": [0.06, 0.18, 0.27], "seeds": 8, "horizon": 5000.0, "noise": 0.001, "field_names": list(FIELDS)}, "counts": {"matched_pairs": len(pairs), "comparison_logs": len(pairs), "field_error_vectors": len(pairs), "field_errors_zero": sum(all(float(p["comparison"]["field_errors"].get(k, float("nan"))) == 0.0 for k in FIELDS) for p in pairs)}, "pairs": sorted(pairs, key=lambda p: (p["current"], p["seed"])), "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.runroot)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"schema": "tdgl-cpu-campaign-artifacts-v1", "valid": False, "failures": [str(exc)], "pairs": []}
    output = args.output or (args.runroot / "production_5000_artifacts.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result.get("valid"), "matched_pairs": result.get("counts", {}).get("matched_pairs", 0), "output": str(output), "failures": len(result.get("failures", []))}))
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
