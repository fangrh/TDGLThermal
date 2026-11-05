#!/usr/bin/env python3
"""Strict noisy field/RNG comparison of frozen and cached-receipt runners."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_full_curl_ndr import validate_chain_artifacts

MODEL = "4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS = "900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"
FROZEN_RUNNER = "eaadd36355d1c1b0582efe91c6d1e047b867c1a4f626de4cfc51fd30fd65a248"
DATASETS = ("times", "V", "T_avg", "psi_avg", "psi_real", "psi_imag", "T", "Ax", "Ay")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def compare_fields(a, b):
    with h5py.File(a, "r") as x, h5py.File(b, "r") as y:
        for name in DATASETS:
            u, v = x[name], y[name]
            if u.shape != v.shape or u.dtype != v.dtype or u.dtype != np.dtype("float64"):
                raise ValueError(f"field shape/Float64 mismatch {name}")
            for i in range(0, u.shape[0], 64):
                aa, bb = u[i:i + 64], v[i:i + 64]
                if not np.isfinite(aa).all() or not np.isfinite(bb).all() or not np.array_equal(aa, bb):
                    raise ValueError(f"nonfinite or unequal field {name}")


def validate_campaign(root, jobs, segments):
    campaign = read(root / "campaign.json")
    if set(campaign["chains"]) != {j["id"] for j in jobs}:
        raise ValueError("campaign job matrix mismatch")
    for job in jobs:
        n = job.get("segments", segments)
        es = campaign["chains"][job["id"]]
        if len(es) != n:
            raise ValueError("unexpected segment count")
        for k, e in enumerate(es, 1):
            m = e["manifest"]
            expected_start = job.get("prior_cumulative_horizon", 0) + (k - 1) * 20
            if e["segment"] != k or e["cumulative_start"] != expected_start or e["cumulative_end"] != expected_start + 20:
                raise ValueError("cumulative time mismatch")
            for key, expected in {"horizon": 20, "final_time": 20, "noise": .001,
                                  "tolerance": 1e-4, "skip_ratio": .6, "snapshot_interval": 1,
                                  "model_hash": MODEL, "rhs_source_hash": RHS,
                                  "backend": "cpu_compatible", "precision": "ComplexF64/Float64",
                                  "solver": "Tsit5", "Nx": 60, "Ny": 30, "hx": .5, "hy": .5,
                                  "seed": job["seed"], "Je": job["Je"], "complete": True}.items():
                if m.get(key) != expected:
                    raise ValueError(f"manifest {key} mismatch")
            e["_chain_path"] = str(root / job["id"] / f"segment{k}" / "chain.json")
            e["_segment_dir"] = str(Path(e["_chain_path"]).parent)
            if k == 1 and job.get("initial_state_path"):
                if m.get("initial_source_hash") != sha(job["initial_state_path"]) or e.get("preceding_sha256") != sha(job["initial_state_path"]):
                    raise ValueError("imported field source mismatch")
                if e["rng_before_sha256"] != sha(job["initial_rng_path"]):
                    raise ValueError("imported RNG source mismatch")
        errors = validate_chain_artifacts(es)
        if errors:
            raise ValueError("artifact validation: " + "; ".join(errors[:4]))
        final = campaign["final_states"][job["id"]]
        if final["segments_run"] != n or final["cumulative_horizon"] != job.get("prior_cumulative_horizon", 0) + 20 * n:
            raise ValueError("final state count/horizon mismatch")
        if n == 0 and (final["lastHDF"] != job["initial_state_path"] or final["lastRNG"] != job["initial_rng_path"]):
            raise ValueError("zero-segment source changed")
    return campaign


def compare_campaigns(a, b, stage):
    rows = []
    for jid in sorted(a["chains"]):
        for old, new in zip(a["chains"][jid], b["chains"][jid]):
            om, nm = old["manifest"], new["manifest"]
            compare_fields(om["output_path"], nm["output_path"])
            if sha(om["initial_vector_path"]) != sha(nm["initial_vector_path"]):
                raise ValueError("initial vector bytes differ")
            if sha(old["rng_after_path"]) != sha(new["rng_after_path"]) or old["rng_before_sha256"] != new["rng_before_sha256"]:
                raise ValueError("RNG states differ")
            rows.append({"stage": stage, "id": jid, "segment": old["segment"],
                         "datasets_exact": list(DATASETS), "rng_before_exact": True,
                         "rng_after_exact": True, "initial_vector_exact": True,
                         "reference_HDF_sha256": sha(om["output_path"]),
                         "candidate_HDF_sha256": sha(nm["output_path"])})
    return rows


def run(args):
    old, new, out = args.reference_checkout.resolve(), args.candidate_checkout.resolve(), args.out.resolve()
    if old == new or out.exists():
        raise ValueError("distinct checkouts and a new output directory are required")
    for repo in (old, new):
        if sha(repo / "current_sweep_thermal_nofastscan.jl") != MODEL or sha(repo / "src/cpu_compatible.jl") != RHS:
            raise ValueError("full-curl source identity mismatch")
    if sha(old / "bench/reproduce_ndr.jl") != FROZEN_RUNNER:
        raise ValueError("reference runner is not the declared frozen version")
    out.mkdir(parents=True)
    cfg = old / "studies/full_curl_ndr/enhancement_20261105/configs/baseline.yaml"
    jobs = [{"id": f"probe_{i}", "Je": je, "seed": 2026110591 + i, "config_path": str(cfg),
             "segments": 2, "Nx": 60, "Ny": 30, "noise": .001} for i, je in enumerate((.2, .21))]
    jp = out / "fresh_jobs.json"; write(jp, jobs)
    env = os.environ.copy()
    overrides = {"JULIA_DEPOT_PATH": args.depot, "GKSwstype": "100", "NOFASTSCAN_SKIP_WORKERS": "1",
                 "NOFASTSCAN_SKIP_MAIN": "1", "OPENBLAS_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1"}
    env.update(overrides)
    calls, records, campaigns = [], [], {}

    def solve(repo, label, stage, jobfile, stage_jobs, segments):
        target = out / label / stage
        log = out / f"{label}_{stage}.log"
        argv = ["timeout", "--signal=TERM", "--kill-after=30s", "900", str(args.julia), "-p", "1",
                "--project=.", "--startup-file=no", "bench/reproduce_ndr.jl", "--jobs", str(jobfile),
                "--out", str(target), "--segments", str(segments), "--horizon", "20", "--skip", ".6",
                "--backend", "cpu_compatible"]
        started = time.monotonic()
        with log.open("wb") as f:
            p = subprocess.run(argv, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        calls.append({"label": label, "stage": stage, "argv": argv, "returncode": p.returncode,
                      "wall_seconds": time.monotonic() - started, "log": str(log)})
        write(out / "calls.json", {"env_overrides": overrides, "calls": calls})
        if p.returncode:
            raise RuntimeError(f"{label}/{stage} failed; see {log}")
        result = validate_campaign(target, stage_jobs, segments)
        print(json.dumps({"completed": f"{label}/{stage}"}), flush=True)
        return result

    for repo, label in ((old, "reference"), (new, "candidate")):
        campaigns[label] = solve(repo, label, "fresh", jp, jobs, 2)
    records += compare_campaigns(campaigns["reference"], campaigns["candidate"], "fresh")
    imported = {}
    for repo, label in ((old, "reference"), (new, "candidate")):
        jj = []
        for job in jobs:
            state = campaigns[label]["final_states"][job["id"]]
            jj.append({**job, "id": "import_" + job["id"], "segments": 1,
                       "prior_cumulative_horizon": 40, "initial_state_path": state["lastHDF"], "initial_rng_path": state["lastRNG"]})
        jj.append({**jj[0], "id": "zero_segment_import", "segments": 0})
        path = out / f"{label}_imported_jobs.json"; write(path, jj)
        imported[label] = solve(repo, label, "imported", path, jj, 1)
    records += compare_campaigns(imported["reference"], imported["candidate"], "imported")
    sources = {label: {p: sha(repo / p) for p in ("current_sweep_thermal_nofastscan.jl", "src/cpu_compatible.jl", "bench/reproduce_ndr.jl", "Project.toml", "Manifest.toml")}
               for label, repo in (("reference", old), ("candidate", new))}
    result = {"schema": "tdgl-receipt-cache-differential-v1", "complete": True, "errors": [],
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "point_segment_pairs": len(records),
              "dataset_pair_comparisons": len(records) * len(DATASETS), "zero_segment_import_checked": True,
              "records": records, "source_hashes": sources, "helper_sha256": sha(new / "bench/ndr_receipt_hashes.jl"),
              "driver_sha256": sha(Path(__file__)), "python": sys.version, "numpy": np.__version__, "h5py": h5py.__version__,
              "scope": "Short noisy fresh/continued/imported trajectories; exact saved-field, initial-vector and RNG checks. No whole-campaign performance claim."}
    write(out / "differential_summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-checkout", required=True, type=Path)
    p.add_argument("--candidate-checkout", required=True, type=Path)
    p.add_argument("--julia", required=True, type=Path)
    p.add_argument("--depot", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"complete": False, "error": str(exc)}), flush=True)
        return 1
    print(json.dumps({"complete": result["complete"], "point_segment_pairs": result["point_segment_pairs"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
