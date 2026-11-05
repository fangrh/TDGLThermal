#!/usr/bin/env python3
"""Prepare an immutable local-current state-response campaign (no solver run)."""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_HASH = "4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS_HASH = "900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"
CONFIG_HASH = "d8c8a70d618b1bd17519552f483890d55fb1b0110b69a44bfdf2ad0591deca12"
GUARD_HASH = "9c764ecaf9a58078394abc2f39651a7a5a07f054761e55c3ceeccfade9b6fa70"
SEEDS = (2026110411, 2026110412, 2026110413, 2026110414)
SOURCE = {"inner1": 0.20918367346938774, "inner2": 0.21020408163265306}


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_atomic(out: Path, files: dict[str, Any]) -> None:
    if out.exists():
        raise FileExistsError(f"refusing existing output directory: {out}")
    parent = out.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(parent)))
    try:
        for name, value in files.items():
            (tmp / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, out)
    except Exception:
        if (tmp.exists() and tmp.is_dir() and tmp.parent.resolve() == parent
                and tmp.name.startswith(f".{out.name}.")):
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _source(root: Path, config: Path, label: str, seed: int, je: float) -> dict[str, Any]:
    chain_path = root / f"screen_{label}_seed{seed}" / "segment9" / "chain.json"
    if not chain_path.is_file():
        raise ValueError(f"missing source chain: {chain_path}")
    chain = _read(chain_path)
    m = chain.get("manifest", {})
    if chain.get("segment") != 9 or int(chain.get("seed", -1)) != seed:
        raise ValueError(f"source identity mismatch: {chain_path}")
    def close(value: Any, expected: float, tol: float = 1e-12) -> bool:
        try:
            x = float(value)
            return math.isfinite(x) and math.isclose(x, expected, rel_tol=0, abs_tol=tol)
        except (TypeError, ValueError):
            return False
    if not close(chain.get("Je"), je, 1e-15) or not close(m.get("Je"), je, 1e-15):
        raise ValueError(f"source Je mismatch: {chain_path}")
    for k, v, tol in (("noise", .001, 1e-12), ("horizon", 5000., 1e-12), ("tolerance", 1e-4, 1e-12),
                      ("Nx", 60, 0), ("Ny", 30, 0), ("hx", .5, 1e-12), ("hy", .5, 1e-12),
                      ("final_time", 5000., 1e-12), ("snapshot_interval", 1., 1e-12), ("skip_ratio", .6, 1e-12)):
        if not close(m.get(k), v, tol):
            raise ValueError(f"source {k} mismatch: {chain_path}")
    if not close(chain.get("cumulative_start"), 45000., 1e-12) or not close(chain.get("cumulative_end"), 50000., 1e-12):
        raise ValueError(f"source cumulative age mismatch: {chain_path}")
    if int(m.get("seed", -1)) != seed or m.get("complete") is not True or m.get("status") not in ("success", "Success", "ReturnCode.Success"):
        raise ValueError(f"source completion mismatch: {chain_path}")
    if m.get("config_hash") != CONFIG_HASH or m.get("model_hash") != MODEL_HASH or m.get("rhs_source_hash") != RHS_HASH:
        raise ValueError(f"source provenance hash mismatch: {chain_path}")
    if (m.get("precision") not in ("Float64", "ComplexF64/Float64") or m.get("solver") != "Tsit5"
            or m.get("backend") != "cpu_compatible" or m.get("save_fields") is not True
            or m.get("output_schema") != "streamed_hdf5_fields"):
        raise ValueError(f"source solver metadata mismatch: {chain_path}")
    h5 = Path(str(m.get("output_path", "")))
    if not h5.is_file() or _sha(h5) != m.get("output_hash"):
        raise ValueError(f"source HDF5 hash mismatch: {chain_path}")
    side = h5.with_suffix(".json")
    if not side.is_file() or _read(side) != m:
        raise ValueError(f"source sidecar mismatch: {chain_path}")
    rng = Path(str(chain.get("rng_after_path", "")))
    if not rng.is_file() or chain.get("rng_after_sha256") != _sha(rng):
        raise ValueError(f"source RNG hash mismatch: {chain_path}")
    return {"source_label": label, "seed": seed, "Je": je, "hdf5_path": str(h5.resolve()),
            "output_hash": m["output_hash"], "sidecar_path": str(side.resolve()), "sidecar_sha256": _sha(side),
            "rng_path": str(rng.resolve()), "rng_sha256": _sha(rng), "chain_path": str(chain_path.resolve()),
            "chain_sha256": _sha(chain_path)}


def prepare(baseline_root: Path, config: Path, out: Path) -> dict[str, Any]:
    if out.exists(): raise FileExistsError(f"refusing existing output directory: {out}")
    if _sha(config) != CONFIG_HASH: raise ValueError("config SHA-256 mismatch")
    sources = [_source(baseline_root, config, label, seed, je) for label, je in SOURCE.items() for seed in SEEDS]
    by = {(x["source_label"], x["seed"]): x for x in sources}
    delta = (SOURCE["inner2"] - SOURCE["inner1"]) / 4.0
    jobs = []
    for label, source_je in SOURCE.items():
        for arm, factor in (("minus", -1), ("sham", 0), ("plus", 1)):
            target = source_je + factor * delta
            for seed in SEEDS:
                s = by[(label, seed)]
                jobs.append({"id": f"{label}_{arm}_seed{seed}", "source_label": label, "arm": arm,
                    "source_Je": source_je, "target_Je": target, "currents": [source_je] + [target] * 4,
                    "config_path": str(config.resolve()), "seed": seed, "segments": 5, "Nx": 60, "Ny": 30,
                    "noise": .001, "prior_cumulative_horizon": 50000, "initial_state_path": s["hdf5_path"],
                    "initial_rng_path": s["rng_path"]})
    jobs_text = json.dumps(jobs, indent=2, allow_nan=False) + "\n"
    jobs_hash = hashlib.sha256(jobs_text.encode()).hexdigest()
    protocol = {"schema": "tdgl-full-curl-local-response-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_currents": SOURCE, "arms": {"minus": -1, "sham": 0, "plus": 1}, "delta": delta, "seeds": list(SEEDS),
        "source_hashes": {"config": CONFIG_HASH, "model": MODEL_HASH, "rhs_source": RHS_HASH}, "backend": "cpu_compatible",
        "tolerance": .0001, "solver_guard": {"maxiters_override": 10000000, "fail_fast": True, "source_sha256": GUARD_HASH},
        "jobs": {"path": "state_response_jobs.json", "jobs_sha256": jobs_hash, "count": 24, "steps_per_job": 5,
                 "segment_horizon": 5000, "snapshot_interval": 1, "skip_ratio": .6, "prior_cumulative_horizon": 50000,
                 "total_segments": 120, "total_tau": 600000}, "sources": sources,
        "confidence": {"bonferroni_contrasts": 4, "stationarity_ci_abs_max": .0025, "negative_delta_threshold": -.01},
        "final_window": [18000, 20000], "drift_windows": [[13000, 15000], [18000, 20000]],
        "scope": "exploratory reused prior seeds; no continuous-branch or confirmation claim",
        "preparer_source_sha256": _sha(Path(__file__)),
        "note": "First conditioning step is 50000-55000; four fixed 5000 blocks follow at target current. No new physical search or fresh seeds within the 120-segment budget."}
    _write_atomic(out, {"state_response_jobs.json": json.loads(jobs_text), "protocol.json": protocol})
    return protocol


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--baseline-root", required=True); p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    args = p.parse_args(); prepare(Path(args.baseline_root).resolve(), Path(args.config).resolve(), Path(args.out).resolve())


if __name__ == "__main__": main()
