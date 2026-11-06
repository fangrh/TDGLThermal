#!/usr/bin/env python3
"""Prepare the immutable gamma fine-tuning screen (no solver launch)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_HASH = "d8c8a70d618b1bd17519552f483890d55fb1b0110b69a44bfdf2ad0591deca12"
MODEL_HASH = "4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS_HASH = "900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"
GUARD_HASH = "9c764ecaf9a58078394abc2f39651a7a5a07f054761e55c3ceeccfade9b6fa70"
CURRENTS = [0.20816326530612245, 0.20918367346938774, 0.21020408163265306, 0.21122448979591835]
SEEDS = [2026110601, 2026110602]
VARIANTS = [("baseline", 40.0), ("gamma38", 38.0), ("gamma42", 42.0)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    return yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    value = _yaml().safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("base config must parse to a dictionary")
    return value


def _canonical_base(path: Path) -> bytes:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValueError("base config must use canonical LF line endings")
    if hashlib.sha256(raw).hexdigest() != BASE_HASH:
        raise ValueError("base config hash mismatch")
    return raw


def _variant_bytes(base: bytes, gamma: float) -> bytes:
    text = base.decode("utf-8")
    matches = list(re.finditer(r"(?m)^  gamma: 40\.0\n", text))
    if len(matches) != 1:
        raise ValueError("base config must contain exactly one canonical gamma line")
    replacement = f"  gamma: {gamma:.1f}\n"
    return (text[: matches[0].start()] + replacement + text[matches[0].end() :]).encode("utf-8")


def _without_gamma(value: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(value))
    out["tdgl"].pop("gamma", None)
    return out


def _validate_baseline(value: dict[str, Any]) -> None:
    expected = {
        "grid": {"Nx": 60, "Ny": 30, "hx": 0.5, "hy": 0.5},
        "tdgl": {"kappa": 2.0, "sigma1": 1.0, "sigma2": 1.0, "sigma3": 1.0, "u": 5.79,
                 "Tc1": 1.0, "Tc2": 1.0, "Tc3": 1.0, "Tc1gap": 0.2, "Tc3gap": 0.0},
        "thermal": {"enabled": True, "C_eff": 1.0, "k_eff": 5.0, "T0": 0.18, "epsilon_type": "linear",
                    "eta_env": 10.0, "eta_hole": 0.1, "holewidth": 20.0},
        "fields": {"He": 0.0},
        "boundary": {"bc_x": "periodic", "bc_y": "neumann", "bc_y_coverage": 0.88, "loop": False, "field_side": "both", "psi_y_zero": False},
        "noise": {"strength": 0.001},
        "sweep": {"psi_down_init": 0.0, "T_initial_down": 1.2},
    }
    for section, fields in expected.items():
        actual = value.get(section)
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in fields.items()):
            raise ValueError(f"base config has unexpected {section} physics or initialization")


def _write_atomic(out: Path, files: dict[str, bytes | str]) -> None:
    target = out.resolve()
    if target.exists():
        raise FileExistsError(f"refusing existing output directory: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(parent))).resolve()
    if tmp.parent != parent or not tmp.name.startswith(f".{target.name}."):
        raise RuntimeError("unsafe temporary output path")
    try:
        for name, value in files.items():
            path = (tmp / name).resolve()
            if tmp not in path.parents:
                raise ValueError("output file escapes temporary directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = value.encode("utf-8") if isinstance(value, str) else value
            path.write_bytes(data)
        os.replace(str(tmp), str(target))
    except Exception:
        if tmp.parent == parent and tmp.name.startswith(f".{target.name}."):
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def prepare(base_config: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing existing output directory: {out}")
    base_config = base_config.resolve()
    base = _canonical_base(base_config)
    parsed = _load_yaml(base_config)
    _validate_baseline(parsed)
    if parsed.get("sweep", {}).get("psi_down_init") != 0.0 or parsed.get("sweep", {}).get("T_initial_down") != 1.2:
        raise ValueError("base config lacks required fresh configured-hot initialization")
    configs: dict[str, bytes] = {"baseline": base}
    for variant, gamma in VARIANTS[1:]:
        configs[variant] = _variant_bytes(base, gamma)
    parsed_variants = [_load_yaml_from_bytes(data) for data in configs.values()]
    reference = _without_gamma(parsed_variants[0])
    if any(_without_gamma(item) != reference for item in parsed_variants[1:]):
        raise ValueError("gamma variants differ from baseline outside tdgl.gamma")
    jobs = []
    for variant, _ in VARIANTS:
        cfg_abs = (out.resolve() / "configs" / f"{variant}.yaml")
        for i, je in enumerate(CURRENTS):
            for seed in SEEDS:
                jobs.append({"id": f"{variant}_i{i:03d}_seed{seed}", "variant": variant,
                             "config_path": str(cfg_abs), "Je": je, "seed": seed,
                             "segments": 10, "Nx": 60, "Ny": 30, "noise": 0.001,
                             "prior_cumulative_horizon": 0, "preparation": "fresh_configured_hot"})
    config_hashes = {v: hashlib.sha256(configs[v]).hexdigest() for v, _ in VARIANTS}
    protocol = {
        "schema": "tdgl-full-curl-gamma-fine-tune-screen-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variants": [{"id": v, "gamma": g, "config": f"configs/{v}.yaml"} for v, g in VARIANTS],
        "currents": CURRENTS, "seeds": SEEDS, "config_sha256": config_hashes,
        "source_hashes": {"model": MODEL_HASH, "rhs": RHS_HASH},
        "solver_guard": {"maxiters_override": 10000000, "source_sha256": GUARD_HASH, "fail_fast": True},
        "backend": "cpu_compatible", "tolerance": 1e-4,
        "jobs": {"path": "screen_jobs.json", "jobs_sha256": "", "count": 24, "segments": 10, "horizon": 5000, "total_segments": 240},
        "snapshot_interval": 1, "skip_ratio": 0.6, "final_window": [48000, 50000],
        "drift_windows": [[43000, 45000], [48000, 50000]],
        "primary": {"Je_low": CURRENTS[1], "Je_high": CURRENTS[2]},
        "eligibility": {"positive_drop_each_seed": True, "positive_improvement_each_seed": True, "absolute_drop_min": 0.01, "low_voltage_min": 0.05, "drift_ci_bound": 0.0025, "rank": "minimum paired fractional improvement over contemporaneous baseline"},
        "conditional_confirmation": {"configs": 2, "points": 3, "seeds": 8, "segments": 10, "total_segments": 480, "mean_fraction_min": 0.20, "fraction_lower95_min": 0.15, "paired_improvement_lower95_min": 0, "drift_ci_bound": 0.0025, "deltaV_upper95_max": -0.01},
        "conditional_ordered": {"seeds": 4, "directions": 2, "currents": 5, "dwell": 20000, "total_equivalent_segments": 160},
        "budget_total_equivalent_segments": 880,
        "preparer": {"source_sha256": _sha(Path(__file__)), "version": "v1", "python": sys.version, "pyyaml": _yaml().__version__, "scope": "immutable gamma 38/40/42 screen preparation only"},
        "preparer_source_sha256": _sha(Path(__file__)), "preparer_version": "v1",
        "scope": "all other J pairs exploratory and cannot replace primary",
    }
    jobs_text = json.dumps(jobs, indent=2, allow_nan=False) + "\n"
    protocol["jobs"]["jobs_sha256"] = hashlib.sha256(jobs_text.encode()).hexdigest()
    files: dict[str, bytes | str] = {"screen_jobs.json": jobs_text, "screen_protocol.json": json.dumps(protocol, indent=2, allow_nan=False) + "\n"}
    for v, _ in VARIANTS:
        files[f"configs/{v}.yaml"] = configs[v]
    _write_atomic(out, files)
    return protocol


def _load_yaml_from_bytes(data: bytes) -> dict[str, Any]:
    value = _yaml().safe_load(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("generated config is not a dictionary")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    prepare(args.base_config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
