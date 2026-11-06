#!/usr/bin/env python3
"""Prepare the prospectively amended, outcome-blind gamma validation plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_ndr_gamma_tune as prep

SCHEMA = "tdgl-full-curl-gamma-fresh-validation-v2"
SEEDS = list(range(2026110611, 2026110619))
SUCCESS = {"all_eight_drops_positive": True, "mean_fraction_min": .20, "fraction_lower_min": .15,
           "paired_improvement_lower_min": 0, "deltaV_upper_max": -.01, "drift_confidence": .95,
           "drift_ci_bound": .0025}
WINNER_RULE = "Among candidates meeting every criterion: largest paired-improvement lower bound, then larger mean fraction, then gamma38."


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_screen(protocol_path: Path, amendment_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read(protocol_path); amendment = _read(amendment_path)
    if protocol.get("schema") != "tdgl-full-curl-gamma-fine-tune-screen-v1":
        raise ValueError("screen protocol schema mismatch")
    if amendment.get("schema") != "tdgl-full-curl-gamma-validation-amendment-v2":
        raise ValueError("amendment schema mismatch")
    if amendment.get("screen_protocol_sha256") != _sha(protocol_path):
        raise ValueError("screen protocol hash mismatch")
    if amendment.get("screen_protocol_v1_preserved") is not True:
        raise ValueError("amendment does not preserve v1 screen")
    if amendment.get('currents') != prep.CURRENTS[1:3]:
        raise ValueError('fixed primary currents mismatch')
    if amendment.get('success') != SUCCESS or amendment.get('winner_rule') != WINNER_RULE:
        raise ValueError('frozen success or winner rules mismatch')
    if (amendment.get('fresh_total_segments') != 480 or amendment.get('total_budget_equivalent_segments') != 880
            or amendment.get('early_stop') is not False):
        raise ValueError('allocation/budget/early-stop rules mismatch')
    context = amendment.get('blinding_context', {})
    if context.get('screen_campaign_complete') is not False or context.get('screen_analysis_exists') is not False:
        raise ValueError('missing prospective outcome-blind context')
    if amendment.get("currents") != [protocol.get("primary", {}).get("Je_low"), protocol.get("primary", {}).get("Je_high")]:
        raise ValueError("amendment currents do not match frozen primary")
    if amendment.get("gamma") != [38, 40, 42] or amendment.get("seeds") != SEEDS:
        raise ValueError("amendment gamma/seeds mismatch")
    if amendment.get("fresh_jobs") != 48 or amendment.get("segments_per_job") != 10:
        raise ValueError("amendment allocation mismatch")
    if amendment.get("alpha_family") != 0.05 or amendment.get("candidate_comparisons") != 2 or amendment.get("alpha_per_candidate") != 0.025:
        raise ValueError("amendment alpha design mismatch")
    if protocol.get("source_hashes") != {"model": prep.MODEL_HASH, "rhs": prep.RHS_HASH}:
        raise ValueError("screen source hashes mismatch")
    if protocol.get("solver_guard") != {"maxiters_override": 10000000, "source_sha256": prep.GUARD_HASH, "fail_fast": True}:
        raise ValueError("screen solver guard mismatch")
    if protocol.get("primary") != {"Je_low": amendment["currents"][0], "Je_high": amendment["currents"][1]}:
        raise ValueError("screen primary mismatch")
    variants = protocol.get("variants", [])
    if variants != [{"id": v, "gamma": g, "config": f"configs/{v}.yaml"} for v,g in prep.VARIANTS]:
        raise ValueError("screen gamma variants mismatch")
    if protocol.get("currents") != [0.20816326530612245, 0.20918367346938774, 0.21020408163265306, 0.21122448979591835] or protocol.get("seeds") != [2026110601, 2026110602]:
        raise ValueError("screen exploratory matrix mismatch")
    return protocol, amendment


def _validate_configs(screen_path: Path, protocol: dict[str, Any]) -> dict[str, bytes]:
    configs: dict[str, bytes] = {}
    for variant in protocol["variants"]:
        name = variant["id"]; path = (screen_path.parent / variant["config"]).resolve()
        if not path.is_file() or _sha(path) != protocol["config_sha256"].get(name):
            raise ValueError(f"screen config provenance mismatch: {name}")
        configs[name] = path.read_bytes()
    if _sha_bytes(configs["baseline"]) != prep.BASE_HASH or b"\r" in configs["baseline"]:
        raise ValueError("baseline config hash or line ending mismatch")
    base = prep._load_yaml_from_bytes(configs["baseline"])
    prep._validate_baseline(base)
    for variant, gamma in (("gamma38", 38.0), ("gamma42", 42.0)):
        expected = prep._variant_bytes(configs["baseline"], gamma)
        if configs[variant] != expected or prep._without_gamma(prep._load_yaml_from_bytes(configs[variant])) != prep._without_gamma(base):
            raise ValueError(f"{variant} config differs outside tdgl.gamma")
    return configs


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prepare(screen_protocol: Path, amendment: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f'refusing existing output directory: {out}')
    screen_protocol = screen_protocol.resolve(); amendment = amendment.resolve()
    protocol, amend = _require_screen(screen_protocol, amendment)
    configs = _validate_configs(screen_protocol, protocol)
    jobs = []
    currents = list(amend["currents"])
    for variant in protocol["variants"]:
        vid = variant["id"]
        cfg = (out.resolve() / "configs" / f"{vid}.yaml")
        for i, je in enumerate(currents):
            for seed in SEEDS:
                jobs.append({"id": f"{vid}_i{i:03d}_seed{seed}", "variant": vid, "config_path": str(cfg), "Je": je, "seed": seed,
                             "segments": 10, "Nx": 60, "Ny": 30, "noise": .001, "prior_cumulative_horizon": 0,
                             "preparation": "fresh_hot"})
    jobs_text = json.dumps(jobs, indent=2, allow_nan=False) + "\n"
    source_hashes = {"model": prep.MODEL_HASH, "rhs": prep.RHS_HASH}
    guard = {"maxiters_override": 10000000, "source_sha256": prep.GUARD_HASH, "fail_fast": True}
    out_protocol = {
        "schema": SCHEMA, "screen_protocol_path": str(screen_protocol), "screen_protocol_sha256": _sha(screen_protocol),
        "amendment_path": str(amendment), "amendment_sha256": _sha(amendment),
        "variants": protocol["variants"], "config_sha256": protocol["config_sha256"], "source_hashes": source_hashes,
        "solver_guard": guard, "backend": protocol["backend"], "tolerance": protocol["tolerance"],
        "snapshot_interval": protocol["snapshot_interval"], "skip_ratio": protocol["skip_ratio"],
        "final_window": protocol["final_window"], "drift_windows": protocol["drift_windows"], "currents": currents, "seeds": SEEDS,
        "primary": protocol["primary"], "jobs": {"path": "validation_jobs.json", "jobs_sha256": _sha_bytes(jobs_text.encode()), "count": 48, "segments": 10, "horizon": 5000, "total_segments": 480},
        "success": amend["success"], "alpha_family": amend["alpha_family"], "candidate_comparisons": amend["candidate_comparisons"], "alpha_per_candidate": amend["alpha_per_candidate"],
        "winner_rule": amend["winner_rule"], "missing_run_policy": amend["missing_run_policy"], "early_stop": amend["early_stop"],
        "confidence_rules": amend["confidence_rules"], "estimand": amend["estimand"], "conditional_ordered": amend["conditional_ordered"],
        "total_budget_equivalent_segments": amend["total_budget_equivalent_segments"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "amendment_created_utc": amend["created_utc"], "preparer_source_sha256": _sha(Path(__file__)), "preparer_version": "v1",
        "scope": amend["scope"],
    }
    files: dict[str, bytes | str] = {"validation_jobs.json": jobs_text, "validation_protocol.json": json.dumps(out_protocol, indent=2, allow_nan=False) + "\n"}
    for variant in protocol["variants"]:
        files[f"configs/{variant['id']}.yaml"] = configs[variant["id"]]
    prep._write_atomic(out, files)
    return out_protocol


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--screen-protocol", required=True, type=Path); p.add_argument("--amendment", required=True, type=Path); p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv); prepare(a.screen_protocol, a.amendment, a.out); return 0


if __name__ == "__main__":
    raise SystemExit(main())
