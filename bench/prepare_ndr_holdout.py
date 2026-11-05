#!/usr/bin/env python3
"""Freeze a verified enhancement-screen winner into a holdout plan.

This module deliberately only prepares immutable job/protocol JSON.  It never
launches a solver and refuses to write into an existing output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

MODEL_HASH = "4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422"
RHS_HASH = "900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe"
SCREEN_SEEDS = {2026110501, 2026110502}
HOLDOUT_SEEDS = list(range(2026110511, 2026110519))
HISTORY_SEEDS = list(range(2026110511, 2026110515))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_atomic(out: Path, files: dict[str, Any]) -> None:
    if out.exists():
        raise FileExistsError(f"refusing existing output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=str(out.parent)))
    try:
        for name, value in files.items():
            (tmp / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, out)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _protocol_jobs(protocol_path: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    raw = protocol.get("jobs", {})
    rel = raw.get("path", "screen_jobs.json") if isinstance(raw, dict) else raw
    path = Path(rel)
    if not path.is_absolute():
        path = protocol_path.parent / path
    return _read(path), path.resolve()


def _require_screen_provenance(analysis_path: Path, protocol_path: Path, analysis: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, dict[str, str]]:
    if analysis.get("schema") != "tdgl-ndr-enhancement-analysis-v1":
        raise ValueError("screen analysis schema mismatch")
    if protocol.get("schema") != "tdgl-full-curl-ndr-enhancement-screen-v1":
        raise ValueError("screen protocol schema mismatch")
    if analysis.get("complete") is not True:
        raise ValueError("screen analysis is not complete")
    if analysis.get("artifact_validation") is not True or analysis.get("finite_saved_fields_validation") is not True:
        raise ValueError("screen analysis lacks both artifact validations")
    if analysis.get("errors") != []:
        raise ValueError("screen analysis contains errors")
    ap = analysis.get("provenance", {})
    if not ap.get("protocol_path") or Path(ap["protocol_path"]).resolve() != protocol_path.resolve():
        raise ValueError("screen analysis protocol provenance mismatch")
    if not ap.get("protocol_sha256") or ap["protocol_sha256"] != _sha(protocol_path):
        raise ValueError("screen analysis protocol hash mismatch")
    jobs, jobs_path = _protocol_jobs(protocol_path, protocol)
    declared_jobs = protocol.get("jobs_sha256") or protocol.get("jobs", {}).get("jobs_sha256")
    if not declared_jobs or declared_jobs != _sha(jobs_path):
        raise ValueError("screen jobs hash mismatch")
    if not ap.get("jobs_sha256") or ap["jobs_sha256"] != _sha(jobs_path):
        raise ValueError("screen analysis jobs provenance mismatch")
    if protocol.get("normalized_source_hashes") != {"model": MODEL_HASH, "rhs": RHS_HASH}:
        raise ValueError("screen source hashes mismatch")
    if not isinstance(ap.get("config_sha256"), dict):
        raise ValueError("screen analysis config provenance missing")
    source_hashes = ap.get("analysis_source_sha256", {})
    required_sources = {"analyze_ndr_enhancement.py", "analyze_full_curl_ndr.py", "analyze_full_iv.py", "compare_results.py"}
    if set(source_hashes) != required_sources or any(
        not (Path(__file__).parent / name).is_file() or _sha(Path(__file__).parent / name) != value
        for name, value in source_hashes.items()
    ):
        raise ValueError("screen analysis code provenance mismatch")
    variants = {v["id"]: v for v in protocol.get("variants", [])}
    hashes = protocol.get("config_sha256", {})
    if set(hashes) != set(variants):
        raise ValueError("screen config hash coverage mismatch")
    for vid, variant in variants.items():
        cfg = (protocol_path.parent / variant["config"]).resolve()
        if not cfg.is_file() or _sha(cfg) != hashes[vid]:
            raise ValueError(f"screen config provenance mismatch: {vid}")
        if ap["config_sha256"].get(vid) != hashes[vid]:
            raise ValueError(f"screen analysis config hash mismatch: {vid}")
    return jobs, jobs_path, hashes


def _verified_pair_score(pair: dict[str, Any]) -> float:
    seeds = pair.get("seeds", [])
    if len(seeds) != 2 or {int(s.get("seed", -1)) for s in seeds} != SCREEN_SEEDS:
        raise ValueError("selected bracket does not pass both declared screen seeds")
    fractions = []
    for s in seeds:
        vl, vr = float(s.get("V_left", math.nan)), float(s.get("V_right", math.nan))
        drop = vl - vr
        if not all(math.isfinite(x) for x in (vl, vr)) or vl < .05 or drop <= .01:
            raise ValueError("selected bracket does not pass both declared screen seeds")
        fraction = drop / vl
        if not math.isclose(float(s.get("absolute_drop", math.nan)), drop, rel_tol=0, abs_tol=1e-12) or not math.isclose(float(s.get("fraction", math.nan)), fraction, rel_tol=0, abs_tol=1e-12):
            raise ValueError("pair values are inconsistent with voltage outcomes")
        fractions.append(fraction)
    score = min(fractions)
    if not math.isclose(float(pair.get("score", math.nan)), score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("pair score is inconsistent with per-seed fractions")
    return score


def _selected_bracket(analysis: dict[str, Any], currents: list[float]) -> tuple[str, float, float, float]:
    selected = analysis.get("selected_candidate")
    if not isinstance(selected, dict) or selected.get("variant") == "baseline" or not selected.get("eligible"):
        raise ValueError("no eligible nonbaseline selected candidate")
    baseline_pairs = [x for x in analysis.get("pair_outcomes", []) if x.get("variant") == "baseline" and x.get("eligible") is True]
    baseline = max((_verified_pair_score(x) for x in baseline_pairs), default=0.0)
    if not math.isclose(float(analysis.get("baseline_best_score", math.nan)), baseline, rel_tol=0, abs_tol=1e-12):
        raise ValueError("analysis baseline score provenance mismatch")
    score = float(selected.get("score", math.nan))
    if not math.isfinite(score) or not score > baseline:
        raise ValueError("selected candidate does not exceed baseline best score")
    lo, hi = float(selected["Je_left"]), float(selected["Je_right"])
    if not any(math.isclose(lo, a, rel_tol=0, abs_tol=1e-15) and math.isclose(hi, b, rel_tol=0, abs_tol=1e-15) for a, b in zip(currents, currents[1:])):
        raise ValueError("selected bracket is not an exact adjacent declared pair")
    pair = next((x for x in analysis.get("pair_outcomes", [])
                 if x.get("variant") == selected.get("variant")
                 and math.isclose(float(x.get("Je_left", math.nan)), lo, rel_tol=0, abs_tol=1e-15)
                 and math.isclose(float(x.get("Je_right", math.nan)), hi, rel_tol=0, abs_tol=1e-15)), None)
    if not isinstance(pair, dict) or pair.get("eligible") is not True:
        raise ValueError("selected bracket lacks a verified eligible pair outcome")
    expected_score = _verified_pair_score(pair)
    if not math.isclose(float(pair.get("score", math.nan)), expected_score, rel_tol=0, abs_tol=1e-12) or not math.isclose(score, expected_score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("selected score is inconsistent with per-seed fractions")
    return str(selected["variant"]), lo, hi, score


def freeze_holdout(screen_analysis: Path, screen_protocol: Path, out: Path) -> dict[str, Any]:
    analysis, protocol = _read(screen_analysis), _read(screen_protocol)
    jobs, jobs_path, hashes = _require_screen_provenance(screen_analysis, screen_protocol, analysis, protocol)
    currents = [float(x) for x in protocol["currents"]]
    if len(protocol["seeds"]) != 2 or set(int(x) for x in protocol["seeds"]) != SCREEN_SEEDS:
        raise ValueError("screen seeds do not match declared two-seed screen")
    if set(HOLDOUT_SEEDS) & SCREEN_SEEDS:
        raise ValueError("holdout seed reuse")
    variant, lo, hi, score = _selected_bracket(analysis, currents)
    step = (hi - lo) / 4.0
    holdout_currents = [lo - step, lo, lo + step, lo + 2 * step, lo + 3 * step, hi, hi + step]
    if len(set(holdout_currents)) != 7 or any(x < -1e-15 or x > 0.3 + 1e-15 for x in holdout_currents):
        raise ValueError("derived holdout currents are invalid")
    variants = {v["id"]: v for v in protocol["variants"]}
    if variant not in variants:
        raise ValueError("selected variant is not declared")
    cfg = (screen_protocol.parent / variants[variant]["config"]).resolve()
    if not cfg.is_file() or _sha(cfg) != hashes[variant]:
        raise ValueError("selected config provenance mismatch")
    holdout_jobs = []
    for i, je in enumerate(holdout_currents):
        for seed in HOLDOUT_SEEDS:
            holdout_jobs.append({"id": f"{variant}_holdout_i{i:03d}_seed{seed}", "config_path": str(cfg), "Je": je,
                "seed": seed, "segments": 10, "Nx": 60, "Ny": 30, "noise": .001,
                "prior_cumulative_horizon": 0, "preparation": "fresh_hot_start", "initial_state_path": None, "initial_rng_path": None})
    hp = {"schema": "tdgl-full-curl-ndr-holdout-v1", "source": {"screen_analysis": str(screen_analysis.resolve()), "screen_analysis_sha256": _sha(screen_analysis), "screen_analysis_source_sha256": analysis.get("provenance", {}).get("analysis_source_sha256", {}), "screen_protocol": str(screen_protocol.resolve()), "screen_protocol_sha256": _sha(screen_protocol), "screen_jobs": str(jobs_path), "screen_jobs_sha256": _sha(jobs_path), "screen_config_sha256": hashes, "model_hash": MODEL_HASH, "rhs_source_hash": RHS_HASH, "freeze_builder_sha256": _sha(Path(__file__))},
          "selected_candidate": {"variant": variant, "score": score, "screen_bracket": [lo, hi], "baseline_best_score": float(analysis["baseline_best_score"])},
          "currents": holdout_currents, "seeds": HOLDOUT_SEEDS, "jobs": {"path": "holdout_jobs.json", "count": 56, "jobs_sha256": hashlib.sha256((json.dumps(holdout_jobs, indent=2, allow_nan=False) + "\n").encode()).hexdigest(), "segments": 10, "segment_horizon": 5000, "total_horizon": 50000, "snapshot_interval": 1, "skip_ratio": .6, "tolerance": 1e-4, "backend": "cpu_compatible", "precision": "Float64", "integrator": "Tsit5"},
          "primary_contrast": {"only": True, "lo": lo, "hi": hi, "final_window": [48000, 50000], "prior_window": [43000, 45000]},
          "gates": {"mean_paired_fraction_min": .20, "nominal_student_one_sided_95_lower_gt": .15, "paired_V_hi_minus_V_lo_upper": "< -0.01", "stationarity_ci_abs_max": .0025, "stationarity_scope": "descriptive if mixed"},
          "scope": "frozen holdout; primary contrast is the selected screen bracket, not a best holdout interval"}
    _write_atomic(out, {"holdout_jobs.json": holdout_jobs, "holdout_protocol.json": hp})
    return hp


def _chain_entries(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    chains = campaign.get("chains", {})
    return [e for values in chains.values() for e in values]


def freeze_history(campaign_path: Path, holdout_protocol_path: Path, out: Path) -> dict[str, Any]:
    hp = _read(holdout_protocol_path); campaign = _read(campaign_path)
    if hp.get("schema") != "tdgl-full-curl-ndr-holdout-v1": raise ValueError("holdout protocol schema mismatch")
    holdout_jobs_path = holdout_protocol_path.parent / hp["jobs"]["path"]
    holdout_jobs = _read(holdout_jobs_path)
    exact_holdout_hash = _sha(holdout_jobs_path)
    if hp["jobs"].get("jobs_sha256") != exact_holdout_hash: raise ValueError("holdout jobs hash mismatch")
    currents = [float(x) for x in hp["currents"]]
    expected_ids = {j["id"] for j in holdout_jobs}
    if len(holdout_jobs) != 56 or len(expected_ids) != 56: raise ValueError("holdout matrix is not exactly 56 jobs")
    combos=[(float(j["Je"]),int(j["seed"])) for j in holdout_jobs]
    expected_combos={(j,s) for j in currents for s in HOLDOUT_SEEDS}
    if len(currents)!=7 or len(set(combos))!=56 or set(combos)!=expected_combos: raise ValueError("holdout current/seed matrix mismatch")
    selected_variant = hp["selected_candidate"]["variant"]
    cfg_hash = hp["source"]["screen_config_sha256"][selected_variant]
    config_paths = {str(Path(j["config_path"]).resolve()) for j in holdout_jobs}
    if len(config_paths) != 1 or _sha(Path(next(iter(config_paths)))) != cfg_hash: raise ValueError("holdout selected config mismatch")
    source_cfg = Path(next(iter(config_paths)))
    chains = campaign.get("chains", {})
    if not isinstance(chains, dict): raise ValueError("campaign chains missing")
    entries_by_id = {jid: sorted(vals, key=lambda e: int(e.get("segment", 0))) for jid, vals in chains.items()}
    if set(entries_by_id) != expected_ids: raise ValueError("campaign does not contain exact holdout job matrix")
    outer = {}
    for job in holdout_jobs:
        jid, seed, je = job["id"], int(job["seed"]), float(job["Je"])
        es = entries_by_id[jid]
        if len(es) != 10 or [int(e.get("segment", -1)) for e in es] != list(range(1, 11)):
            raise ValueError(f"incomplete holdout chain: {jid}")
        previous_output = previous_rng = None
        for i, e in enumerate(es, 1):
            m = e.get("manifest", {})
            if int(e.get("seed", -1)) != seed or not math.isclose(float(e.get("Je", math.nan)), je, rel_tol=0, abs_tol=1e-12): raise ValueError(f"holdout identity mismatch: {jid}")
            if int(m.get("seed",-1))!=seed or not math.isclose(float(m.get("Je",math.nan)),je,rel_tol=0,abs_tol=1e-12):raise ValueError(f"holdout manifest identity mismatch: {jid}")
            if int(e.get("cumulative_start", -1)) != (i - 1) * 5000 or int(e.get("cumulative_end", -1)) != i * 5000: raise ValueError(f"holdout continuity mismatch: {jid}")
            for key, val in (("horizon", 5000), ("final_time", 5000), ("snapshot_interval", 1), ("skip_ratio", .6), ("tolerance", 1e-4), ("noise", .001), ("Nx", 60), ("Ny", 30)):
                if not math.isclose(float(m.get(key, math.nan)), val, rel_tol=0, abs_tol=1e-9): raise ValueError(f"holdout {key} mismatch: {jid}")
            if m.get("complete") is not True or m.get("status") not in ("success", "Success", "ReturnCode.Success"): raise ValueError(f"holdout success mismatch: {jid}")
            if m.get("solver")!="Tsit5" or m.get("precision") not in ("Float64","ComplexF64/Float64") or m.get("save_fields") is not True or m.get("output_schema")!="streamed_hdf5_fields":raise ValueError(f"holdout numeric/output mismatch: {jid}")
            if m.get("config_hash") != cfg_hash or m.get("model_hash") != MODEL_HASH or m.get("rhs_source_hash") != RHS_HASH or m.get("backend") != "cpu_compatible": raise ValueError(f"holdout source mismatch: {jid}")
            if not Path(m.get("output_path", "")).is_file() or not Path(e.get("rng_after_path", "")).is_file(): raise ValueError(f"holdout output/RNG missing: {jid}")
            if i > 1 and (e.get("preceding_sha256") != previous_output or e.get("rng_before_sha256") != previous_rng or m.get("initial_source_hash") != previous_output): raise ValueError(f"holdout link mismatch: {jid}")
            previous_output, previous_rng = m.get("output_hash"), e.get("rng_after_sha256")
            if i == 10 and (math.isclose(je, currents[0], abs_tol=1e-12) or math.isclose(je, currents[-1], abs_tol=1e-12)):
                outer[(seed, round(je, 15))] = e
    if len(outer) != 16: raise ValueError("missing 16 outer history sources")
    for (seed, _), e in outer.items():
        m=e["manifest"]; h5=Path(m["output_path"]); rng=Path(e["rng_after_path"]); side=Path(str(h5).replace(".h5", ".json"))
        if _sha(h5) != m.get("output_hash") or _sha(rng) != e.get("rng_after_sha256"): raise ValueError("outer source hash mismatch")
        if not side.is_file(): raise ValueError("outer sidecar missing")
        sm=_read(side)
        for key in ("output_path", "output_hash", "model_hash", "config_hash", "rhs_source_hash", "Je", "seed"):
            if sm.get(key) != m.get(key): raise ValueError(f"outer sidecar identity mismatch: {key}")
    jobs=[]
    for seed in HISTORY_SEEDS:
        for direction, seq, source in (("ascending", currents, currents[0]), ("descending", list(reversed(currents)), currents[-1])):
            e=outer[(seed,round(source,15))]
            jobs.append({"id":f"{direction}_seed{seed}","config_path":str(source_cfg),"seed":seed,"currents":seq,"initial_state_path":str(Path(e["manifest"]["output_path"]).resolve()),"initial_rng_path":str(Path(e["rng_after_path"]).resolve()),"prior_cumulative_horizon":50000,"Nx":60,"Ny":30,"noise":.001,"source_Je":source})
    hist_jobs_hash=hashlib.sha256((json.dumps(jobs, indent=2) + "\n").encode()).hexdigest()
    histp={"schema":"tdgl-full-curl-ndr-history-plan-v1","holdout_protocol":str(holdout_protocol_path.resolve()),"holdout_protocol_sha256":_sha(holdout_protocol_path),"holdout_jobs_sha256":exact_holdout_hash,"holdout_campaign":str(campaign_path.resolve()),"campaign_sha256":_sha(campaign_path),"source_hashes":{"config":cfg_hash,"model":MODEL_HASH,"rhs_source":RHS_HASH},"backend":"cpu_compatible","tolerance":1e-4,"confidence":{"bonferroni_contrasts":12,"stationarity_ci_abs_max":.0025},"source_Je":{"ascending":currents[0],"descending":currents[-1]},"seeds":HISTORY_SEEDS,"directions":["ascending","descending"],"currents":currents,"jobs":{"path":"history_jobs.json","count":8,"jobs_sha256":hist_jobs_hash,"segment_horizon":20000,"segments_equivalent_5000":224,"equivalent_tau":1120000,"snapshot_interval":1,"skip_ratio":.6,"prior_cumulative_horizon":50000},"raw_window":[12000,20000],"analysis_final_window":[18000,20000],"drift_windows":[[13000,15000],[18000,20000]],"validation_scope":"all 56x10 metadata/continuity checked; HDF5 content hash checked for 16 outer initialization sources only","scope":"ordered-history descriptive control; not a primary contrast"}
    _write_atomic(out,{"history_jobs.json":jobs,"history_protocol.json":histp}); return histp


def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_mutually_exclusive_group(required=True); sub.add_argument("--screen-analysis",type=Path); sub.add_argument("--history-from",type=Path)
    p.add_argument("--screen-protocol",type=Path); p.add_argument("--holdout-protocol",type=Path); p.add_argument("--out",required=True,type=Path); a=p.parse_args(argv)
    if a.screen_analysis:
        if not a.screen_protocol: p.error("--screen-protocol is required with --screen-analysis")
        freeze_holdout(a.screen_analysis,a.screen_protocol,a.out)
    else:
        if not a.holdout_protocol: p.error("--holdout-protocol is required with --history-from")
        freeze_history(a.history_from,a.holdout_protocol,a.out)
    return 0


if __name__ == "__main__": raise SystemExit(main())
