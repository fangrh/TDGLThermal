import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import prepare_ndr_holdout as ph


def _screen_files(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_bytes(b"baseline")
    jobs = [{"id": "baseline_i000_seed2026110501", "config_path": str(cfg), "Je": .18, "seed": 2026110501}]
    jp = tmp_path / "screen_jobs.json"; jp.write_text(json.dumps(jobs), encoding="utf-8")
    protocol = {"schema": "tdgl-full-curl-ndr-enhancement-screen-v1", "backend": "cpu_compatible",
        "currents": [.18, .19], "seeds": [2026110501, 2026110502],
        "variants": [{"id": "baseline", "config": "cfg.yaml"}, {"id": "candidate", "config": "cfg.yaml"}],
        "config_sha256": {"baseline": ph._sha(cfg), "candidate": ph._sha(cfg)},
        "normalized_source_hashes": {"model": ph.MODEL_HASH, "rhs": ph.RHS_HASH},
        "jobs": {"path": jp.name, "jobs_sha256": ph._sha(jp)}}
    pp = tmp_path / "screen_protocol.json"; pp.write_text(json.dumps(protocol), encoding="utf-8")
    analysis = {"schema": "tdgl-ndr-enhancement-analysis-v1", "complete": True, "artifact_validation": True, "finite_saved_fields_validation": True, "errors": [],
        "provenance": {"protocol_path": str(pp), "protocol_sha256": ph._sha(pp), "jobs_sha256": ph._sha(jp), "config_sha256": {"baseline": ph._sha(cfg), "candidate": ph._sha(cfg)}, "analysis_source_sha256": {name: ph._sha(Path(ph.__file__).parent / name) for name in ("analyze_ndr_enhancement.py", "analyze_full_curl_ndr.py", "analyze_full_iv.py", "compare_results.py")}},
        "baseline_best_score": .1, "selected_candidate": {"variant": "candidate", "eligible": True, "score": .2, "Je_left": .18, "Je_right": .19},
        "pair_outcomes": [{"variant": "baseline", "Je_left": .18, "Je_right": .19, "eligible": True, "score": .1, "seeds": [{"seed": 2026110501, "V_left": 1, "absolute_drop": .1, "fraction": .1}, {"seed": 2026110502, "V_left": 1, "absolute_drop": .1, "fraction": .1}]}, {"variant": "candidate", "Je_left": .18, "Je_right": .19, "eligible": True, "score": .2, "seeds": [{"seed": 2026110501, "V_left": 1, "absolute_drop": .2, "fraction": .2}, {"seed": 2026110502, "V_left": 1, "absolute_drop": .2, "fraction": .2}]}]}
    for pair in analysis["pair_outcomes"]:
        for seed in pair["seeds"]:
            seed["V_right"] = seed["V_left"] - seed["absolute_drop"]
    ap = tmp_path / "analysis.json"; ap.write_text(json.dumps(analysis), encoding="utf-8")
    return ap, pp


def test_selected_bracket_must_be_declared_adjacent():
    with pytest.raises(ValueError, match="exact adjacent"):
        ph._selected_bracket({"selected_candidate": {"variant": "x", "eligible": True, "score": 1, "Je_left": .18, "Je_right": .205}, "baseline_best_score": 0, "pair_outcomes": []}, [.18, .19, .21])


def test_corrupted_verified_flags_are_rejected(tmp_path):
    ap, pp = _screen_files(tmp_path)
    analysis = json.loads(ap.read_text()); analysis["artifact_validation"] = False; ap.write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match="both artifact validations"):
        ph.freeze_holdout(ap, pp, tmp_path / "out")


def test_seed_reuse_is_rejected(tmp_path):
    ap, pp = _screen_files(tmp_path)
    protocol = json.loads(pp.read_text()); protocol["seeds"] = [2026110511, 2026110502]; pp.write_text(json.dumps(protocol))
    analysis = json.loads(ap.read_text()); analysis["provenance"]["protocol_sha256"] = ph._sha(pp); ap.write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match="screen seeds"):
        ph.freeze_holdout(ap, pp, tmp_path / "out")


def test_atomic_output_refuses_existing_directory(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(FileExistsError):
        ph._write_atomic(out, {"x.json": {"ok": True}})


def test_successful_freeze_writes_56_jobs_and_primary_contrast(tmp_path):
    ap, pp = _screen_files(tmp_path)
    out = tmp_path / "holdout"
    hp = ph.freeze_holdout(ap, pp, out)
    jobs = json.loads((out / "holdout_jobs.json").read_text())
    assert len(jobs) == 56 and hp["jobs"]["count"] == 56
    assert hp["primary_contrast"]["only"] is True
    assert hp["jobs"]["jobs_sha256"] == hashlib.sha256((json.dumps(jobs, indent=2) + "\n").encode()).hexdigest()
    assert hp["jobs"]["jobs_sha256"] == ph._sha(out / "holdout_jobs.json")


def _history_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ap, pp = _screen_files(tmp_path)
    holdout = tmp_path / "holdout"
    hp = ph.freeze_holdout(ap, pp, holdout)
    jobs = json.loads((holdout / "holdout_jobs.json").read_text())
    campaign = {"chains": {}}
    for job in jobs:
        vals = []
        previous_output = previous_rng = None
        for seg in range(1, 11):
            folder = tmp_path / "raw" / job["id"] / f"segment{seg}"; folder.mkdir(parents=True)
            h5 = folder / "out.h5"; h5.write_bytes(f"{job['id']}/{seg}".encode())
            rng = folder / "rng_after.bin"; rng.write_bytes(f"rng/{job['id']}/{seg}".encode())
            oh, rh = ph._sha(h5), ph._sha(rng)
            manifest = {"output_path": str(h5), "output_hash": oh, "model_hash": ph.MODEL_HASH, "config_hash": hp["source"]["screen_config_sha256"]["candidate"], "rhs_source_hash": ph.RHS_HASH,
                        "solver":"Tsit5","precision":"Float64","save_fields":True,"output_schema":"streamed_hdf5_fields",
                        "Je": job["Je"], "seed": job["seed"], "horizon": 5000, "final_time": 5000, "snapshot_interval": 1, "skip_ratio": .6, "tolerance": 1e-4, "noise": .001, "Nx": 60, "Ny": 30, "complete": True, "status": "success", "backend": "cpu_compatible", "initial_source_hash": previous_output}
            if seg == 10:
                (folder / "out.json").write_text(json.dumps(manifest), encoding="utf-8")
            vals.append({"id": job["id"], "Je": job["Je"], "seed": job["seed"], "segment": seg, "cumulative_start": (seg-1)*5000, "cumulative_end": seg*5000,
                         "manifest": manifest, "preceding_sha256": previous_output, "rng_before_sha256": previous_rng or ("a"*64), "rng_after_sha256": rh, "rng_after_path": str(rng)})
            previous_output, previous_rng = oh, rh
        campaign["chains"][job["id"]] = vals
    cp = tmp_path / "campaign.json"; cp.write_text(json.dumps(campaign), encoding="utf-8")
    return holdout / "holdout_protocol.json", cp


def test_history_freeze_validates_complete_matrix_and_outer_sources(tmp_path):
    hp, cp = _history_fixture(tmp_path)
    out = tmp_path / "history"
    result = ph.freeze_history(cp, hp, out)
    jobs = json.loads((out / "history_jobs.json").read_text())
    assert result["jobs"]["count"] == 8 and len(jobs) == 8
    assert result["validation_scope"].startswith("all 56x10")


def test_history_freeze_rejects_missing_inner_job(tmp_path):
    hp, cp = _history_fixture(tmp_path)
    campaign = json.loads(cp.read_text()); campaign["chains"].pop(next(k for k in campaign["chains"] if "i003" in k)); cp.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="exact holdout job matrix"):
        ph.freeze_history(cp, hp, tmp_path / "history")


def test_history_freeze_rejects_wrong_source_je_and_rng_hash(tmp_path):
    hp, cp = _history_fixture(tmp_path)
    campaign = json.loads(cp.read_text()); key = next(k for k in campaign["chains"] if "i000_seed2026110511" in k)
    campaign["chains"][key][-1]["Je"] = .19; cp.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout identity mismatch"):
        ph.freeze_history(cp, hp, tmp_path / "history1")
    hp, cp = _history_fixture(tmp_path / "second")
    campaign = json.loads(cp.read_text()); key = next(k for k in campaign["chains"] if "i000_seed2026110511" in k)
    campaign["chains"][key][-1]["rng_after_sha256"] = "b" * 64; cp.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="outer source hash mismatch"):
        ph.freeze_history(cp, hp, tmp_path / "history2")


@pytest.mark.parametrize("key", ["protocol_sha256", "jobs_sha256", "config_sha256", "analysis_source_sha256"])
def test_missing_provenance_rejected(tmp_path, key):
    ap, pp = _screen_files(tmp_path)
    analysis = json.loads(ap.read_text()); analysis["provenance"].pop(key)
    ap.write_text(json.dumps(analysis))
    with pytest.raises(ValueError, match="provenance|hash"):
        ph.freeze_holdout(ap, pp, tmp_path / "out")


def test_no_baseline_drop_can_advance_verified_candidate(tmp_path):
    ap, pp = _screen_files(tmp_path)
    analysis = json.loads(ap.read_text()); analysis["pair_outcomes"] = analysis["pair_outcomes"][1:]
    analysis["baseline_best_score"] = 0
    ap.write_text(json.dumps(analysis))
    hp = ph.freeze_holdout(ap, pp, tmp_path / "out")
    assert hp["selected_candidate"]["baseline_best_score"] == 0


@pytest.mark.parametrize("kind", ["seed", "fraction", "score"])
def test_forged_selected_pair_rejected(tmp_path, kind):
    ap, pp = _screen_files(tmp_path); analysis = json.loads(ap.read_text())
    pair = analysis["pair_outcomes"][1]
    if kind == "seed": pair["seeds"][1]["seed"] = pair["seeds"][0]["seed"]
    elif kind == "fraction": pair["seeds"][0]["fraction"] = .9
    else: analysis["selected_candidate"]["score"] = .9
    ap.write_text(json.dumps(analysis))
    with pytest.raises(ValueError): ph.freeze_holdout(ap, pp, tmp_path / "out")
