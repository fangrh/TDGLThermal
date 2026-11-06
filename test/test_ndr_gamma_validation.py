from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import prepare_ndr_gamma_validation as prep
import prepare_ndr_gamma_tune as base
import analyze_ndr_gamma_validation as ar

ROOT = Path(__file__).parents[1]
SCREEN = ROOT / "studies/full_curl_ndr/gamma_fine_tune_20261106/plan/screen_protocol.json"
AMEND = ROOT / "studies/full_curl_ndr/gamma_fine_tune_20261106/amendment_v2/amendment.json"
SEEDS = list(range(2026110611, 2026110619))
CURRENT = [0.20918367346938774, 0.21020408163265306]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _campaign(tmp_path):
    plan = tmp_path / "plan"
    protocol = prep.prepare(SCREEN, AMEND, plan)
    jobs = json.loads((plan / "validation_jobs.json").read_text())
    root = tmp_path / "campaign"; times = np.linspace(3000., 5000., 2001)
    for job in jobs:
        variant, je, seed = job["variant"], job["Je"], job["seed"]
        high = je == CURRENT[1]
        frac = {"baseline": .15, "gamma38": .25, "gamma42": .16}[variant]
        final_v = 1.0 - frac if high else 1.0
        prev_h5 = prev_rng = prev_h5_hash = prev_rng_hash = None
        first_iv = f"initial-{je}-{seed}".encode()
        first_rng_hash = hashlib.sha256(f"fresh RNG {seed}".encode()).hexdigest()
        for seg in range(1, 11):
            d = root / job["id"] / f"segment{seg}"; d.mkdir(parents=True)
            h5 = d / f"output_{seg}.h5"; iv = d / "initial.bin"; rng = d / "rng_after.bin"
            iv.write_bytes(first_iv if seg == 1 else f"iv-{seg}-{job['id']}".encode()); rng.write_bytes(f"rng-{job['id']}-{seg}".encode())
            field = np.ones((times.size, 3, 3))
            with h5py.File(h5, "w") as f:
                for name, data in {"times": times, "V": np.full(times.shape, final_v), "T_avg": np.full(times.shape, .2), "psi_avg": np.full(times.shape, .3), "psi_real": field, "psi_imag": field, "Ax": field, "Ay": field, "T": field}.items(): f.create_dataset(name, data=data, compression="gzip")
            m = {"schema_version":"tdgl-cpu-benchmark-v1","role":"candidate","backend":"cpu_compatible","config_hash":protocol["config_sha256"][variant],"model_hash":base.MODEL_HASH,"rhs_source_hash":base.RHS_HASH,"julia_version":"1.12","machine":"fixture","initial_hash":sha(iv),"seed":seed,"Je":je,"noise":.001,"Nx":60,"Ny":30,"hx":.5,"hy":.5,"horizon":5000.,"tolerance":1e-4,"snapshot_interval":1.,"skip_ratio":.6,"solver":"Tsit5","precision":"ComplexF64/Float64","save_fields":True,"status":"success","complete":True,"final_time":5000.,"wall_seconds":1.,"output_path":str(h5),"output_hash":sha(h5),"initial_vector_path":str(iv),"initial_vector_hash":sha(iv),"output_schema":"streamed_hdf5_fields","preparation":"configured_down","accepted_steps":100,"rejected_steps":0,"rhs_calls":600,"stream_maxiters_override":10000000,"stream_guard_source_hash":base.GUARD_HASH,"initial_state_source":None,"initial_source_hash":None}
            if seg > 1: m["initial_state_source"],m["initial_source_hash"] = str(prev_h5),prev_h5_hash
            e = {"id":job["id"],"variant":variant,"Je":je,"seed":seed,"segment":seg,"cumulative_start":(seg-1)*5000,"cumulative_end":seg*5000,"preceding_hdf5":str(prev_h5) if prev_h5 else None,"preceding_sha256":prev_h5_hash,"manifest":m,"full_field_output":True,"backend":"cpu_compatible","rng_before_sha256":first_rng_hash if seg == 1 else prev_rng_hash,"rng_after_sha256":sha(rng),"rng_after_path":str(rng),"preceding_rng_path":str(prev_rng) if prev_rng else None,"rng_protocol":"seed_case_first_segment" if seg == 1 else "restore_serialized_checkpoint"}
            (d / "chain.json").write_text(json.dumps(e), encoding="utf-8")
            prev_h5,prev_rng,prev_h5_hash,prev_rng_hash = h5,rng,sha(h5),sha(rng)
    return plan / "validation_protocol.json", root


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    protocol, root = _campaign(tmp_path)
    monkeypatch.setattr(ar, "validate_chain_artifacts", lambda *a, **k: [])
    return protocol, root


def test_valid_complete_and_selects_gamma38(fixture):
    protocol, root = fixture
    r = ar.analyze(protocol, root, validate=True)
    assert r["complete"] is True and r["artifact_validation"] is True and r["finite_saved_fields_validation"] is True
    assert len(r["rows"]) == 48 and len(r["segment_rows"]) == 480
    assert r["fresh_initial_identity_checked"] is True and r["selected_variant"] == "gamma38"
    assert r["stationary_branch_confirmed"] is False


def test_unvalidated_has_no_selection(fixture):
    protocol, root = fixture
    r = ar.analyze(protocol, root, validate=False)
    assert r["complete"] is True and r["selected_variant"] is None


@pytest.mark.parametrize("field", ["Je", "config_hash", "stream_guard_source_hash", "rng_after_sha256", "initial_source_hash"])
def test_receipt_tamper_is_reported(fixture, field):
    protocol, root = fixture; chain = next(root.glob("**/segment1/chain.json")); d = json.loads(chain.read_text())
    if field == "Je": d["manifest"][field] = .3
    elif field == "initial_source_hash": d["manifest"][field] = "bad"
    elif field == "rng_after_sha256": d[field] = "bad"
    else: d["manifest"][field] = "bad"
    chain.write_text(json.dumps(d), encoding="utf-8")
    r = ar.analyze(protocol, root, validate=True)
    assert r["complete"] is False and r["errors"]


def test_missing_segment_and_cross_variant_initial_guard(fixture):
    protocol, root = fixture
    chain = next(root.glob("gamma38_i000_seed*/segment1/chain.json")); d = json.loads(chain.read_text()); iv = Path(d["manifest"]["initial_vector_path"]); iv.write_bytes(b"changed"); d["manifest"]["initial_vector_hash"] = sha(iv); chain.write_text(json.dumps(d), encoding="utf-8")
    missing = next(root.glob("**/segment10/chain.json")); missing.unlink()
    r = ar.analyze(protocol, root, validate=True)
    assert r["complete"] is False and r["errors"]


def test_frozen_success_alpha_metadata_tamper_is_reported(fixture):
    protocol, root = fixture
    p = json.loads(protocol.read_text()); p["alpha_per_candidate"] = .05; protocol.write_text(json.dumps(p), encoding="utf-8")
    r = ar.analyze(protocol, root, validate=True)
    assert r["complete"] is False and r["errors"]


def test_two_candidate_one_sided_confidence_quantile():
    from scipy.stats import t
    values=[.20,.21,.19,.22,.20,.21,.19,.22]
    result=ar.get_confidence(values)
    assert result['crit']==pytest.approx(float(t.ppf(1-.05/2,7)))
    assert result['lower']==pytest.approx(np.mean(values)-t.ppf(.975,7)*np.std(values,ddof=1)/np.sqrt(8))
