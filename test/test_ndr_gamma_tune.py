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
import prepare_ndr_gamma_tune as prep
import analyze_ndr_gamma_tune as ar

BASE = Path(__file__).parents[1] / "studies/full_curl_ndr/enhancement_20261105/configs/baseline.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_campaign(tmp_path: Path):
    base = tmp_path / "baseline.yaml"
    shutil.copyfile(BASE, base)
    plan = tmp_path / "plan"
    protocol = prep.prepare(base, plan)
    jobs = json.loads((plan / "screen_jobs.json").read_text())
    campaign = tmp_path / "campaign"
    times = np.linspace(3000.0, 5000.0, 2001)
    for job in jobs:
        variant, je, seed = job["variant"], float(job["Je"]), int(job["seed"])
        idx = prep.CURRENTS.index(je)
        final_v = ({"baseline": 1.0, "gamma38": 1.0, "gamma42": 1.0}[variant] if idx == 1 else
                    {"baseline": .85, "gamma38": .75, "gamma42": .84}[variant] if idx == 2 else 1.0 - .01 * idx)
        previous_h5 = previous_rng = previous_h5_hash = None
        previous_rng_hash = hashlib.sha256(f"fresh RNG {seed}".encode()).hexdigest()
        first_iv = f"initial-{idx}-{seed}".encode()
        for segment in range(1, 11):
            d = campaign / job["id"] / f"segment{segment}"; d.mkdir(parents=True)
            h5 = d / f"cpu_compatible_Je{je}_seed{seed}_seg{segment}.h5"
            iv = d / "initial.bin"; rng = d / "rng_after.bin"
            iv.write_bytes(first_iv if segment == 1 else f"iv-{segment}-{job['id']}".encode()); rng.write_bytes(f"rng-{job['id']}-{segment}".encode())
            field = np.ones((times.size, 3, 3))
            with h5py.File(h5, "w") as f:
                for name, data in {"times": times, "V": np.full(times.shape, final_v), "T_avg": np.full(times.shape, .2), "psi_avg": np.full(times.shape, .3), "psi_real": field, "psi_imag": field, "Ax": field, "Ay": field, "T": field}.items():
                    f.create_dataset(name, data=data, compression="gzip")
            m = {"schema_version": "tdgl-cpu-benchmark-v1", "role": "candidate", "backend": "cpu_compatible", "config_hash": protocol["config_sha256"][variant], "model_hash": prep.MODEL_HASH, "rhs_source_hash": prep.RHS_HASH, "julia_version": "1.12", "machine": "fixture", "initial_hash": sha(iv), "seed": seed, "Je": je, "noise": .001, "Nx": 60, "Ny": 30, "hx": .5, "hy": .5, "horizon": 5000., "tolerance": 1e-4, "snapshot_interval": 1., "skip_ratio": .6, "solver": "Tsit5", "precision": "ComplexF64/Float64", "save_fields": True, "status": "success", "complete": True, "final_time": 5000., "wall_seconds": 1., "output_path": str(h5), "output_hash": sha(h5), "initial_vector_path": str(iv), "initial_vector_hash": sha(iv), "output_schema": "streamed_hdf5_fields", "initial_state_source": None, "initial_source_hash": None, "preparation": "configured_down", "accepted_steps": 100, "rejected_steps": 0, "rhs_calls": 600, "stream_maxiters_override": 10000000, "stream_guard_source_hash": prep.GUARD_HASH}
            if segment > 1: m["initial_state_source"], m["initial_source_hash"] = str(previous_h5), previous_h5_hash
            entry = {"id": job["id"], "variant": variant, "Je": je, "seed": seed, "segment": segment, "cumulative_start": (segment - 1) * 5000, "cumulative_end": segment * 5000, "preceding_hdf5": str(previous_h5) if previous_h5 else None, "preceding_sha256": previous_h5_hash, "manifest": m, "full_field_output": True, "backend": "cpu_compatible", "rng_before_sha256": previous_rng_hash, "rng_after_sha256": sha(rng), "rng_after_path": str(rng), "preceding_rng_path": str(previous_rng) if previous_rng else None, "rng_protocol": "seed_case_first_segment" if segment == 1 else "restore_serialized_checkpoint"}
            (d / "chain.json").write_text(json.dumps(entry), encoding="utf-8")
            previous_h5, previous_rng, previous_h5_hash, previous_rng_hash = h5, rng, sha(h5), sha(rng)
    return plan / "screen_protocol.json", campaign


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    protocol, campaign = _write_campaign(tmp_path)
    monkeypatch.setattr(ar, "validate_chain_artifacts", lambda *args, **kwargs: [])
    return protocol, campaign


def test_complete_validated_plan_selects_gamma38(fixture):
    protocol, campaign = fixture
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["complete"] is True and result["artifact_validation"] is True
    assert result["finite_saved_fields_validation"] is True and len(result["rows"]) == 24
    assert result["selected_variant"] == "gamma38"
    assert result["negative_branch"]["stationary_claim"] is False


def test_no_validate_does_not_select(fixture):
    protocol, campaign = fixture
    result = ar.analyze(protocol, campaign, validate=False)
    assert result["complete"] is True and result["selected_variant"] is None


@pytest.mark.parametrize("mutation,needle", [("je", "Je"), ("config", "config"), ("guard", "guard"), ("missing", "missing"), ("rng", "RNG"), ("initial", "initial")])
def test_tampered_receipts_are_reported(fixture, mutation, needle):
    protocol, campaign = fixture
    chain = next(campaign.glob("**/segment1/chain.json")); data = json.loads(chain.read_text())
    if mutation == "je": data["manifest"]["Je"] = .3
    elif mutation == "config": data["manifest"]["config_hash"] = "bad"
    elif mutation == "guard": data["manifest"]["stream_guard_source_hash"] = "bad"
    elif mutation == "missing": data["manifest"]["initial_vector_path"] = str(chain.parent / "missing.bin")
    elif mutation == "rng": data["rng_after_sha256"] = "bad"
    else: data["manifest"]["initial_source_hash"] = "bad"
    chain.write_text(json.dumps(data), encoding="utf-8")
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["complete"] is False
    assert any(needle.lower() in str(e).lower() for e in result["errors"])


def test_primary_pair_remains_fixed(fixture):
    protocol, campaign = fixture
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["primary"]["Je_low"] == prep.CURRENTS[1]
    assert result["primary"]["Je_high"] == prep.CURRENTS[2]
    assert result["selected_variant"] == "gamma38"


def test_first_initial_identity_is_cross_variant_guarded(fixture):
    protocol, campaign = fixture
    paths = sorted(campaign.glob("gamma38_i001_seed*/segment1/chain.json"))
    chain = paths[0]; data = json.loads(chain.read_text())
    iv = Path(data["manifest"]["initial_vector_path"])
    iv.write_bytes(b"changed first packed vector")
    data["manifest"]["initial_vector_hash"] = sha(iv)
    data["manifest"]["initial_hash"] = sha(iv)
    chain.write_text(json.dumps(data), encoding="utf-8")
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["complete"] is False
    assert any("initial" in str(e).lower() for e in result["errors"])


def test_frozen_jobs_and_config_hashes_cannot_be_bypassed(fixture):
    protocol, campaign = fixture
    jobs = protocol.parent / "screen_jobs.json"
    value = json.loads(jobs.read_text()); value[0]["segments"] = 9
    jobs.write_text(json.dumps(value, indent=2) + "\n")
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["complete"] is False
    assert any("job" in str(e).lower() or "hash" in str(e).lower() for e in result["errors"])


def test_config_mutation_is_reported_even_with_campaign_receipts(fixture):
    protocol, campaign = fixture
    cfg = protocol.parent / "configs" / "gamma42.yaml"
    cfg.write_bytes(cfg.read_bytes().replace(b"gamma: 42.0", b"gamma: 43.0"))
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["complete"] is False
    assert any("config" in str(e).lower() or "hash" in str(e).lower() for e in result["errors"])


def test_missing_segment_is_reported(fixture):
    protocol, campaign = fixture
    chain = next(campaign.glob("**/segment10/chain.json"))
    chain.unlink()
    result = ar.analyze(protocol, campaign, validate=True)
    assert result["complete"] is False
    assert any("segment" in str(e).lower() or "chain" in str(e).lower() for e in result["errors"])
