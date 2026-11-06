import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import prepare_ndr_gamma_validation as pv

ROOT = Path(__file__).parents[1]
SCREEN = ROOT / "studies/full_curl_ndr/gamma_fine_tune_20261106/plan/screen_protocol.json"
AMEND = ROOT / "studies/full_curl_ndr/gamma_fine_tune_20261106/amendment_v2/amendment.json"


def test_valid_matrix_hashes_and_gamma_only(tmp_path):
    out = tmp_path / "validation"
    p = pv.prepare(SCREEN, AMEND, out)
    jobs = json.loads((out / "validation_jobs.json").read_text())
    assert len(jobs) == 48
    assert {(j["variant"], j["Je"], j["seed"]) for j in jobs} == {
        (v, je, seed) for v in ("baseline", "gamma38", "gamma42")
        for je in (0.20918367346938774, 0.21020408163265306)
        for seed in range(2026110611, 2026110619)
    }
    assert all(j["segments"] == 10 and j["prior_cumulative_horizon"] == 0 and "initial_state_path" not in j for j in jobs)
    assert p["jobs"]["jobs_sha256"] == hashlib.sha256((json.dumps(jobs, indent=2) + "\n").encode()).hexdigest()
    configs = [json.loads(json.dumps(__import__("yaml").safe_load((out / "configs" / f"{v}.yaml").read_text()))) for v in ("baseline", "gamma38", "gamma42")]
    for c in configs: c["tdgl"].pop("gamma")
    assert configs[0] == configs[1] == configs[2]
    assert p["success"]["all_eight_drops_positive"] is True


def test_amendment_hash_mismatch_rejected(tmp_path):
    amend = tmp_path / "amend.json"; value = json.loads(AMEND.read_text()); value["screen_protocol_sha256"] = "bad"; amend.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protocol hash"):
        pv.prepare(SCREEN, amend, tmp_path / "out")


@pytest.mark.parametrize("field", ["currents", "seeds", "alpha_family"])
def test_amendment_design_tampering_rejected(tmp_path, field):
    amend = tmp_path / "amend.json"; value = json.loads(AMEND.read_text())
    if field == "currents": value[field] = [0.2, 0.21]
    elif field == "seeds": value[field] = [1]
    else: value[field] = .1
    value["screen_protocol_sha256"] = hashlib.sha256(SCREEN.read_bytes()).hexdigest(); amend.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        pv.prepare(SCREEN, amend, tmp_path / "out")


def test_screen_config_tamper_rejected(tmp_path):
    screen_dir = tmp_path / "screen"; (screen_dir / "configs").mkdir(parents=True)
    screen = json.loads(SCREEN.read_text()); screen_path = screen_dir / "screen_protocol.json"
    for variant in screen["variants"]:
        src = SCREEN.parent / variant["config"]; shutil.copyfile(src, screen_dir / variant["config"])
    screen_path.write_text(json.dumps(screen))
    amend = json.loads(AMEND.read_text()); amend["screen_protocol_sha256"] = hashlib.sha256(screen_path.read_bytes()).hexdigest(); amend_path = tmp_path / "amend.json"; amend_path.write_text(json.dumps(amend))
    (screen_dir / "configs/gamma42.yaml").write_bytes((screen_dir / "configs/gamma42.yaml").read_bytes().replace(b"gamma: 42.0", b"gamma: 43.0"))
    with pytest.raises(ValueError, match="config"):
        pv.prepare(screen_path, amend_path, tmp_path / "out")


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(FileExistsError):
        pv.prepare(SCREEN, AMEND, out)


@pytest.mark.parametrize('field', ['success', 'early_stop', 'fresh_total_segments'])
def test_frozen_acceptance_and_allocation(tmp_path, field):
    value=json.loads(AMEND.read_text())
    if field=='success':value[field]['mean_fraction_min']=.10
    elif field=='early_stop':value[field]=True
    else:value[field]=240
    path=tmp_path/'amendment.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):pv.prepare(SCREEN,path,tmp_path/'out')
