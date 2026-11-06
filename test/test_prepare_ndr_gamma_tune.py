import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import prepare_ndr_gamma_tune as pg


BASE = Path(__file__).parents[1] / "studies/full_curl_ndr/enhancement_20261105/configs/baseline.yaml"


def _fixture(tmp_path):
    cfg = tmp_path / "baseline.yaml"
    shutil.copyfile(BASE, cfg)
    return cfg


def test_full_plan_matrix_hashes_and_gamma_only(tmp_path):
    cfg = _fixture(tmp_path)
    out = tmp_path / "plan"
    protocol = pg.prepare(cfg, out)
    jobs = json.loads((out / "screen_jobs.json").read_text())
    assert len(jobs) == 24
    assert {(j["variant"], j["Je"], j["seed"]) for j in jobs} == {
        (v, je, s) for v, _ in pg.VARIANTS for je in pg.CURRENTS for s in pg.SEEDS
    }
    assert all(j["segments"] == 10 and "initial_state_path" not in j and "initial_rng_path" not in j for j in jobs)
    assert protocol["jobs"]["jobs_sha256"] == hashlib.sha256((json.dumps(jobs, indent=2) + "\n").encode()).hexdigest()
    parsed = [yaml.safe_load((out / "configs" / f"{v}.yaml").read_text()) for v, _ in pg.VARIANTS]
    assert (out / "configs/baseline.yaml").read_bytes() == cfg.read_bytes()
    baseline = parsed[0]
    for value, (_, gamma) in zip(parsed, pg.VARIANTS):
        assert value["tdgl"]["gamma"] == gamma
        value["tdgl"].pop("gamma")
    assert parsed[0] == parsed[1] == parsed[2]


def test_wrong_base_hash_rejected(tmp_path):
    cfg = _fixture(tmp_path)
    cfg.write_bytes(cfg.read_bytes().replace(b"T0: 0.18", b"T0: 0.19"))
    with pytest.raises(ValueError, match="hash"):
        pg.prepare(cfg, tmp_path / "plan")


def test_wrong_physics_rejected_even_if_fixture_hash_is_patched(tmp_path):
    cfg = _fixture(tmp_path)
    cfg.write_bytes(cfg.read_bytes().replace(b"k_eff: 5.0", b"k_eff: 4.0"))
    old = pg.BASE_HASH
    pg.BASE_HASH = hashlib.sha256(cfg.read_bytes()).hexdigest()
    try:
        with pytest.raises(ValueError, match="physics"):
            pg.prepare(cfg, tmp_path / "plan")
    finally:
        pg.BASE_HASH = old


def test_hot_initialization_rejected(tmp_path):
    cfg = _fixture(tmp_path)
    cfg.write_bytes(cfg.read_bytes().replace(b"T_initial_down: 1.2", b"T_initial_down: 1.1"))
    old = pg.BASE_HASH
    pg.BASE_HASH = hashlib.sha256(cfg.read_bytes()).hexdigest()
    try:
        with pytest.raises(ValueError, match="physics|initialization"):
            pg.prepare(cfg, tmp_path / "plan")
    finally:
        pg.BASE_HASH = old


def test_refuses_overwrite(tmp_path):
    cfg = _fixture(tmp_path)
    out = tmp_path / "plan"
    out.mkdir()
    with pytest.raises(FileExistsError):
        pg.prepare(cfg, out)
