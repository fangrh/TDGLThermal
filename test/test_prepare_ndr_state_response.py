import hashlib, json, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import prepare_ndr_state_response as pr


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fixture(tmp_path):
    cfg = tmp_path / "config_T0.18.yaml"
    cfg.write_bytes(b"config fixture")
    old = pr.CONFIG_HASH
    pr.CONFIG_HASH = _sha(cfg)
    root = tmp_path / "baseline"
    for label, je in pr.SOURCE.items():
        for seed in pr.SEEDS:
            d = root / f"screen_{label}_seed{seed}" / "segment9"
            d.mkdir(parents=True)
            h5 = d / f"cpu_compatible_Je{je}_seed{seed}.h5"; h5.write_bytes(f"h5-{label}-{seed}".encode())
            rng = d / "rng_after.bin"; rng.write_bytes(f"rng-{label}-{seed}".encode())
            m = {"output_path": str(h5), "output_hash": _sha(h5), "Je": je, "noise": .001, "horizon": 5000.,
                 "tolerance": 1e-4, "snapshot_interval": 1., "skip_ratio": .6, "final_time": 5000.,
                 "Nx": 60, "Ny": 30, "hx": .5, "hy": .5, "seed": seed, "complete": True, "status": "success",
                 "config_hash": pr.CONFIG_HASH, "model_hash": pr.MODEL_HASH, "rhs_source_hash": pr.RHS_HASH,
                 "precision": "Float64", "solver": "Tsit5", "backend": "cpu_compatible",
                 "save_fields": True, "output_schema": "streamed_hdf5_fields"}
            (d / h5.with_suffix(".json").name).write_text(json.dumps(m), encoding="utf-8")
            chain = {"segment": 9, "seed": seed, "Je": je, "cumulative_start": 45000, "cumulative_end": 50000,
                     "manifest": m, "rng_after_path": str(rng), "rng_after_sha256": _sha(rng)}
            (d / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
    return root, cfg, old


def test_successful_frozen_matrix(tmp_path):
    root, cfg, old = _fixture(tmp_path)
    try:
        out = tmp_path / "plan"
        p = pr.prepare(root, cfg, out)
        jobs = json.loads((out / "state_response_jobs.json").read_text())
        assert len(jobs) == 24 and p["jobs"]["total_segments"] == 120
        assert all(j["currents"][0] == j["source_Je"] for j in jobs)
        assert {j["arm"] for j in jobs} == {"minus", "sham", "plus"}
        assert p["jobs"]["jobs_sha256"] == _sha(out / "state_response_jobs.json")
    finally:
        pr.CONFIG_HASH = old


def test_tampered_source_is_rejected(tmp_path):
    root, cfg, old = _fixture(tmp_path)
    try:
        chain = root / "screen_inner1_seed2026110411" / "segment9" / "chain.json"
        data = json.loads(chain.read_text()); data["manifest"]["Je"] = .2
        chain.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="source Je"):
            pr.prepare(root, cfg, tmp_path / "plan")
    finally:
        pr.CONFIG_HASH = old


@pytest.mark.parametrize("field,value", [("Je", float("nan")), ("final_time", float("nan")), ("hx", 0.25), ("seed", 1)])
def test_missing_or_invalid_metadata_is_rejected(tmp_path, field, value):
    root, cfg, old = _fixture(tmp_path)
    try:
        chain = root / "screen_inner1_seed2026110411" / "segment9" / "chain.json"
        data = json.loads(chain.read_text()); data["manifest"][field] = value
        chain.write_text(json.dumps(data, allow_nan=True), encoding="utf-8")
        with pytest.raises(ValueError): pr.prepare(root, cfg, tmp_path / "plan")
    finally:
        pr.CONFIG_HASH = old


def test_hdf_rng_and_source_age_are_rejected(tmp_path):
    root, cfg, old = _fixture(tmp_path)
    try:
        chain_dir = root / "screen_inner1_seed2026110411" / "segment9"
        chain = json.loads((chain_dir / "chain.json").read_text())
        (chain_dir / "rng_after.bin").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="RNG"):
            pr.prepare(root, cfg, tmp_path / "plan")
        chain["cumulative_start"] = 0
        (chain_dir / "rng_after.bin").write_bytes(b"rng-inner1-2026110411")
        (chain_dir / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
        with pytest.raises(ValueError, match="age"):
            pr.prepare(root, cfg, tmp_path / "plan2")
    finally:
        pr.CONFIG_HASH = old


def test_refuses_overwrite(tmp_path):
    root, cfg, old = _fixture(tmp_path)
    try:
        out = tmp_path / "plan"; out.mkdir()
        with pytest.raises(FileExistsError): pr.prepare(root, cfg, out)
    finally:
        pr.CONFIG_HASH = old


@pytest.mark.parametrize('kind', ['hdf', 'exact_rng_path', 'config'])
def test_source_integrity_guards(tmp_path, kind):
    root, cfg, old = _fixture(tmp_path)
    try:
        folder = root / 'screen_inner1_seed2026110411' / 'segment9'
        receipt = folder / 'chain.json'
        entry = json.loads(receipt.read_text())
        if kind == 'hdf':
            Path(entry['manifest']['output_path']).write_bytes(b'corrupt HDF payload')
        elif kind == 'exact_rng_path':
            entry['rng_after_path'] = str(tmp_path / 'missing' / 'rng_after.bin')
            receipt.write_text(json.dumps(entry))
        else:
            cfg.write_bytes(b'changed configuration')
        with pytest.raises(ValueError):
            pr.prepare(root, cfg, tmp_path / 'plan')
    finally:
        pr.CONFIG_HASH = old
