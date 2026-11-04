import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
import analyze_full_iv as iv


def catalog():
    return {"schema": "tdgl-full-iv-catalog-v1", "protocol": iv.PROTOCOL,
            "points": [{"id": f"p{i}", "Je": x, "seed": iv.SEED,
                        "kind": "base" if i < 99 else "refinement",
                        "existing_chain_paths": [], "segments_to_run": 10}
                       for i, x in enumerate(iv.ALL_CURRENTS)]}


def test_catalog_accepts_exact_101_point_grid():
    assert iv.validate_catalog(catalog()) == []


@pytest.mark.parametrize("mutation", [
    lambda c: c["points"].pop(),
    lambda c: c["points"][0].update(seed=123),
    lambda c: c["points"][4].update(Je=0.123),
])
def test_catalog_rejects_missing_wrong_grid_or_seed(mutation):
    c = catalog(); mutation(c)
    assert iv.validate_catalog(c)


def _receipt(je, segment, *, horizon=5000.0):
    return {"id": f"p_{je}", "Je": je, "seed": iv.SEED, "segment": segment,
            "cumulative_start": (segment - 1) * 5000.0,
            "cumulative_end": segment * 5000.0,
            "manifest": {"seed": iv.SEED, "Je": je, "complete": True,
                         "status": "success", "horizon": horizon,
                         "skip_ratio": .6, "noise": .001, "tolerance": 1e-4,
                         "Nx": 60, "Ny": 30, "save_fields": True,
                         "output_schema": "streamed_hdf5_fields", "precision": "ComplexF64/Float64",
                         "config_hash": "cfg", "model_hash": "model", "rhs_source_hash": "rhs",
                         "final_time": 5000.0, "snapshot_interval": 1.0, "hx": .5, "hy": .5, "solver": "Tsit5"},
            "_chain_path": f"/x/segment{segment}/chain.json", "_segment_dir": "/x"}


def test_mixed_imported_one_plus_new_nine_merges(monkeypatch):
    je = iv.BASE_CURRENTS[0]
    monkeypatch.setattr(iv, "_complete_chain", lambda rows, expected_je=None: len(rows) == 10)
    rows, errors = iv._merge_entries([_receipt(je, 1)] + [_receipt(je, i) for i in range(2, 11)], je)
    assert not errors
    assert [x["segment"] for x in rows] == list(range(1, 11))


def test_short_horizon_fails_merge():
    je = iv.BASE_CURRENTS[0]
    rows, errors = iv._merge_entries([_receipt(je, 1, horizon=2500.0)], je)
    assert errors


@pytest.mark.parametrize("start,end", [(0.0, 4999.9), (0.1, 5000.0)])
def test_nonmultiple_or_inconsistent_cumulative_interval_is_rejected(start, end):
    je = iv.BASE_CURRENTS[0]
    entry = _receipt(je, 1)
    entry["cumulative_start"], entry["cumulative_end"] = start, end
    _, errors = iv._merge_entries([entry], je)
    assert any("invalid cumulative interval" in error for error in errors)


def test_conflicting_duplicate_segment_fails(monkeypatch):
    je = iv.BASE_CURRENTS[0]
    monkeypatch.setattr(iv, "_complete_chain", lambda rows, expected_je=None: False)
    a, b = _receipt(je, 1), _receipt(je, 1)
    b["manifest"]["output_hash"] = "different"
    _, errors = iv._merge_entries([a, b], je)
    assert any("conflicting duplicate" in x for x in errors)


def test_partial_result_serializes_without_nan(tmp_path):
    cpath = tmp_path / "catalog.json"; cpath.write_text(json.dumps(catalog()), encoding="utf-8")
    result = iv.analyze(cpath, tmp_path)
    out = tmp_path / "out"; iv.write_outputs(result, out)
    parsed = json.loads((out / "full_iv_analysis.json").read_text(encoding="utf-8"))
    assert parsed["pending"] is True
    assert "NaN" not in (out / "full_iv_analysis.json").read_text(encoding="utf-8")


def test_top_level_catalog_hash_mismatch_is_reported(tmp_path):
    c = catalog(); c["model_hash"] = "catalog-model"
    cpath = tmp_path / "catalog.json"; cpath.write_text(json.dumps(c), encoding="utf-8")
    root = tmp_path / "campaign"; seg = root / "point" / "segment1"; seg.mkdir(parents=True)
    receipt = _receipt(iv.BASE_CURRENTS[0], 1); receipt["manifest"]["model_hash"] = "wrong-model"
    (seg / "chain.json").write_text(json.dumps(receipt), encoding="utf-8")
    result = iv.analyze(cpath, root)
    assert any("model_hash mismatch catalog" in error for error in result["errors"])


def test_imported_one_ulp_current_is_canonicalized_to_declared_point(tmp_path):
    je = iv.BASE_CURRENTS[10]
    imported = tmp_path / "imported.json"
    receipt = _receipt(math.nextafter(je, math.inf), 1)
    imported.write_text(json.dumps(receipt), encoding="utf-8")
    c = catalog(); c["points"][10]["existing_chain_paths"] = [str(imported)]
    cpath = tmp_path / "catalog.json"; cpath.write_text(json.dumps(c), encoding="utf-8")
    result = iv.analyze(cpath, tmp_path / "campaign")
    assert not any("entry Je not in declared grid" in error for error in result["errors"])
    point = next(p for p in result["points"] if math.isclose(p["Je"], je, abs_tol=1e-15))
    assert point["complete"] is False


@pytest.mark.parametrize("bad_dataset", ["V", "psi_real"])
def test_saved_field_scan_detects_nonfinite_scalar_and_spatial_data(tmp_path, bad_dataset):
    import h5py
    import numpy as np
    path = tmp_path / "output.h5"
    with h5py.File(path, "w") as h:
        for name in iv.SAVED_DATASETS:
            h[name] = np.zeros((2, 2, 2) if name in ("psi_real", "psi_imag", "T", "Ax", "Ay") else (2,))
        h[bad_dataset][0] = np.nan
    entry = {"segment": 1, "manifest": {"output_path": str(path)}, "_segment_dir": str(tmp_path)}
    assert iv._finite_fields_errors([entry]) == [f"nonfinite saved field {bad_dataset} segment 1"]
    with h5py.File(path, "a") as h:
        h[bad_dataset][0] = 0.0
    assert iv._finite_fields_errors([entry]) == []
