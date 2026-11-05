import copy
import sys

sys.path.insert(0, "bench")
from summarize_full_scan import contract_failures, phase_timings
from compare_results import meta_errors


def valid_receipt():
    rows = []
    for rep in (1, 2, 3):
        for idx in range(99):
            for backend in ("reference", "cpu_compatible"):
                rows.append({"repetition": rep, "current_index": idx + 1, "sample": 1,
                             "backend": backend, "complete": True, "status": "success",
                             "seed": 2026110401 + 100 * (rep - 1), "Je": idx / 98})
    phases = []
    for rep in (1, 2, 3):
        for phase in (1, 2):
            backend = "reference" if (phase == 1) == (rep % 2 == 1) else "cpu_compatible"
            phases.append({"repetition": rep, "phase": phase, "backend": backend,
                           "jobs": 99, "jobs_per_phase": 99, "total_phase_wall_seconds": 10.0})
    return {"rows": rows, "horizon": 5000.0, "repetitions": 3,
            "samples_per_current": 1, "noise": 0.001, "preparation": "configured_down",
            "full_scan": True, "actual_nworkers": 24, "current_grid": [idx / 98 for idx in range(99)],
            "selected_indices": list(range(1, 100)), "seed_base": 2026110401, "phases": phases}


def test_full_scan_contract_accepts_complete_matrix():
    assert contract_failures(valid_receipt()) == []


def test_full_scan_contract_rejects_empty_or_missing_pairs():
    data = valid_receipt()
    data["rows"] = [r for r in data["rows"] if not (r["repetition"] == 2 and r["current_index"] == 10)]
    failures = contract_failures(data)
    assert any("594" in x for x in failures)
    assert any("missing backend" in x or "297" in x for x in failures)


def test_full_scan_contract_rejects_shortened_horizon():
    data = copy.deepcopy(valid_receipt())
    data["horizon"] = 100.0
    assert any("horizon" in x for x in contract_failures(data))


def test_phase_timings_use_three_phase_receipts_not_per_point_rows():
    data = {"phases": [{"backend": "reference", "total_phase_wall_seconds": x} for x in (10, 20, 30)] +
            [{"backend": "cpu_compatible", "total_phase_wall_seconds": x} for x in (8, 16, 24)]}
    timings = phase_timings(data)
    assert timings == {"reference": [10.0, 20.0, 30.0], "cpu_compatible": [8.0, 16.0, 24.0]}


def test_valid_reference_candidate_roles_are_compared_as_a_pair():
    common = {"schema_version": "x", "config_hash": "c", "model_hash": "m", "rhs_source_hash": "r",
              "julia_version": "j", "machine": "host", "initial_hash": "i", "seed": 1, "Je": .1,
              "noise": .001, "Nx": 60, "Ny": 30, "hx": .5, "hy": .5, "horizon": 5000.,
              "tolerance": 1e-4, "snapshot_interval": 1., "skip_ratio": .9, "solver": "Tsit5",
              "precision": "Float64", "save_fields": True, "status": "success", "complete": True,
              "final_time": 5000., "wall_seconds": 2., "output_path": "x.h5", "output_hash": "h",
              "initial_vector_path": "x.bin", "initial_vector_hash": "iv", "output_schema": "streamed_hdf5_fields"}
    ref = dict(common, backend="reference", role="corrected_reference")
    cand = dict(common, backend="cpu_compatible", role="compatible_candidate", wall_seconds=1.)
    assert meta_errors(ref, cand) == []
