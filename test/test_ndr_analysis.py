import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
from analyze_full_curl_ndr import _paired_stats, _stationarity, analyze, read_segment


def test_known_linear_positive_curve_fails_drop():
    assert not _paired_stats([0.02] * 8)["pass"]


def test_known_drop_passes_one_sided_gate():
    assert _paired_stats([-0.02] * 8)["pass"]


def test_three_seed_drop_does_not_confirm_eight_seed_protocol():
    assert not _paired_stats([-0.02] * 3)["pass"]


def test_incomplete_seed_set_is_explicitly_unresolved(tmp_path):
    result = analyze(tmp_path)
    assert result["incomplete"] is True
    assert result["complete_paired_seeds"] == []
    assert result["sustained_NDR"]["status"] == "unresolved"


def test_stationarity_drift_fails_gate():
    result = _stationarity([0.01] * 8)
    assert result["pass"] is False


def test_stationarity_zero_drift_passes_gate():
    assert _stationarity([0.0] * 8)["pass"] is True


def test_malformed_receipt_marks_campaign_incomplete(tmp_path):
    bad = tmp_path / "x" / "segment1"
    bad.mkdir(parents=True)
    (bad / "chain.json").write_text("{not-json", encoding="utf-8")
    result = analyze(tmp_path)
    assert result["incomplete"] is True
    assert result["artifact_errors"]


def test_integrated_voltage_uses_ax_interior(tmp_path):
    seg = tmp_path / "segment1"
    seg.mkdir()
    h5 = seg / "out.h5"
    times = np.array([0.0, 1.0])
    ax = np.zeros((2, 5, 4))
    ax[0, 1:-1, 1:-1] = 1.0
    ax[1, 1:-1, 1:-1] = 0.9
    with h5py.File(h5, "w") as f:
        f["times"] = times; f["V"] = [1.0, 2.0]; f["T_avg"] = [0.1, 0.1]; f["psi_avg"] = [0.2, 0.2]; f["Ax"] = ax
    mp = {"output_path": str(h5), "complete": True}
    entry = {"manifest": mp, "_segment_dir": str(seg), "cumulative_start": 0, "cumulative_end": 1}
    data = read_segment(entry)
    assert data["V_integrated"] == pytest.approx(3.0)
