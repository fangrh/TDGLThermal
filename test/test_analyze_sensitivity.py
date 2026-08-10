import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from tools.analyze_sensitivity import (
    NDRMetrics,
    compute_ndr_metrics,
    evaluate_gate,
    load_run,
)


def write_fixture_run(root: Path, *, include_voltage: bool = True) -> None:
    (root / "config.yaml").write_text(
        """grid:
  Nx: 2
  Ny: 2
  hx: 1.0
  hy: 1.0
thermal:
  holewidth: 2.0
""",
        encoding="utf-8",
    )
    output = root / "sweep_nofastscan_fixture"
    output.mkdir()
    # HDF5.jl writes Julia (x, y, time) arrays such that h5py reads them as
    # (time, y, x).  Match the real solver output contract here.
    temperature = np.zeros((2, 3, 3), dtype=float)
    temperature[:, 1, 1] = 0.4
    with h5py.File(output / "current_Je0.2000_down.h5", "w") as handle:
        handle["Je"] = 0.2
        if include_voltage:
            handle["V"] = np.array([1.0, 3.0])
        handle["T"] = temperature
        handle["T_avg"] = np.array([0.1, 0.2])


class AnalyzeSensitivityTest(unittest.TestCase):
    def test_monotonic_curve_has_no_ndr(self):
        current = np.array([0.30, 0.25, 0.20, 0.15])
        voltage = np.array([1.00, 0.80, 0.60, 0.40])
        metrics = compute_ndr_metrics(current, voltage)
        self.assertFalse(metrics.present)

    def test_known_return_branch_reports_ndr_onset_slope_and_area(self):
        current = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        voltage = np.array([1.00, 0.75, 0.82, 0.90, 0.20])
        metrics = compute_ndr_metrics(current, voltage)
        self.assertTrue(metrics.present)
        self.assertAlmostEqual(metrics.onset_current, 0.25)
        self.assertAlmostEqual(metrics.minimum_slope, -1.6)
        self.assertAlmostEqual(metrics.ndr_area, 0.0075)

    def test_h5_loader_averages_voltage_and_suspended_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_run(root)
            series = load_run(root)
            self.assertAlmostEqual(series.voltage[0], 2.0)
            self.assertAlmostEqual(series.suspended_temperature_mean[0], 0.4)
            self.assertAlmostEqual(series.suspended_temperature_peak[0], 0.4)

    def test_h5_loader_rejects_missing_voltage_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture_run(root, include_voltage=False)
            with self.assertRaisesRegex(ValueError, "missing HDF5 dataset: V"):
                load_run(root)

    def test_gate_passes_only_when_onset_and_area_are_within_five_percent(self):
        legacy = NDRMetrics(True, 0.20, -2.0, 0.010)
        close = NDRMetrics(True, 0.208, -1.8, 0.0104)
        far = NDRMetrics(True, 0.22, -1.8, 0.0104)
        self.assertTrue(evaluate_gate(legacy, close).passed)
        self.assertFalse(evaluate_gate(legacy, far).passed)

    def test_failed_gate_without_ndr_is_strict_json(self):
        absent = NDRMetrics(False, None, None, 0.0)
        decision = evaluate_gate(absent, absent)
        json.dumps(asdict(decision), allow_nan=False)
        self.assertFalse(decision.passed)


if __name__ == "__main__":
    unittest.main()
