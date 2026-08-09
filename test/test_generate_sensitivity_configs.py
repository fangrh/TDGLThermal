import tempfile
import unittest
from pathlib import Path

from tools.generate_sensitivity_configs import generate_configs


class GenerateSensitivityConfigsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.baseline = self.root / "studies/reviewer_sensitivity/baseline.yaml"
        self.matrix = self.root / "studies/reviewer_sensitivity/matrix.json"

    def test_generates_baseline_plus_eleven_unique_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = generate_configs(self.baseline, self.matrix, Path(tmp))
            self.assertEqual(len(runs), 12)
            self.assertEqual(len({run.name for run in runs}), 12)

    def test_each_variant_changes_exactly_one_baseline_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = generate_configs(self.baseline, self.matrix, Path(tmp))
            for run in runs[1:]:
                self.assertEqual(len(run.changes), 1, run.name)

    def test_rejects_wrong_baseline_k_eff(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "baseline.yaml"
            bad.write_text(
                self.baseline.read_text(encoding="utf-8").replace(
                    "k_eff: 5.0", "k_eff: 0.1"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "baseline thermal.k_eff must be 5.0"
            ):
                generate_configs(bad, self.matrix, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
