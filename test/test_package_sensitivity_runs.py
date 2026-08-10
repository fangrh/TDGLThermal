import json
import tempfile
import unittest
from pathlib import Path

from tools.package_sensitivity_runs import (
    CURRENT_SOLVER_BLOB,
    LEGACY_SOLVER_BLOB,
    canonical_solver_bytes,
    git_blob_sha1,
    package_runs,
    reconstruct_legacy_solver,
    verify_bundle,
)
from tools.generate_sensitivity_configs import generate_configs


ROOT = Path(__file__).resolve().parents[1]


class PackageSensitivityRunsTest(unittest.TestCase):
    def test_reconstructed_legacy_solver_has_archived_blob_id(self):
        current = (ROOT / "current_sweep_thermal_nofastscan.jl").read_text(
            encoding="utf-8"
        )
        legacy = reconstruct_legacy_solver(current)
        self.assertEqual(git_blob_sha1(legacy.encode("utf-8")), LEGACY_SOLVER_BLOB)

    def test_current_solver_has_approved_blob_id(self):
        data = canonical_solver_bytes(
            ROOT / "current_sweep_thermal_nofastscan.jl"
        )
        self.assertEqual(git_blob_sha1(data), CURRENT_SOLVER_BLOB)

    def test_solver_bytes_are_normalized_to_lf_before_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solver.jl"
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(canonical_solver_bytes(path), b"first\nsecond\n")

    def test_stage1_bundles_differ_only_by_solver_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundles = package_runs(ROOT, Path(tmp), stage="gate")
            self.assertEqual(
                {path.name for path in bundles},
                {"baseline-legacy-t018", "baseline-fullcurl-t018"},
            )
            legacy = Path(tmp) / "baseline-legacy-t018"
            corrected = Path(tmp) / "baseline-fullcurl-t018"
            self.assertEqual(
                (legacy / "config.yaml").read_bytes(),
                (corrected / "config.yaml").read_bytes(),
            )
            self.assertNotEqual(
                (legacy / "current_sweep_thermal_nofastscan.jl").read_bytes(),
                (corrected / "current_sweep_thermal_nofastscan.jl").read_bytes(),
            )
            self.assertTrue(verify_bundle(legacy))
            self.assertTrue(verify_bundle(corrected))

    def test_matrix_packages_eleven_variants_without_resubmitting_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            configs = tmp_path / "configs"
            generate_configs(
                ROOT / "studies/reviewer_sensitivity/baseline.yaml",
                ROOT / "studies/reviewer_sensitivity/matrix.json",
                configs,
            )
            bundles = package_runs(
                ROOT,
                tmp_path / "bundles",
                stage="matrix",
                configs_dir=configs,
            )
            self.assertEqual(len(bundles), 11)
            self.assertNotIn("baseline-fullcurl-t018", {path.name for path in bundles})
            self.assertTrue(all(verify_bundle(path) for path in bundles))

    def test_temperature_gate_packages_two_corrected_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            configs = tmp_path / "configs"
            generate_configs(
                ROOT / "studies/reviewer_sensitivity/baseline.yaml",
                ROOT / "studies/reviewer_sensitivity/temperature_gate.json",
                configs,
            )
            bundles = package_runs(
                ROOT,
                tmp_path / "bundles",
                stage="temperature-gate",
                configs_dir=configs,
            )
            self.assertEqual(
                {path.name for path in bundles},
                {"baseline-fullcurl-t002", "baseline-fullcurl-t012"},
            )
            for bundle in bundles:
                metadata = json.loads((bundle / "run_metadata.json").read_text())
                self.assertEqual(metadata["stage"], "temperature-gate")
                self.assertEqual(metadata["solver_role"], "corrected")
                submit = (bundle / "submit.sbatch").read_text()
                self.assertIn("#SBATCH --nodes=1", submit)
                self.assertIn("#SBATCH --ntasks=1", submit)
                self.assertIn("#SBATCH --cpus-per-task=24", submit)
                self.assertIn("julia -p 23", submit)
                self.assertTrue(verify_bundle(bundle))


if __name__ == "__main__":
    unittest.main()
