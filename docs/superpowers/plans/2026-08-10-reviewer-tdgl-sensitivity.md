# Reviewer TDGL Sensitivity Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, submit, and analyze a provenance-controlled TDGL study that gates the corrected full curl--curl solver against the Figure 3 solver before running the reviewer-requested parameter sensitivity matrix.

**Architecture:** Keep the Julia solver unchanged. Add three focused Python tools: one generates one-factor-at-a-time configs from a tracked baseline, one creates immutable Slurm bundles and reconstructs the exact legacy solver, and one extracts NDR/temperature metrics from HDF5 output. Stage 1 submits only the legacy/corrected baseline pair; Stage 2 is generated and submitted only after the 5% compatibility gate is evaluated.

**Tech Stack:** Julia TDGL solver, Python 3.11+, `unittest`, NumPy, h5py, YAML-as-text configuration, Slurm, SSH/SCP, Git.

## Global Constraints

- Use the corrected repository solver at commit `1dbef4fb2e52b3e2621389b33054d8a6b31e9b3e` and verify its Git blob ID is `cd66915ac3c29defe7406900eef3b0d2da919744`.
- Reconstruct the legacy solver and verify its Git blob ID is exactly `93a34dd8f50a76d4ef014d37d541ac36de6955c3`.
- The baseline must use `k_eff=5.0`; reject `k_eff=0.1`.
- Stage 1 changes only the solver implementation. Stage 2 changes exactly one parameter per variant.
- Generated HDF5, plots, and Slurm logs stay outside Git.
- Remote study root is `/scratch/work/fangr1/tdgl-review-sensitivity-20260810`.
- Do not submit Stage 2 until Stage 1 has a recorded compatibility-gate decision.

---

### Task 1: Generate and validate the one-factor sensitivity matrix

**Files:**
- Create: `studies/reviewer_sensitivity/baseline.yaml`
- Create: `studies/reviewer_sensitivity/matrix.json`
- Create: `tools/generate_sensitivity_configs.py`
- Test: `test/test_generate_sensitivity_configs.py`

**Interfaces:**
- Consumes: tracked baseline YAML and literal JSON variants.
- Produces: `generate_configs(baseline_path: Path, matrix_path: Path, output_dir: Path) -> list[RunConfig]`, plus one `config.yaml` per named run and `generated_matrix.json`.

- [ ] **Step 1: Write failing tests for baseline and one-factor validation**

```python
import json
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
            bad.write_text(self.baseline.read_text().replace("k_eff: 5.0", "k_eff: 0.1"))
            with self.assertRaisesRegex(ValueError, "baseline thermal.k_eff must be 5.0"):
                generate_configs(bad, self.matrix, Path(tmp) / "out")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest test.test_generate_sensitivity_configs -v`

Expected: import failure for `tools.generate_sensitivity_configs`.

- [ ] **Step 3: Add the tracked baseline and exact matrix**

Copy the repository `config.yaml` to `studies/reviewer_sensitivity/baseline.yaml`, then set:

```yaml
thermal:
  k_eff: 5.0
  T0: 0.18
outputs:
  generate_mp4: false
```

Create `matrix.json` with this exact content:

```json
{
  "baseline": "baseline-fullcurl-t018",
  "variants": [
    {"name": "kappa-1p5", "section": "tdgl", "key": "kappa", "value": 1.5},
    {"name": "kappa-3p0", "section": "tdgl", "key": "kappa", "value": 3.0},
    {"name": "gamma-20", "section": "tdgl", "key": "gamma", "value": 20.0},
    {"name": "gamma-80", "section": "tdgl", "key": "gamma", "value": 80.0},
    {"name": "keff-2p5", "section": "thermal", "key": "k_eff", "value": 2.5},
    {"name": "keff-10", "section": "thermal", "key": "k_eff", "value": 10.0},
    {"name": "etaenv-5", "section": "thermal", "key": "eta_env", "value": 5.0},
    {"name": "etaenv-20", "section": "thermal", "key": "eta_env", "value": 20.0},
    {"name": "etahole-0p05", "section": "thermal", "key": "eta_hole", "value": 0.05},
    {"name": "etahole-0p2", "section": "thermal", "key": "eta_hole", "value": 0.2},
    {"name": "holewidth-15", "section": "thermal", "key": "holewidth", "value": 15.0}
  ]
}
```

- [ ] **Step 4: Implement the minimal config generator**

Implement these public types/functions:

```python
@dataclass(frozen=True)
class RunConfig:
    name: str
    config_path: Path
    changes: tuple[tuple[str, str, float], ...]


def replace_yaml_scalar(text: str, section: str, key: str, value: float) -> str:
    """Replace exactly one scalar key under exactly one top-level section."""


def read_yaml_scalar(text: str, section: str, key: str) -> float:
    """Read one numeric scalar and reject missing or duplicate definitions."""


def generate_configs(baseline_path: Path, matrix_path: Path, output_dir: Path) -> list[RunConfig]:
    """Write the baseline plus validated one-factor variants and a JSON manifest."""
```

The CLI must be:

```powershell
python tools/generate_sensitivity_configs.py `
  --baseline studies/reviewer_sensitivity/baseline.yaml `
  --matrix studies/reviewer_sensitivity/matrix.json `
  --output generated/reviewer_sensitivity/configs
```

- [ ] **Step 5: Run tests and inspect generated differences**

Run:

```powershell
python -m unittest test.test_generate_sensitivity_configs -v
python tools/generate_sensitivity_configs.py --baseline studies/reviewer_sensitivity/baseline.yaml --matrix studies/reviewer_sensitivity/matrix.json --output generated/reviewer_sensitivity/configs
git diff --no-index studies/reviewer_sensitivity/baseline.yaml generated/reviewer_sensitivity/configs/kappa-1p5/config.yaml
```

Expected: tests pass; the sample diff changes only `tdgl.kappa`.

- [ ] **Step 6: Commit Task 1**

```bash
git add studies/reviewer_sensitivity/baseline.yaml studies/reviewer_sensitivity/matrix.json tools/generate_sensitivity_configs.py test/test_generate_sensitivity_configs.py
git commit -m "feat: generate reviewer sensitivity configs"
```

---

### Task 2: Reconstruct solver provenance and package immutable Slurm runs

**Files:**
- Create: `studies/reviewer_sensitivity/submit.sbatch`
- Create: `tools/package_sensitivity_runs.py`
- Test: `test/test_package_sensitivity_runs.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: generated configs, current solver, `animation_data_utils.jl`, and submission template.
- Produces: `reconstruct_legacy_solver(current_text: str) -> str`, `git_blob_sha1(data: bytes) -> str`, and `package_runs(...) -> list[Path]`; every bundle contains `manifest.sha256`.

- [ ] **Step 1: Write failing provenance and packaging tests**

```python
class PackageSensitivityRunsTest(unittest.TestCase):
    def test_reconstructed_legacy_solver_has_archived_blob_id(self):
        current = (ROOT / "current_sweep_thermal_nofastscan.jl").read_text()
        legacy = reconstruct_legacy_solver(current)
        self.assertEqual(
            git_blob_sha1(legacy.encode()),
            "93a34dd8f50a76d4ef014d37d541ac36de6955c3",
        )

    def test_current_solver_has_approved_blob_id(self):
        data = (ROOT / "current_sweep_thermal_nofastscan.jl").read_bytes()
        self.assertEqual(git_blob_sha1(data), "cd66915ac3c29defe7406900eef3b0d2da919744")

    def test_stage1_bundles_differ_only_by_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundles = package_runs(ROOT, Path(tmp), stage="gate")
            self.assertEqual({p.name for p in bundles}, {"baseline-legacy-t018", "baseline-fullcurl-t018"})
            legacy_cfg = (Path(tmp) / "baseline-legacy-t018/config.yaml").read_bytes()
            fixed_cfg = (Path(tmp) / "baseline-fullcurl-t018/config.yaml").read_bytes()
            self.assertEqual(legacy_cfg, fixed_cfg)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest test.test_package_sensitivity_runs -v`

Expected: import failure for `tools.package_sensitivity_runs`.

- [ ] **Step 3: Implement exact legacy reconstruction**

Starting from the corrected solver, remove the two mixed-derivative assignments and restore these legacy expressions:

```julia
dAx[i,j] = (Jsx[i,j] + kappa2 * d2Ax_dy2) / sigma_local
dAy[i,j] = (Jsy[i,j] + kappa2 * d2Ay_dx2) / sigma_local
```

Also remove the added curl--curl stencil comment. Require each replacement to occur exactly once, normalize output to LF, and fail unless the reconstructed Git blob ID is `93a34dd8...`.

- [ ] **Step 4: Add the Slurm template**

```bash
#!/bin/bash
#SBATCH --job-name=tdgl-review
#SBATCH --output=%j_slurm.out
#SBATCH --error=%j_slurm.err
#SBATCH --ntasks=35
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G

set -euo pipefail
module load julia
test -s config.yaml
test -s current_sweep_thermal_nofastscan.jl
julia -p 34 current_sweep_thermal_nofastscan.jl config.yaml
```

- [ ] **Step 5: Implement packaging and manifests**

For each run, copy only:

```text
config.yaml
current_sweep_thermal_nofastscan.jl
animation_data_utils.jl
submit.sbatch
run_metadata.json
manifest.sha256
```

`run_metadata.json` records run name, stage, parameter change, solver role,
solver Git blob ID, source commit, and generation UTC time. The manifest uses
lowercase SHA-256 and relative POSIX paths.

Support:

```powershell
python tools/package_sensitivity_runs.py --stage gate --output generated/reviewer_sensitivity/gate
python tools/package_sensitivity_runs.py --stage matrix --configs generated/reviewer_sensitivity/configs --output generated/reviewer_sensitivity/matrix
```

- [ ] **Step 6: Ignore generated local staging**

Append:

```gitignore
# Reviewer sensitivity staging
generated/reviewer_sensitivity/
```

- [ ] **Step 7: Run tests and verify manifests**

Run:

```powershell
python -m unittest test.test_package_sensitivity_runs -v
python tools/package_sensitivity_runs.py --stage gate --output generated/reviewer_sensitivity/gate
Get-ChildItem generated/reviewer_sensitivity/gate -Recurse
```

Expected: two gate bundles; tests pass; each manifest verifies with an independent SHA-256 calculation.

- [ ] **Step 8: Commit Task 2**

```bash
git add studies/reviewer_sensitivity/submit.sbatch tools/package_sensitivity_runs.py test/test_package_sensitivity_runs.py .gitignore
git commit -m "feat: package provenance-controlled TDGL runs"
```

---

### Task 3: Extract NDR and suspended-temperature metrics

**Files:**
- Create: `requirements-review.txt`
- Create: `tools/analyze_sensitivity.py`
- Test: `test/test_analyze_sensitivity.py`

**Interfaces:**
- Consumes: `current_Je*.h5` files containing `Je`, `V`, `T`, and `T_avg`.
- Produces: `load_run(run_dir: Path) -> RunSeries`, `compute_ndr_metrics(current: ndarray, voltage: ndarray) -> NDRMetrics`, per-run `summary.json`, combined `summary.csv`, and comparison plots.

- [ ] **Step 1: Write failing synthetic-curve tests**

```python
class AnalyzeSensitivityTest(unittest.TestCase):
    def test_monotonic_curve_has_no_ndr(self):
        j = np.array([0.30, 0.25, 0.20, 0.15])
        v = np.array([1.00, 0.80, 0.60, 0.40])
        metrics = compute_ndr_metrics(j, v)
        self.assertFalse(metrics.present)

    def test_known_return_branch_reports_ndr_onset_slope_and_area(self):
        j = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        v = np.array([1.00, 0.75, 0.82, 0.90, 0.20])
        metrics = compute_ndr_metrics(j, v)
        self.assertTrue(metrics.present)
        self.assertAlmostEqual(metrics.onset_current, 0.25)
        self.assertLess(metrics.minimum_slope, 0.0)
        self.assertGreater(metrics.ndr_area, 0.0)

    def test_h5_loader_time_averages_voltage_and_suspended_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture_h5(Path(tmp), je=0.2, voltage=[1.0, 3.0], suspended_temperature=0.4)
            series = load_run(Path(tmp))
            self.assertAlmostEqual(series.voltage[0], 2.0)
            self.assertAlmostEqual(series.suspended_temperature_mean[0], 0.4)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest test.test_analyze_sensitivity -v`

Expected: import failure for `tools.analyze_sensitivity`.

- [ ] **Step 3: Declare analysis dependencies**

```text
numpy>=1.24
h5py>=3.8
matplotlib>=3.7
```

- [ ] **Step 4: Implement robust metric extraction**

Use the HDF5 `Je` scalar rather than trusting the filename. Time-average all
stored `V` snapshots. For temperature, use the `T` dataset and select x nodes
with `abs(x) < holewidth/2`; average over selected x, all interior y, and all
stored snapshots. Reject empty datasets, duplicate currents, missing keys, and
non-finite values.

For the return branch, preserve descending current order. Compute adjacent
finite-difference slopes and require at least two contiguous negative-slope
intervals for an NDR segment. Integrate the positive voltage rise over the
absolute current interval to report `ndr_area`.

- [ ] **Step 5: Implement the 5% gate report**

```python
def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), np.finfo(float).eps)


def evaluate_gate(legacy: NDRMetrics, corrected: NDRMetrics) -> GateDecision:
    onset_delta = relative_difference(legacy.onset_current, corrected.onset_current)
    area_delta = relative_difference(legacy.ndr_area, corrected.ndr_area)
    return GateDecision(
        passed=legacy.present and corrected.present and onset_delta <= 0.05 and area_delta <= 0.05,
        onset_relative_difference=onset_delta,
        area_relative_difference=area_delta,
    )
```

The CLI must write JSON and CSV without modifying run data:

```powershell
python tools/analyze_sensitivity.py gate --legacy generated/reviewer_sensitivity/downloads/gate/baseline-legacy-t018 --corrected generated/reviewer_sensitivity/downloads/gate/baseline-fullcurl-t018 --output generated/reviewer_sensitivity/analysis/gate
python tools/analyze_sensitivity.py matrix --root generated/reviewer_sensitivity/downloads/matrix --output generated/reviewer_sensitivity/analysis/matrix
```

- [ ] **Step 6: Run tests**

Run: `python -m unittest test.test_analyze_sensitivity -v`

Expected: all synthetic monotonic, NDR, HDF5, and gate tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add requirements-review.txt tools/analyze_sensitivity.py test/test_analyze_sensitivity.py
git commit -m "feat: analyze TDGL NDR sensitivity metrics"
```

---

### Task 4: Verify locally and build Stage 1 artifacts

**Files:**
- Generated only: `generated/reviewer_sensitivity/gate/**`

**Interfaces:**
- Consumes: Tasks 1--3.
- Produces: two locally verified Stage 1 bundles.

- [ ] **Step 1: Install analysis dependencies**

Run: `python -m pip install -r requirements-review.txt`

Expected: NumPy, h5py, and Matplotlib import successfully.

- [ ] **Step 2: Run all repository tests relevant to the change**

Run:

```powershell
python -m unittest discover -s test -p 'test_*.py' -v
python test/verify_full_curl_sympy.py
bash test/test_package_continuation_runs.sh
bash test/test_package_continuation_scan_runs.sh
bash test/test_continuation_scan_selection.sh
```

Expected: zero failures.

- [ ] **Step 3: Generate and package Stage 1**

```powershell
python tools/generate_sensitivity_configs.py --baseline studies/reviewer_sensitivity/baseline.yaml --matrix studies/reviewer_sensitivity/matrix.json --output generated/reviewer_sensitivity/configs
python tools/package_sensitivity_runs.py --stage gate --output generated/reviewer_sensitivity/gate
```

- [ ] **Step 4: Independently verify solver IDs and manifests**

Run the packager's `--verify` mode against both bundle directories and compare
the two `config.yaml` files byte-for-byte. Expected solver blob IDs are the two
values in Global Constraints.

---

### Task 5: Upload and submit the Stage 1 gate to Triton

**Files:**
- Remote create: `/scratch/work/fangr1/tdgl-review-sensitivity-20260810/gate/**`
- Remote create: `/scratch/work/fangr1/tdgl-review-sensitivity-20260810/submission-ledger.tsv`

**Interfaces:**
- Consumes: verified Stage 1 bundles.
- Produces: two Slurm job IDs and a remote immutable ledger.

- [ ] **Step 1: Preflight the exact remote target**

Resolve the remote path with `realpath -m` and require exact equality with
`/scratch/work/fangr1/tdgl-review-sensitivity-20260810`. Refuse to upload if a
non-empty gate directory already exists; preserve prior attempts under a named
`failed-runs/${job_id}` directory, where `job_id` is read from the submission
ledger.

- [ ] **Step 2: Upload the gate directory**

Use non-interactive SCP/SSH through `triton-via-local-kosh`. Recompute every
SHA-256 on Triton and compare with each uploaded `manifest.sha256`.

- [ ] **Step 3: Verify Julia and Slurm inputs remotely**

Run `module load julia`, print `julia --version`, verify 35 Slurm tasks and 64
GiB memory in both templates, and confirm the two configs are byte-identical.

- [ ] **Step 4: Submit exactly two jobs**

Run `sbatch --parsable submit.sbatch` from each gate directory. Append UTC
submission time, run name, job ID, solver blob ID, and config SHA-256 to the
ledger.

- [ ] **Step 5: Record initial scheduler state**

Use `squeue` and `sacct` to record state, resource reason, node count, and start
estimate without treating `PENDING` as a failure.

---

### Task 6: Evaluate the gate and execute the conditional Stage 2 path

**Files:**
- Remote create: `/scratch/work/fangr1/tdgl-review-sensitivity-20260810/analysis/gate/**`
- Conditional remote create: `/scratch/work/fangr1/tdgl-review-sensitivity-20260810/matrix/**`
- Create after results: `studies/reviewer_sensitivity/results/gate-decision.md`

**Interfaces:**
- Consumes: terminal Stage 1 outputs.
- Produces: compatibility decision, then either 11 corrected-solver variants or three corrected-temperature baselines.

- [ ] **Step 1: Wait for both Stage 1 jobs to terminate**

Poll with `sacct`; accept only `COMPLETED` with exit code `0:0`. Verify each run
contains non-empty `current_Je*.h5` files covering all 99 configured currents.

- [ ] **Step 2: Download scalar/HDF5 outputs needed for analysis**

Copy to a local ignored analysis directory without deleting remote output.

- [ ] **Step 3: Run the gate analyzer**

```powershell
python tools/analyze_sensitivity.py gate --legacy generated/reviewer_sensitivity/downloads/gate/baseline-legacy-t018 --corrected generated/reviewer_sensitivity/downloads/gate/baseline-fullcurl-t018 --output generated/reviewer_sensitivity/analysis/gate
```

Manually inspect matched order-parameter snapshots around NDR onset and add the
qualitative phase-slip-sequence judgment to `gate-decision.md`.

- [ ] **Step 4A: If the gate passes, package the eleven variants**

Do not resubmit the corrected `T0=0.18` baseline. Package the eleven generated
variant configs with the corrected solver, verify manifests, upload under
`matrix/`, submit, and append every job to the ledger.

- [ ] **Step 4B: If the gate fails, stop the matrix and submit three baselines**

Generate corrected configs at `T0=0.02`, `0.12`, and `0.18`, preserving all
other baseline keys. Package, upload, and submit only those three jobs. Record
that Figure 3 regeneration is required before sensitivity interpretation.

- [ ] **Step 5: Commit the gate decision**

```bash
git add studies/reviewer_sensitivity/results/gate-decision.md
git commit -m "docs: record TDGL solver compatibility gate"
```

---

### Task 7: Analyze the completed matrix and prepare reviewer-ready outputs

**Files:**
- Create: `studies/reviewer_sensitivity/results/summary.csv`
- Create: `studies/reviewer_sensitivity/results/summary.md`
- Create: `studies/reviewer_sensitivity/results/sensitivity.pdf`

**Interfaces:**
- Consumes: successful Stage 2 corrected-solver outputs.
- Produces: reviewer-ready metrics table, provenance report, and sensitivity figure.

- [ ] **Step 1: Verify Stage 2 completeness**

Cross-check terminal Slurm results against `generated_matrix.json`. Every run
must be completed or have a documented failure entry; never omit a failed
variant from the report.

- [ ] **Step 2: Run matrix analysis**

```powershell
python tools/analyze_sensitivity.py matrix --root generated/reviewer_sensitivity/downloads/matrix --output generated/reviewer_sensitivity/analysis/matrix
```

- [ ] **Step 3: Build the reviewer comparison**

Report absolute and baseline-relative NDR onset, minimum slope, NDR area,
suspended mean temperature, and suspended peak temperature. Group the figure
by parameter family and state that all parameters are dimensionless.

- [ ] **Step 4: Copy only compact, reviewer-ready artifacts into Git**

Copy CSV, Markdown, and PDF; do not copy HDF5 or Slurm logs. Include job IDs,
solver commit/blob, config hashes, and limitations in `summary.md`.

- [ ] **Step 5: Run final verification**

Run all tests from Task 4, `git diff --check`, verify the PDF opens, and compare
the summary row count against the expected run count.

- [ ] **Step 6: Commit and push**

```bash
git add studies/reviewer_sensitivity/results
git commit -m "docs: add reviewer TDGL sensitivity results"
git pull --rebase
git push
git status --short --branch
```
