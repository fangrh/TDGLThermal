# Continuation Packager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a packaging script that mirrors each run folder under an input collection, creates matching output folders, rewrites continuation configs, and prepares one-by-one launch scripts for extended runs.

**Architecture:** Add a new standalone Bash script at the repo root that scans each direct child of `--input_folder`, identifies the inner continuation data directory containing `current_Je*.h5`, and generates a matching packaged run directory under `--output_folder`. Keep the solver unchanged; the script will copy existing runtime files and rewrite `config.yaml` values for `stable_time`, `skip_ratio`, and relative `input_folder`.

**Tech Stack:** Bash, existing Julia runtime files, shell-based regression checks

---

### Task 1: Add a failing packaging test fixture

**Files:**
- Create: `test/test_package_continuation_runs.sh`
- Create: `test/fixtures/package_input/changeT1/runA/source_data/current_Je0.1000_up.h5`
- Create: `test/fixtures/package_input/changeT1/runA/config.yaml`
- Create: `test/fixtures/package_input/changeT1/runB/source_data/current_Je0.2000_down.h5`
- Create: `test/fixtures/package_input/changeT1/runB/config.yaml`

**Step 1: Write the failing test**

```bash
#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

"$ROOT_DIR/package_continuation_runs.sh" \
  --input_folder "$ROOT_DIR/test/fixtures/package_input/changeT1" \
  --output_folder "$WORK_DIR/changeT2" \
  --extend_run 5000 \
  --skip_ratio 0.6

test -d "$WORK_DIR/changeT2/runA"
grep -q "stable_time: 5000" "$WORK_DIR/changeT2/runA/config.yaml"
grep -q "skip_ratio: 0.6" "$WORK_DIR/changeT2/runA/config.yaml"
grep -q "input_folder: ../input_folder/runA/source_data" "$WORK_DIR/changeT2/runA/config.yaml"
```

**Step 2: Run test to verify it fails**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `FAIL` because `package_continuation_runs.sh` does not exist yet.

**Step 3: Write minimal implementation**

Create `package_continuation_runs.sh` with argument parsing, folder scanning, config rewrite, file copying, and per-run launcher generation.

**Step 4: Run test to verify it passes**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add package_continuation_runs.sh test/test_package_continuation_runs.sh test/fixtures/package_input docs/plans/2026-03-23-continuation-packager.md
git commit -m "feat: add continuation run packager"
```

### Task 2: Verify generated launchers and mirrored folder structure

**Files:**
- Modify: `test/test_package_continuation_runs.sh`
- Modify: `package_continuation_runs.sh`

**Step 1: Extend the test**

Add assertions for:
- mirrored output folders for each direct child in `--input_folder`
- copied runtime files (`current_sweep_thermal_nofastscan.jl`, `animation_data_utils.jl`, `submit.sh`)
- executable per-run launcher script
- top-level sequential runner script

**Step 2: Run test to verify it fails if implementation is incomplete**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `FAIL` until missing packaging behavior is added.

**Step 3: Write minimal implementation**

Update `package_continuation_runs.sh` to generate:
- `--output_folder/<run-folder>/run.sh`
- `--output_folder/run_all.sh`

**Step 4: Run test to verify it passes**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `PASS`

**Step 5: Run a shell syntax check**

Run: `bash -n package_continuation_runs.sh test/test_package_continuation_runs.sh`
Expected: exit code `0`

**Step 6: Commit**

```bash
git add package_continuation_runs.sh test/test_package_continuation_runs.sh test/fixtures/package_input docs/plans/2026-03-23-continuation-packager.md
git commit -m "feat: add continuation run packager"
```
