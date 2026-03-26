# Continuation Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a continuation pipeline runner and submission packager that resume from existing `.h5` sweep outputs, branch stable runs from post-ramp states, and preserve the current output structure for further continuation.

**Architecture:** Introduce a dedicated continuation Julia entrypoint plus a dedicated packaging/submission shell script. Keep the existing no-fastscan sweep path intact, reuse the current file/data conventions, and factor only the minimum helper logic needed for current discovery, state selection, and packaging.

**Tech Stack:** Julia, Bash, HDF5-backed sweep outputs, shell regression tests

---

### Task 1: Add failing shell coverage for continuation packaging

**Files:**
- Create: `test/test_package_continuation_scan_runs.sh`
- Test: `test/test_package_continuation_scan_runs.sh`

**Step 1: Write the failing test**

Create a shell regression test modeled on the existing continuation packaging test. Stub `sbatch`, run the new packaging script, and assert that:

```bash
test -d "$WORK_DIR/output/runA"
test -f "$WORK_DIR/output/runA/config.yaml"
test -f "$WORK_DIR/output/runA/current_sweep_thermal_continue.jl"
test -f "$WORK_DIR/output/runA/animation_data_utils.jl"
test -f "$WORK_DIR/output/runA/submit_scan.sh"
test -x "$WORK_DIR/output/runA/run.sh"
grep -q "input_folder:" "$WORK_DIR/output/runA/config.yaml"
grep -q "^submit_scan.sh config.yaml$" "$WORK_DIR/sbatch.log"
```

**Step 2: Run test to verify it fails**

Run: `bash test/test_package_continuation_scan_runs.sh`
Expected: `FAIL` because the new packaging script and copied files do not exist yet.

**Step 3: Write minimal implementation**

Create the new packaging script and any minimal fixture adjustments required so the test can package one or more continuation runs.

**Step 4: Run test to verify it passes**

Run: `bash test/test_package_continuation_scan_runs.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test/test_package_continuation_scan_runs.sh package_continuation_scan_runs.sh
git commit -m "test: cover continuation scan packaging"
```

### Task 2: Add failing Julia-facing coverage for saved-current discovery and range selection

**Files:**
- Create: `test/test_continuation_scan_selection.sh`
- Create or Modify: `test/distributed_continuation_definition.jl`
- Test: `test/test_continuation_scan_selection.sh`

**Step 1: Write the failing test**

Add a focused test harness that exercises helper logic for:

- collecting saved current values from existing `.h5` filenames
- selecting the largest saved value `<= --start`
- trimming the list at `--end`

Example assertions:

```bash
julia test/distributed_continuation_definition.jl --start 0.21 --end 0.44
grep -q "selected_start=0.20" "$WORK_DIR/result.log"
grep -q "targets=0.20,0.30,0.40" "$WORK_DIR/result.log"
```

**Step 2: Run test to verify it fails**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `FAIL` because the continuation-selection helpers are not implemented yet.

**Step 3: Write minimal implementation**

Add helper code in the new continuation runner, or extract a small helper section, so the test harness can validate current discovery and range selection.

**Step 4: Run test to verify it passes**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test/test_continuation_scan_selection.sh test/distributed_continuation_definition.jl current_sweep_thermal_continue.jl
git commit -m "feat: add continuation current selection"
```

### Task 3: Implement the continuation Julia runner

**Files:**
- Create: `current_sweep_thermal_continue.jl`
- Modify: `animation_data_utils.jl`
- Modify: `README.md`

**Step 1: Write the failing execution-oriented test**

Extend the Julia-side harness or add a second scripted test to verify that the runner accepts continuation CLI arguments and prepares the continuation target sequence and output paths.

Expected checks:

- accepts `config.yaml --input_folder ... --start ... --end ... --stable ... --ramptime ...`
- chooses the correct initial `.h5`
- prepares output for only the selected continuation range

**Step 2: Run test to verify it fails**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `FAIL` because the full continuation runner entrypoint is not implemented.

**Step 3: Write minimal implementation**

Implement `current_sweep_thermal_continue.jl` to:

- parse CLI arguments
- discover and validate existing `.h5` inputs
- load the chosen initial state
- maintain the ramp pipeline state across current targets
- dispatch stable runs from each post-ramp state
- write per-current outputs in the existing `.h5` structure

**Step 4: Run test to verify it passes**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add current_sweep_thermal_continue.jl animation_data_utils.jl README.md
git commit -m "feat: add continuation pipeline runner"
```

### Task 4: Implement the continuation submission script

**Files:**
- Create: `submit_scan.sh`
- Modify: `package_continuation_scan_runs.sh`
- Test: `test/test_package_continuation_scan_runs.sh`

**Step 1: Write the failing test**

Tighten the shell regression test to assert the generated run command targets the continuation script through the new submit wrapper:

```bash
grep -q "^submit_scan.sh config.yaml$" "$WORK_DIR/sbatch.log"
grep -q "sbatch submit_scan.sh config.yaml" "$WORK_DIR/output/runA/run.sh"
```

**Step 2: Run test to verify it fails**

Run: `bash test/test_package_continuation_scan_runs.sh`
Expected: `FAIL` until the new submit wrapper is wired in.

**Step 3: Write minimal implementation**

Add `submit_scan.sh` and update the packager to copy and submit it instead of the existing `submit.sh` for continuation-scan runs.

**Step 4: Run test to verify it passes**

Run: `bash test/test_package_continuation_scan_runs.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add submit_scan.sh package_continuation_scan_runs.sh test/test_package_continuation_scan_runs.sh
git commit -m "feat: add continuation scan submit wrapper"
```

### Task 5: Verify syntax and end-to-end packaging flow

**Files:**
- Modify: `README.md`
- Modify: `package_continuation_scan_runs.sh`
- Modify: `submit_scan.sh`
- Modify: `current_sweep_thermal_continue.jl`
- Modify: `test/test_package_continuation_scan_runs.sh`
- Modify: `test/test_continuation_scan_selection.sh`

**Step 1: Run shell syntax verification**

Run: `bash -n package_continuation_scan_runs.sh submit_scan.sh test/test_package_continuation_scan_runs.sh test/test_continuation_scan_selection.sh`
Expected: exit code `0`

**Step 2: Re-run shell regression tests**

Run: `bash test/test_package_continuation_scan_runs.sh`
Expected: `PASS`

**Step 3: Re-run Julia selection/execution harness**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `PASS`

**Step 4: Update README usage**

Document the new continuation-scan packaging flow and the meaning of `--start`, `--end`, `--stable`, and `--ramptime`.

**Step 5: Commit**

```bash
git add README.md package_continuation_scan_runs.sh submit_scan.sh current_sweep_thermal_continue.jl test/test_package_continuation_scan_runs.sh test/test_continuation_scan_selection.sh
git commit -m "docs: document continuation scan workflow"
```
