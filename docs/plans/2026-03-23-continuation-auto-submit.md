# Continuation Auto Submit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update continuation packaging so each valid packaged run is submitted to SLURM immediately as it is created.

**Architecture:** Keep `package_continuation_runs.sh` responsible for folder creation, config rewriting, and launcher generation, but add an immediate `sbatch submit.sh config.yaml` call inside the per-run packaging loop. Extend the shell regression test to stub `sbatch` and verify one submission per valid packaged run.

**Tech Stack:** Bash, shell-based regression test

---

### Task 1: Add failing submission regression coverage

**Files:**
- Modify: `test/test_package_continuation_runs.sh`

**Step 1: Write the failing test**

Add a stub `sbatch` executable ahead of `PATH` that logs invocations, then assert:

```bash
test -f "$WORK_DIR/sbatch.log"
test "$(wc -l < "$WORK_DIR/sbatch.log")" -eq 2
grep -q "submit.sh config.yaml" "$WORK_DIR/sbatch.log"
```

**Step 2: Run test to verify it fails**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `FAIL` because the packager does not submit jobs yet.

**Step 3: Write minimal implementation**

Update `package_continuation_runs.sh` to `cd` into each packaged run directory and execute:

```bash
sbatch submit.sh config.yaml
```

**Step 4: Run test to verify it passes**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add package_continuation_runs.sh test/test_package_continuation_runs.sh docs/plans/2026-03-23-continuation-auto-submit.md
git commit -m "fix: submit continuation jobs during packaging"
```

### Task 2: Verify shell syntax and package flow

**Files:**
- Modify: `package_continuation_runs.sh`
- Modify: `test/test_package_continuation_runs.sh`

**Step 1: Run shell syntax verification**

Run: `bash -n package_continuation_runs.sh test/test_package_continuation_runs.sh`
Expected: exit code `0`

**Step 2: Re-run end-to-end packaging test**

Run: `bash test/test_package_continuation_runs.sh`
Expected: `PASS`

**Step 3: Commit**

```bash
git add package_continuation_runs.sh test/test_package_continuation_runs.sh docs/plans/2026-03-23-continuation-auto-submit.md
git commit -m "fix: submit continuation jobs during packaging"
```
