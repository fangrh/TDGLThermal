# Stream Stable Snapshots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stream stable-run snapshot data to HDF5 in post-skip batches of 100 so large continuation and no-fastscan jobs stop exhausting memory while preserving the existing output file structure.

**Architecture:** Replace the current "solve, materialize all 3D arrays, then write once" stable-run path with a streaming writer that appends trimmed snapshots directly to extendable HDF5 datasets. Compute `skip_idx` from the expected snapshot count up front, drop pre-skip snapshots before buffering, and flush kept snapshots in fixed-size batches of 100 so downstream readers still see the same datasets and trimmed-after-skip semantics.

**Tech Stack:** Julia, HDF5, OrdinaryDiffEq, shell + Julia regression tests

---

### Task 1: Add failing regression coverage for streamed stable output shape

**Files:**
- Modify: `test/distributed_continuation_definition.jl`
- Test: `test/distributed_continuation_definition.jl`

**Step 1: Write the failing test**

Add a regression that writes one small stable-run output file through the shared stable-save path, then asserts the stored datasets match the trimmed layout:

```julia
mktempdir() do dir
    result = simulate_point(0.1, p, config, "up", dir)
    h5_path = joinpath(dir, result[:h5_file])
    h5open(h5_path, "r") do file
        @test haskey(file, "times")
        @test haskey(file, "V")
        @test haskey(file, "psi_real")
        @test size(read(file, "psi_real"), 3) == length(read(file, "times"))
    end
end
```

**Step 2: Run test to verify it fails**

Run: `julia test/distributed_continuation_definition.jl`
Expected: `FAIL` because the test targets the new streaming contract before the implementation exists.

**Step 3: Write minimal implementation**

Add just enough stable-save refactoring so the test can exercise a single streamed stable output path.

**Step 4: Run test to verify it passes**

Run: `julia test/distributed_continuation_definition.jl`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test/distributed_continuation_definition.jl current_sweep_thermal_nofastscan.jl
git commit -m "test: cover streamed stable output layout"
```

### Task 2: Implement chunked HDF5 appends after skip_ratio

**Files:**
- Modify: `current_sweep_thermal_nofastscan.jl`
- Modify: `animation_data_utils.jl`
- Test: `test/distributed_continuation_definition.jl`

**Step 1: Write the failing test**

Tighten the regression to assert trimmed semantics and batching-friendly append behavior:

```julia
@test read(file, "stored_skip_idx") == expected_skip_idx
@test size(read(file, "psi_real"), 3) == expected_kept_snapshots
@test length(read(file, "V")) == expected_kept_snapshots
```

**Step 2: Run test to verify it fails**

Run: `julia test/distributed_continuation_definition.jl`
Expected: `FAIL` until the writer drops pre-skip snapshots and appends only kept batches.

**Step 3: Write minimal implementation**

Refactor the stable path to:

- precompute `n_snap_expected` and `skip_idx`
- create extendable HDF5 datasets for `times`, `V`, `psi_avg`, `T_avg`, `psi_real`, `psi_imag`, `T`, `Ax`, `Ay`
- buffer only post-skip snapshots
- flush every 100 kept snapshots
- flush the final partial batch at the end

Keep dataset names and final shapes unchanged.

**Step 4: Run test to verify it passes**

Run: `julia test/distributed_continuation_definition.jl`
Expected: `PASS`

**Step 5: Commit**

```bash
git add current_sweep_thermal_nofastscan.jl animation_data_utils.jl test/distributed_continuation_definition.jl
git commit -m "feat: stream stable snapshots to hdf5"
```

### Task 3: Wire continuation and old continuation paths to the streamed writer

**Files:**
- Modify: `current_sweep_thermal_nofastscan.jl`
- Modify: `current_sweep_thermal_continue.jl`
- Test: `test/test_continuation_scan_selection.sh`

**Step 1: Write the failing test**

Extend the shell integration test so the real runtime cases still pass with the streamed writer:

```bash
test -f "$WORK_DIR/$OUTPUT_DIR/current_Je0.1000_up.h5"
test -f "$WORK_DIR/$MULTI_OUTPUT_DIR/current_Je0.2000_up.h5"
```

If needed, add a check that the saved `times` and `V` lengths still match.

**Step 2: Run test to verify it fails**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `FAIL` until both the old stable path and the continuation worker use the streamed save path.

**Step 3: Write minimal implementation**

Update both:

- the old `simulate_point` path in `current_sweep_thermal_nofastscan.jl`
- the continuation stable worker in `current_sweep_thermal_continue.jl`

so they use the shared streamed stable-save logic and still return the compact per-point metadata expected by callers.

**Step 4: Run test to verify it passes**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `PASS`

**Step 5: Commit**

```bash
git add current_sweep_thermal_nofastscan.jl current_sweep_thermal_continue.jl test/test_continuation_scan_selection.sh
git commit -m "feat: use streamed stable writer in continuation paths"
```

### Task 4: Final verification and README note

**Files:**
- Modify: `README.md`
- Modify: `current_sweep_thermal_nofastscan.jl`
- Modify: `current_sweep_thermal_continue.jl`
- Modify: `animation_data_utils.jl`
- Modify: `test/distributed_continuation_definition.jl`
- Modify: `test/test_continuation_scan_selection.sh`

**Step 1: Run Julia regression coverage**

Run: `julia test/distributed_continuation_definition.jl`
Expected: `PASS`

**Step 2: Run continuation shell integration**

Run: `bash test/test_continuation_scan_selection.sh`
Expected: `PASS`

**Step 3: Add a short README note**

Document that stable outputs are now written in streamed post-skip batches to reduce memory pressure, while preserving the same `.h5` structure.

**Step 4: Re-run verification**

Run:

```bash
julia test/distributed_continuation_definition.jl
bash test/test_continuation_scan_selection.sh
```

Expected: both `PASS`

**Step 5: Commit**

```bash
git add README.md current_sweep_thermal_nofastscan.jl current_sweep_thermal_continue.jl animation_data_utils.jl test/distributed_continuation_definition.jl test/test_continuation_scan_selection.sh
git commit -m "perf: stream stable snapshots after skip"
```
