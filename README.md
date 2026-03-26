# TDGL Julia No-Fast-Scan

This repository contains a Julia workflow for running no-fast-scan TDGL + thermal simulations, packaging run outputs, and continuing from saved `.h5` data.

## Key Scripts

- `current_sweep_thermal_nofastscan.jl`: main simulation entry point for running a current sweep from `config.yaml`.
- `current_sweep_thermal_continue.jl`: continuation-scan entry point that resumes from saved `.h5` states and ramps forward through existing current points.
- `package_and_submit.sh`: packages a fresh run directory and submits it with `sbatch`.
- `package_continuation_runs.sh`: packages one or more existing run folders for continuation runs using saved output data.
- `package_continuation_scan_runs.sh`: packages continuation-scan jobs that resume from saved current points with `--start`, `--end`, `--stable`, and `--ramptime`.
- `submit_scan.sh`: SLURM submit wrapper for the continuation-scan runner.

## Basic Run

Run the main simulation with a config file:

```bash
julia current_sweep_thermal_nofastscan.jl config.yaml
```

## Continuation Packaging

Package a folder of existing run outputs into a new output folder for an extended run:

```bash
./package_continuation_runs.sh \
  --input_folder changeT1 \
  --output_folder changeT2 \
  --extend_run 5000 \
  --skip_ratio 0.6
```

This creates a mirrored output structure and rewrites each packaged `config.yaml` so the continuation run points back to the corresponding `.h5` data under `../input_folder/...`.

## Continuation Scan Packaging

Package continuation-scan jobs from an existing sweep folder:

```bash
./package_continuation_scan_runs.sh \
  --input_folder changeT1 \
  --output_folder changeT2 \
  --start 0.10 \
  --end 0.30 \
  --stable 5000 \
  --ramptime 50
```

This packages one run directory per input run, copies `current_sweep_thermal_continue.jl`, rewrites `config.yaml`, and submits each job with `sbatch submit_scan.sh config.yaml`.

At runtime, the continuation runner:

- discovers saved current points from the input folder's `current_Je*.h5` files
- selects the saved current at or below `--start`
- runs a stable segment at that starting current
- keeps one ramp pipeline moving forward with `--ramptime`
- dispatches separate stable runs from each post-ramp state until reaching `--end`

The output `.h5` files keep the same naming and layout as the existing no-fastscan outputs so the result can be used for a later continuation run.

## Stable Snapshot Storage

Stable-run outputs are now written to HDF5 in streamed post-skip batches instead of materializing the full kept snapshot history in memory first. The on-disk `.h5` layout is unchanged: datasets such as `times`, `V`, `psi_real`, `psi_imag`, `T`, `Ax`, and `Ay` still contain the post-`skip_ratio` snapshots expected by the existing tooling.

## Git Hygiene

Generated run data, sweep outputs, and other large artifacts are intentionally kept out of git. Commit the source scripts, docs, and tests, not the simulation output folders.
