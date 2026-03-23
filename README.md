# TDGL Julia No-Fast-Scan

This repository contains a Julia workflow for running no-fast-scan TDGL + thermal simulations, packaging run outputs, and continuing from saved `.h5` data.

## Key Scripts

- `current_sweep_thermal_nofastscan.jl`: main simulation entry point for running a current sweep from `config.yaml`.
- `package_and_submit.sh`: packages a fresh run directory and submits it with `sbatch`.
- `package_continuation_runs.sh`: packages one or more existing run folders for continuation runs using saved output data.

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

## Git Hygiene

Generated run data, sweep outputs, and other large artifacts are intentionally kept out of git. Commit the source scripts, docs, and tests, not the simulation output folders.
