# NbSe2 suspended-device simulations for Figure 3 and Videos 1-3

This directory archives the Julia finite-difference TDGL-thermal workflow and
the three YAML configurations used for Figure 3c-e and Supplementary Videos
1-3 of the suspended NbSe2 manuscript.

## Files

- `current_sweep_thermal_nofastscan.jl`: simulation, HDF5 output, and MP4
  generation entry point.
- `animation_data_utils.jl`: alignment and validation utilities used by the
  animation pipeline.
- `config_T0.02_archived.yaml`, `config_T0.12_archived.yaml`, and
  `config_T0.18_archived.yaml`: archived configurations for the three reduced
  bath temperatures.
- `Project.toml`: Julia package environment for the archived script.

## Archived configuration

The domain is `30 xi0 x 15 xi0`, represented by a `60 x 30` grid with
`hx = hy = 0.5 xi0`. The centered weakly cooled region has width `20 xi0`.
The return branch contains 99 current points between `Je/J0 = 0.3` and zero.
Each target current is evolved for `5000 tau0`; fields and voltage are stored
every `1 tau0`, and the final `3000-5000 tau0` interval is retained by
`skip_ratio = 0.6`.

The archived continuation configurations retain the original relative
`input_folder` entries because those entries document how the published run
was initialized from its preceding saved return-branch states. These HDF5
state files are simulation output and are not stored in Git. For a new run
without continuation data, set `input_folder: none`; the initialization values
are then read from the same YAML file.

The numerical complex Gaussian nucleation term has amplitude `0.001`. It is
not fluctuation-dissipation calibrated, and the simulation time `tau0` is not
calibrated to experimental seconds. The archived calculation is therefore a
qualitative model of the vortex, phase-slip, and thermal-feedback mechanism,
not a device-specific prediction of an experimental frequency or onset
current.

## Run

Instantiate the Julia environment and execute one configuration:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. current_sweep_thermal_nofastscan.jl config_T0.02_archived.yaml
```

The script uses `Tsit5` with absolute and relative tolerances of `1e-4` for
the stable current-point evolution. HDF5 files contain `times`, `V`, complex
`psi`, normalized temperature `T`, and vector-potential components `Ax` and
`Ay`. When `generate_mp4: true`, the same stored fields are used for the
simulation videos.
