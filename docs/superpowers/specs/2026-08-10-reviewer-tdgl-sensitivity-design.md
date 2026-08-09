# Reviewer TDGL Sensitivity Study Design

## Objective

Produce a reproducible TDGL--thermal sensitivity study that directly addresses
Reviewer 1 Comments 3 and 10 and Reviewer 2 Comment 4. The study must first
measure whether the corrected full curl--curl operator changes the published
Figure 3 baseline. Only after that comparison may the corrected solver be used
for the parameter sweep.

This design covers TDGL simulations and their numerical analysis only. The
0.1% EPW calculation, device-current-density normalization, electrostatic
doping estimate, manuscript edits, and new experimental measurements are
separate work streams.

## Provenance and fixed baseline

The published Figure 3 run bundle contains an archived solver with Git blob
ID `93a34dd8f50a76d4ef014d37d541ac36de6955c3`. The current corrected solver is
Git blob `cd66915ac3c29defe7406900eef3b0d2da919744` at repository commit
`1dbef4fb2e52b3e2621389b33054d8a6b31e9b3e`. The corrected version adds the
mixed-derivative terms required by the full curl--curl operator written in the
manuscript.

Both baseline jobs use the same configuration and differ only in the solver
file:

- Grid: `Nx=60`, `Ny=30`, `hx=0.5`, `hy=0.5`.
- TDGL: `kappa=2.0`, `u=5.79`, `gamma=40.0`, uniform local `Tc=1.0`.
- Thermal: `C_eff=1.0`, `k_eff=5.0`, `T0=0.18`,
  `eta_env=10.0`, `eta_hole=0.1`, `holewidth=20.0`.
- Field and boundaries: `He=0`, periodic `x`, Neumann `y`, electrode coverage
  `0.88`, symmetric current-induced field.
- Sweep: return branch only, `Jpeak=0.3`, 99 current points,
  `stable_time=5000`, `skip_ratio=0.9`, normal/hot return-branch initial state.
- Video generation is disabled for the sensitivity jobs; HDF5 state and scalar
  outputs remain enabled.

The baseline value is `k_eff=5.0`, because every archived Figure 3 run config
uses that value. The manuscript and rebuttal currently state `k_eff=0.1`; that
text must later be corrected rather than changing the simulation provenance.

## Stage 1: solver-correction gate

Create two isolated run bundles under the remote root
`/scratch/work/fangr1/tdgl-review-sensitivity-20260810`:

1. `baseline-legacy-t018`, containing the exact archived Figure 3 solver.
2. `baseline-fullcurl-t018`, containing the current corrected solver.

Each bundle contains its solver, `animation_data_utils.jl`, an immutable
`config.yaml`, the Slurm script, and a manifest with SHA-256 hashes. The two
jobs use the same Julia module, task count, memory, and wall-time request.

The gate compares:

- NDR onset current, defined as the first return-branch current belonging to a
  contiguous negative-slope segment after smoothing only by the time average
  already present in each current-point output.
- Minimum finite-difference slope `dV/dJ` within that NDR segment.
- Signed return-branch NDR area and total hysteresis area when both branches
  are available; for the return-only gate, the NDR area is mandatory and the
  full hysteresis area is reported as unavailable.
- Peak and spatially averaged normalized temperature in the suspended region.
- Qualitative number and arrangement of phase-slip-line channels at matched
  current points around NDR onset.

The corrected solver passes the compatibility gate when both jobs complete,
the NDR onset current and NDR area differ by no more than 5%, and the
phase-slip-state sequence is qualitatively unchanged. If the gate passes, the
corrected solver becomes the only solver used in Stage 2. If it fails, rerun
the corrected baseline at `T0=0.02`, `0.12`, and `0.18` and regenerate the
Figure 3 simulation panels before interpreting any sensitivity result.

## Stage 2: one-factor-at-a-time sensitivity matrix

Use the corrected solver and the corrected `T0=0.18` baseline. Change one
factor at a time so every effect is attributable to one parameter. The matrix
contains the baseline plus these eleven variants:

| Family | Variant values | Fixed baseline value |
|---|---|---|
| GL parameter | `kappa=1.5`, `kappa=3.0` | `2.0` |
| Inelastic parameter | `gamma=20.0`, `gamma=80.0` | `40.0` |
| Thermal diffusion | `k_eff=2.5`, `k_eff=10.0` | `5.0` |
| Supported anchoring | `eta_env=5.0`, `eta_env=20.0` | `10.0` |
| Suspended anchoring | `eta_hole=0.05`, `eta_hole=0.2` | `0.1` |
| Suspended width | `holewidth=15.0` | `20.0` |

The matrix is intentionally not factorial. Its purpose is to establish local
robustness of the reported mechanism, not to fit phenomenological parameters
to one device. All temperatures remain normalized; no conversion to an
experimental kelvin rise is permitted without measured thermal constants.

## Output and analysis contract

Every run must provide:

- the exact input config and solver hash;
- Slurm job ID, terminal state, exit code, and elapsed time;
- current-resolved mean voltage and suspended-region temperature statistics;
- NDR onset current, minimum `dV/dJ`, NDR area, and whether an NDR segment is
  present;
- selected order-parameter and temperature snapshots around NDR onset;
- a machine-readable summary row and a combined comparison figure.

The analysis must not infer experimental vortex velocity, absolute state
lifetime, absolute thermal conductance, or an activation-energy barrier. The
simulation supports robustness of the qualitative thermal-feedback mechanism,
not a material-specific fit of dimensionless parameters.

## Reproducibility and failure handling

A generator creates named configs from a literal matrix and rejects duplicate
run names, accidental multi-parameter changes, missing baseline keys, and a
baseline `k_eff` other than `5.0`. Unit tests exercise those failures before
the generator is used.

A preflight check verifies all run-bundle files and manifests locally and on
Triton. Generated HDF5, images, videos, and Slurm logs remain outside Git. A
submission ledger inside the study directory records every job ID and config
hash. Failed jobs are preserved and diagnosed; they are never silently
overwritten or combined with successful results.

The analysis tests use synthetic I--V curves with known monotonic and NDR
segments to verify onset, slope, and area calculations. A completed Slurm job
is not considered a successful scientific run unless its expected HDF5 and
scalar outputs are present and non-empty.

## Acceptance criteria

The study is complete when:

1. The legacy-versus-corrected compatibility gate has a recorded decision.
2. The required corrected baselines have completed for the selected gate path.
3. Every Stage 2 matrix entry has a terminal result or a documented technical
   reason it cannot run.
4. Automated tests for config generation and NDR metrics pass.
5. The combined table and figure report parameter values, provenance, and
   uncertainty/limitations without claiming experimental calibration.
6. The manuscript parameter provenance is ready to be corrected from
   `k_eff=0.1` to the actually used `k_eff=5.0`.
