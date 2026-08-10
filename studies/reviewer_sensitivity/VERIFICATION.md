# Reviewer sensitivity verification record

Recorded 2026-08-10 for branch `codex/reviewer-tdgl-sensitivity`.

## Completed checks

- `python -m pytest -q test/test_generate_sensitivity_configs.py test/test_package_sensitivity_runs.py test/test_analyze_sensitivity.py`: 15 passed after adding the temperature-gate packaging test and correcting the analyzer to read HDF5.jl arrays in h5py's `(time, y, x)` order.
- `python test/verify_full_curl_sympy.py`: the implemented two-dimensional mixed-derivative form matches `curl curl A`.
- Both Stage 1 bundles pass `manifest.sha256` verification locally and on Triton.
- Both Stage 1 bundles contain LF-only `submit.sbatch` files and byte-identical baseline configurations (`b7f715f92c8c8b7274ee233694b20db554300b030fc26931dfb9eea9257b3d6a`).
- Triton Julia smoke test: `julia -p 2` loaded the full corrected solver with `NOFASTSCAN_SKIP_MAIN=1`; both workers defined `TDGLThermalParams` and `fdm_rhs_thermal!` (`checks=Bool[1, 1]`).

## Existing test-environment gaps

- The legacy Bash packaging tests refer to the untracked fixture path `test/fixtures/package_input/changeT1`, which is absent from the repository.
- The legacy selection test requires a local Julia installation, which is unavailable in the Windows workspace. The Triton Julia smoke test above covers solver loading and worker propagation for this reviewer run; it does not claim to replace the missing fixture-based legacy tests.

## Stage 1 submissions

Remote root: `/scratch/work/fangr1/tdgl-review-sensitivity-20260810`

| Run | Slurm job | Solver SHA-256 | Config SHA-256 |
| --- | --- | --- | --- |
| `baseline-legacy-t018` | `19639758` | `17102c2e9820c2d41b65b92472d3af48593b70de4d8372f09266c2c232afc4c9` | `b7f715f92c8c8b7274ee233694b20db554300b030fc26931dfb9eea9257b3d6a` |
| `baseline-fullcurl-t018` | `19639759` | `4de0605b63e137cc4895b178da5009dc4b7dcd259f7007f73e5a6b6efa30e304` | `b7f715f92c8c8b7274ee233694b20db554300b030fc26931dfb9eea9257b3d6a` |

The remote `submissions.tsv` ledger records the submission timestamps and hashes.

## Stage 1 gate result

The corrected-versus-legacy compatibility gate failed the predeclared 5%
threshold and therefore does not authorize the 11-point sensitivity matrix.

| Metric | Legacy | Corrected | Relative difference |
| --- | ---: | ---: | ---: |
| NDR onset current | 0.2112244898 | 0.15 | 0.289855 |
| NDR area | 0.00100650009 | 1.783576e-08 | 0.9999823 |
| Minimum differential slope | -89.48575 | -0.0018509 | -- |
| Mean suspended temperature | 0.27848864 | 0.27868515 | -- |
| Peak suspended temperature | 1.80583536 | 1.59527982 | -- |

The temperature statistics are similar, but the NDR observables are not. The
matrix remains blocked until the corrected baseline is checked at lower bath
temperatures as specified in the design.

## Temperature-gate submissions

Submitted on 2026-08-10 at 14:33 UTC under
`/scratch/work/fangr1/tdgl-review-temperature-gate-20260810`.

| Run | Bath temperature `T0` | Slurm job | Initial state |
| --- | ---: | ---: | --- |
| `baseline-fullcurl-t002` | 0.02 | 19657046 | Running on `pe[7,12,14]` |
| `baseline-fullcurl-t012` | 0.12 | 19657047 | Failed during Julia worker startup; superseded |
| `baseline-fullcurl-t012` retry | 0.12 | 19657185 | Running on `milan16` |

Both bundles passed local and Triton SHA-256 manifest verification, Bash syntax
checking, and `sbatch --test-only` before submission. Job 19657047 exposed a
resource-layout error: Slurm spread 35 tasks across three nodes while
`julia -p 34` launched all local workers on the batch host, and the workers did
not connect within 60 seconds. The submission template now requests one Slurm
task with 24 CPUs on one node and starts 23 local Julia workers. A regression
test checks this invariant. The corrected 0.12 bundle was reverified before
job 19657185 was submitted.

The 11-point
one-factor-at-a-time matrix is intentionally not submitted while this gate is
unresolved; submitting it now would produce results that fail the study's
predeclared solver-compatibility criterion.
