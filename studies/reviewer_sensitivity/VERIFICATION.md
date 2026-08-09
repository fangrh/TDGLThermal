# Reviewer sensitivity verification record

Recorded 2026-08-10 for branch `codex/reviewer-tdgl-sensitivity`.

## Completed checks

- `python -m pytest -q test/test_generate_sensitivity_configs.py test/test_package_sensitivity_runs.py test/test_analyze_sensitivity.py`: 14 passed.
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

The remote `submissions.tsv` ledger records the submission timestamps and hashes. Stage 2 remains gated on the corrected-versus-legacy NDR comparison staying within the predeclared 5% threshold.
