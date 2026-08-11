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
| `baseline-fullcurl-t002` | 0.02 | 19657046 | Failed with exit code 1; partial output retained |
| `baseline-fullcurl-t012` | 0.12 | 19657047 | Failed during Julia worker startup; superseded |
| `baseline-fullcurl-t012` retry | 0.12 | 19657185 | Failed with exit code 1; partial output retained |
| `baseline-fullcurl-t002` storage retry | 0.02 | 19667385 | Completed in 30:53 on `milan7` |
| `baseline-fullcurl-t012` storage retry | 0.12 | 19667387 | Completed in 34:18 on `milan8` |

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

## Scratch-space incident and retries

On 2026-08-11 the Triton scratch allocation reached its 536 GB hard limit.
Jobs 19657046 and 19657185 had terminated with exit code 1 and empty Slurm
logs; both left partial timestamped HDF5 output directories. Because a full
filesystem can prevent both HDF5 completion and error-log writes, these runs
are retained as failed technical attempts and are not treated as scientific
results. After the explicitly approved removal of the obsolete
`aiida_run` and `tdgl-runner` scratch trees, `lfs quota` reported approximately
184 GB in use. Fresh bundles using the verified single-node, 24-CPU template
were uploaded and hash-checked before submitting replacement jobs 19667385 and
19667387. The earlier partial outputs and job-specific logs remain separate.

## Full-curl long-time selected-state validation

On 2026-08-11, six selected return-branch states from completed Full Curl job
19667385 were staged under
`/scratch/work/fangr1/tdgl-review-longtime-fullcurl-20260811-v1`. The selected
grid currents are 0.1990, 0.2143, 0.2173, 0.2357, 0.2480, and 0.2786. This
set contains the four Figure 3 marker states and two additional points that
resolve the lower-current transition.

The source HDF5 metadata was checked on Triton before submission: each source
file has `original_n_snap=5001`, `stored_skip_idx=4501`, and 501 stored frames.
The long-time configuration uses `stable_time=50000`, `dt_snapshots=1`, and
`skip_ratio=0.96`, so the solver integrates the full interval but retains only
the final 2,001 frames. The streaming solver uses `save_everystep=false`, skips
all pre-window states, and writes retained states in 100-frame batches.

The uploaded solver has SHA-256
`a5aad45418094e5462e6a695366869886f027bc317e0a0f1a51b8e5ffa701286`,
matching the corrected Full Curl temperature-gate solver. The bundle passed
its SHA-256 manifest, Bash syntax, source-HDF5 metadata, and Slurm test-only
checks. Production job 19668918 started on `csl2` with one node, seven CPUs,
48 GB memory, and an eight-hour time limit.

Job 19668918 reached its eight-hour wall-time limit before any trajectory
entered the retained window at `48000 tau0`; Slurm recorded `TIMEOUT` after
08:05:16. The six HDF5 files contain only their 2,076-byte headers, confirming
that no pre-window field snapshots were retained and that no partial state is
available for continuation. The byte-identical Full Curl solver and the same
six source states were therefore resubmitted with a 24-hour limit as job
19683740. The retry requests no specific node and was pending for priority at
the time of this record.

Job 19683740 began before it could be held and was cancelled after 38:39 once
the rolling-checkpoint wrapper passed its Triton Julia smoke test. The wrapper
loads the byte-identical Full Curl solver above and replaces only the streamed
persistence function. With `TDGL_CHECKPOINT_INTERVAL=5000`, each current point
atomically replaces one single-frame `checkpoint_Je*.h5` file at every
`5000 tau0`; dense trajectory storage remains restricted to the final
`2000 tau0`. The preceding complete checkpoint survives interruption during a
temporary-file write. The checkpoint bundle passed its local and Triton
SHA-256 manifests, Bash syntax check, a one-frame HDF5 layout test, the 15-test
reviewer packaging/analysis gate, and Slurm test-only. Checkpoint-enabled job
19684435 started on `pe69` with seven CPUs, 48 GB, and a 24-hour limit.
