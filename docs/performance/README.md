# Linux full-curl performance

This directory records measured acceleration experiments for the corrected full-curl TDGL + thermal solver. The default reference remains unchanged. The frozen reference is commit `6ce36684a46c8e9938b75cd843c26148fe71e008`; its magnetic-damping sign correction is separate from the performance work.

## Results at a glance

- The compatible CPU backend is about 1.28–1.53× faster than the reference in matched single-point, full-HDF5 100τ measurements (`30` paired runs, three currents, deterministic and noise 0.001 cohorts). Stored fields, observables, and RHS counts were numerically exact.
- The persistent-worker pilot reaches 3.40× versus the normal 8-worker reference phase at 24 workers. This is a 100τ, 36-job/phase pilot, not a full-scan speed claim.
- The completed full scan contains 297 matched pairs / 594 trajectories: 99 distinct current points × three seeds, 24 workers, 5000τ, original noise 0.001, with HDF5 output included. Reference and compatible whole-phase medians are 828.787 s (13.813 min) and 564.539 s (9.409 min). The three-pair geometric-mean speedup is 1.4644× with a Student-t log-ratio 95% interval [1.4119, 1.5189] (df=2; this is not a confidence interval for the median). All nine datasets pass with maximum normalized field error 0. The earlier 24-pair production receipt (three currents × eight seeds) remains a separate ensemble.
- GPU full-RHS F64 batch 99 reaches 52.49× versus a one-core CPU kernel aggregate. This excludes transfer/setup and is not a whole-solver result.
- The F64 whole-solve batch-99 receipt reports CPU-compatible 222.89 s versus GPU 8.316 s (paired median 26.80×; range 26.37–28.76×). It cycles the three currents (0.06/0.18/0.27) over 99 seeded cool states; it is not 99 distinct currents. The all-99 accuracy receipt reports 99/99 achieved-accuracy passes, with worst budget ratio 0.0931.

The accuracy budgets are derived from predeclared original-tolerance (1e-4) reference behavior compared with accepted tighter references. They are not a claim of globally uniform 1e-6 error. `psi_avg` retains its original streamed meaning: mean |ψ|² over interior nodes. The [finite-phase compatibility argument](compatibility.md) explains why the CPU storage/link changes preserve the procedure.

## Scope boundaries

The GPU whole-solve receipt is deterministic F64, state-saving only: no HDF5 or MP4 timing and no production-noise claim. GPU noise compatibility remains pending. The compatible CPU path preserves the packed state, adaptive Tsit5 integration, output schedule, and RNG call pattern; no fast-math or reduced precision is enabled. The [production-noise investigation](noise_acceleration.md) separates measured RNG/RHS overhead from accepted-step count and reviews GPU, stabilized integration, and stochastic-model options.

All lower-precision whole-solve candidates below were rejected. Each uses the same 99 states, horizon, common times and predeclared per-state budgets; no gate was relaxed.

| Integration state / RHS arithmetic | Achieved gates | Candidate median | Paired F64 control median |
|---|---:|---:|---:|
| F32 / original KWT | 94/99 | 7.537 s | 7.774 s |
| F32 / stable KWT | 93/99 | 5.147 s | 7.827 s |
| F32 / promoted KWT mobility | 96/99 | 6.211 s | 7.802 s |
| F64 / F32 cache, stable KWT | 95/99 | 7.079 s | 7.985 s |
| F64 / F32 cache, promoted mobility | 94/99 | 7.667 s | 8.010 s |

An independent 384-bit real 2×2 mobility oracle checks 578 manufactured cases, including zero-order-parameter limits and nearly radial forcing. It validates the equivalent KWT algebra and isolates arithmetic from stored-input quantization; it does not establish trajectory accuracy. The promoted scalar output's maximum relative vector error is 5.73e-8. The complete-RHS stable formula reduces the tested maximum normalized temperature-derivative error from 8.92e-6 to 4.82e-7, yet the full trajectory gates above still reject the lower-precision modes. [Oracle receipt](results/kwt_conditioning.json), [complete-RHS probe](results/gpu_kwt_stable_probe.json).

IMEX KenCarp was slower and rejected; a one-case ROCK4 result was only a preliminary field comparison with noise unresolved. CUDA Graph replay did not improve the measured resident RHS speed.

The original renderer benchmark measured three alternating reference/candidate pairs at 20 frames and 5 fps. Render time was about 1.006 s per run, including HDF loading. A later independent check used the already installed Julia FFmpeg artifact: all six movies have 20 encoded frames, 5 fps and 1400×900 dimensions, with identical SHA-256 hashes. [Encoded-video verification](results/render_video_verification.json). Solver-plus-render addition is illustrative, not a new full end-to-end benchmark.

## Receipts and figures

- [CPU matched summary](results/cpu_v2_summary.json) · [worker pilot](results/worker_scaling_summary.json) · [GPU RHS](results/gpu_rhs_benchmark_v2.json)
- [Full 99-point scan receipt](results/full_scan_24_5000_summary.json) · [pair CSV](results/full_scan_24_5000_pairs.csv) · [I–V PNG](results/full_scan_24_5000_iv.png) · [I–V PDF](results/full_scan_24_5000_iv.pdf)
- [Earlier 5000τ production artifact receipt](results/production_5000_artifacts.json) · [production statistics](results/production_statistics.json)
- [GPU whole-solve receipt](results/gpu_solver_benchmark.json) · [all-99 accuracy receipt](results/gpu_batch99_accuracy.json)
- [Acceleration figure PNG](results/acceleration_summary.png) · [PDF](results/acceleration_summary.pdf)

Large HDF5 trajectories remain outside Git. JSON receipts retain source, input, output, and configuration hashes. No SSH addresses or credentials are part of the reproducibility record.

## Reproduce

Use Julia 1.12.1 in the isolated project environment and set caller-specific paths. The practical compatible-CPU command is:

```bash
export JULIA_BIN=/path/to/julia-1.12.1/bin/julia
export JULIA_DEPOT_PATH=/path/to/isolated_depot:/path/to/existing_depot
export OPENBLAS_NUM_THREADS=1
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate()'
TDGL_WORKERS=24 TDGL_RHS_BACKEND=cpu_compatible bash bench/run_linux.sh run config.yaml
```

To run the bounded 100τ CPU comparison, use `bench/run_linux.sh cpu` with an explicit `--out` directory, currents `0.06,0.18,0.27`, the intended preparation/noise settings, and the repetition count recorded in the receipt. GPU RHS reproduction is `bash bench/run_linux.sh gpu-rhs --out /caller/path/gpu_rhs --grid 60x30 --batches 1,8,32,99 --repetitions 5 --repeats 20`. The render harness is `bench/benchmark_rendering.jl`; it requires matched reference/candidate manifests and does not start solver jobs.

For the deterministic GPU usability receipt, first instantiate the isolated `experiments/gpu_probe` environment (using the same Julia 1.12.1), then run:

```bash
bash bench/run_linux.sh gpu-solve --out /caller/path/gpu_solve \
  --grid 60x30 --batches 1,8,32,99 --repetitions 3 \
  --horizon 100 --tolerance 1e-6 --noise 0
```

This command writes `gpu_solver_benchmark.json` inside the output directory. It measures state-saving whole-solve runs, not HDF5 or MP4 timing; batch 99 cycles the three currents over 99 states.

The launcher explicitly supplies Julia worker count. Python reporting uses NumPy, h5py, and Matplotlib; project and environment versions are recorded in the receipts. Primary implementation references are the [Julia performance guide](https://docs.julialang.org/en/v1/manual/performance-tips/), [CUDA.jl performance guide](https://cuda.juliagpu.org/stable/tutorials/performance/), [CUDA graph semantics](https://cuda.juliagpu.org/stable/usage/graphs/), and [SciML split-solver documentation](https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/).

## Pending gates

The full-scan measurements were made on a shared server with an existing GPU service; the hardware was not exclusive, and warm timings exclude startup. Larger grids, continuation, tolerance sensitivity, GPU production-noise validation, and any full animation-inclusive end-to-end benchmark remain pending. No approximation candidate has been promoted to production; the corrected reference remains the default.
