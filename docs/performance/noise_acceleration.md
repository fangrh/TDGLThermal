# Production-noise acceleration: measured bottleneck and available methods

The present evidence does not support a rejection storm as the reason the production run is slow. Accelerating random-number generation alone also cannot explain a tenfold gain. The practical route is to accelerate the complete noisy RHS on GPU and investigate the large number of accepted integration steps, while preserving the current protocol as the reference.

## Direct observations

The read-only [campaign observation](results/noise_path_observation.json) covers all 97 newly computed first segments of the full-range campaign. Each segment reaches 5000τ. This campaign uses `skip_ratio = 0.6`, retaining 2001 diagnostic samples per segment; the standalone root configuration's `0.9` default would retain 501. Accepted-step counts range from 269285 to 309457, with median 302181; 95 trajectories have one rejected step, one has two, and one has three. Median total RHS calls are 1815096 and median wall time is 121.519 s under the 24-worker shared-server load. The median mean accepted step is 0.01655τ. These observations do not identify whether accuracy, diffusion stability, or stochastic stage variability limits the accepted step.

The [fixed-state microbenchmark](results/noise_cost_probe.json) isolates the cost of evaluating the same fields with noise disabled/enabled. It uses seven alternating-order repetitions, 5000 calls per measurement, the full-curl study configuration, and two preparations. The input-field hashes agree within each noise pair. The allocation probe reports zero bytes for the sampled post-warmup call in every RHS/dpsi measurement.

| Fixed input preparation | Noise-off RHS | Noise-on RHS | Ratio of median costs | Increment as fraction of noise-on RHS |
|---|---:|---:|---:|---:|
| Configured hot start | 46.87 µs | 57.71 µs | 1.231 | 18.8% |
| Random nontrivial cool state | 48.58 µs | 60.49 µs | 1.245 | 19.7% |

These are fixed-state microbenchmarks on a shared CPU, not whole-solve speedups or physical-equivalence tests. Paired per-repetition ratios vary with load (hot 1.069–1.264; cool 1.199–1.371). Noise changes both arithmetic and random draws. Eliminating this measured overhead would yield roughly 1.23–1.25× for these RHS evaluations with all other costs fixed; it does not demonstrate such a whole-trajectory gain. A trajectory with noise disabled changes its dynamics and cannot by itself isolate RNG cost.

## What the current noise actually does

`compute_dpsi_dt!` draws two scalar Gaussians at each of 59 × 29 interior sites, giving 3422 draws per RHS call. It adds

```text
s(T) * (normal_real + i*normal_imag)
s(T) = noise_strength * sqrt(max(epsilon_T, 0) + 0.1)
```

after the KWT mobility inversion. In this study `epsilon_T = Tc - T` and `noise_strength = 0.001`. With the observed median call count, this is approximately 6.21 billion scalar Gaussian draws per segment. It is resampled at each RK stage/RHS evaluation, rather than attached to an increment of physical time. Streaming diagnostics also evaluate this noisy RHS and advance the integration RNG. Their calls and the retained-output schedule are part of the legacy procedure.

The heat equation uses a quadratic expression in the noisy `dpsi`. Consequently this forcing affects heating as well as nucleation. It is not an established fluctuation-dissipation calibrated thermal-noise model.

For a fixed-step numerical thought experiment, an independently resampled random derivative gives random state increments proportional to `h`; a Brownian SDE requires increments proportional to `sqrt(h)`. RK weights, stage reuse, FSAL, adaptivity and diagnostic calls further affect the legacy process. Thus the number `0.001` has no universal conversion into an SDE diffusion coefficient. Changing solver or sharing adaptive timesteps across a GPU batch can change the effective forcing, even if all generated Gaussians have the same marginal distribution.

## Methods worth pursuing

| Route | What it addresses | Compatibility and verification |
|---|---|---|
| GPU full noisy RHS with reproducible parallel RNG | Spatial kernels, Gaussian generation and many independent current/seed jobs | Retain Float64, full mixed curl, temperature coupling, noise amplitude and heat calculation. Document per-trajectory adaptation and diagnostic draws. A new RNG requires statistical equivalence tests; shared adaptive timesteps do not preserve the old noise protocol automatically. |
| Semi-implicit or stabilized integration | Hundreds of thousands of accepted steps, if deterministic diffusion stiffness is limiting | Split or stabilize the full coupled magnetic operator and thermal diffusion. Do not replace mixed curl with separate component Laplacians. Compare achieved accuracy and physical observables, including transition/recovery statistics. Changing legacy RK stages changes its noise. |
| Physical-time finite-correlation forcing | Makes phenomenological nucleation forcing independent of RHS evaluation count | Specify a correlation time/refresh schedule, resolve it, and show timestep convergence at fixed correlation time. This is a separately named model, not an equivalent optimization of the present one. |
| Calibrated SDE with GPU/stabilized solver | Defines a timestep-consistent stochastic model | Derive diffusion, normalization, stochastic interpretation and heat balance first. Then validate temporal/spatial convergence, equilibrium where applicable, and rare-event observables. |

**GPU methods are available.** SciML documents adaptive diagonal-noise SDE ensembles with `SOSRI` on `EnsembleGPUArray`. The lower-level `EnsembleGPUKernel` SDE algorithms currently use fixed steps and require supported noise structures/static GPU-safe states. The small-system examples are not a drop-in implementation for this large coupled TDGL grid. The existing repository GPU kernel explicitly rejects nonzero noise; its deterministic 26.80× versus one CPU core is not a production-noise speed claim. See [the official SDE example](https://docs.sciml.ai/DiffEqGPU/stable/examples/sde/) and [GPU solver contracts](https://docs.sciml.ai/DiffEqGPU/stable/manual/developer_interfaces/).

**Parallel RNG is a solved building block.** Random123 supplies counter-based generators such as Philox for CPU and GPU. A proposed mapping would distinguish trajectory, RHS-call or physical-increment index, cell and real/imaginary channel. CUDA.jl also supplies GPU array normal generation. Pin and test the installed APIs and generator choice; do not assume CPU/GPU Gaussian streams are identical. See [Random123](https://github.com/DEShawResearch/random123/blob/main/README.md) and [CUDA.jl array RNG](https://cuda.juliagpu.org/stable/usage/array/).

**Large timestep gains have TDGL precedent, with a different model and scheme.** Winiecki and Adams report a second-order semi-implicit TDGL/Maxwell method allowing much larger steps than the explicit schemes they compare. That motivates a candidate, not a promised 100× speedup for the present KWT/thermal/full-curl/noise case. The repository's existing noise-off ROCK4 pilot was about twice as fast in one full-grid test; its small-grid IMEX test was slower and failed its accuracy gate. Neither establishes production-noise acceptance. See [the TDGL paper](https://arxiv.org/abs/cond-mat/0106466), [ROCK4 pilot](results/rock4_full_grid.json), and [IMEX pilot](results/imex_smoke.json).

**SDE solver choice must follow the derived diffusion.** SciML lists `SOSRI` for diagonal/scalar Itô noise, `SOSRA` for additive noise, `SKenCarp` for stiff additive noise, and S-ROCK methods for stiff SDEs arising from PDE discretization. Here the coefficient depends on the evolving temperature, so the joint system is not automatically an additive-noise problem. `SKenCarp` therefore cannot be chosen solely because the added term is a Gaussian. See [official SDE solver guidance](https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/).

Adaptive SDE rejection must preserve the Brownian path rather than redraw a favorable increment after rejection. Brownian bridges and rejection sampling with memory provide this mechanism. NoiseWrapper can couple comparisons to the same noise path. This is correctness machinery for a new SDE model, not an explanation for the current one-to-three rejected steps. See [Rackauckas and Nie (2017)](https://escholarship.org/uc/item/5t6501dz), [DiffEqNoiseProcess](https://docs.sciml.ai/DiffEqNoiseProcess/stable/), and [NoiseWrapper](https://docs.sciml.ai/DiffEqNoiseProcess/stable/abstract_noise_processes/).

## Thermal and stochastic-model constraints

Temperature dependence alone does not force a nonzero Itô–Stratonovich correction: it vanishes if noise acts only on ψ, its coefficient depends only on finite-variation T, and T has no shared stochastic increment. A fluctuation-dissipation construction based on the KWT mobility is different: that mobility depends on ψ, and the covariance, drift convention, energy normalization and cell area must be derived consistently. The existing isotropic post-inversion forcing does not establish such a construction. Consistent Langevin terms matter in TDGL, as illustrated by [Berger's nonuniform-wire study](https://arxiv.org/abs/cond-mat/0605087); its wire-specific formula is not directly substituted into this geometry.

Do not transfer the current `|dpsi/dt|²` heating rule unchanged to Brownian white-noise trajectories. Their ordinary derivative does not exist, and squaring a stochastic finite-difference derivative introduces a term growing as `1/h`. A calibrated SDE needs a stochastic energy balance, including bath exchange and the relevant correction terms. Dropping stochastic heating also changes the model. Likewise, one initial random seed followed by deterministic evolution changes ongoing nucleation/recovery physics.

## Recommended implementation order and acceptance

Keep the running full-range 50000τ curve and original noisy CPU protocol unchanged. First compare short matched integrations at superconducting, transition, and normal/recovery points, recording accepted-step distributions, RHS calls and synchronized wall time. Isolate diffusion stability with deterministic controls; do not infer it merely from few rejected steps.

For the practical GPU candidate, compare throughput against the actual 24-worker CPU baseline, with the same time horizon, precision, output requirements and hardware-service context. Use independent seed ensembles and predeclared equivalence margins for voltage, temperature, condensate density, switching/retrapping currents and recovery probabilities. Overlapping error bars alone do not establish equivalence. Noisy GPU speedup remains unmeasured.

Only a separately defined physical-time noise/SDE candidate should change forcing normalization or stochastic heat. Its acceptance requires timestep convergence at fixed physical noise parameters, mesh/cutoff calibration and appropriate equilibrium/energy checks. These requirements separate reproducing the old numerical procedure from improving the physical stochastic model.

## Reproduce the cost probe

```bash
export JULIA_DEPOT_PATH=/path/to/isolated_depot:/path/to/existing_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 GKSwstype=100
NOFASTSCAN_SKIP_MAIN=1 NOFASTSCAN_SKIP_WORKERS=1 \
  /path/to/julia-1.12.1/bin/julia --project=. --startup-file=no \
  bench/profile_noise_cost.jl --count 5000 --repeats 7 \
  --out /path/to/new/noise_cost_probe.json
```

The output path must be new. The probe runs no trajectory and changes no production source or active campaign settings.
