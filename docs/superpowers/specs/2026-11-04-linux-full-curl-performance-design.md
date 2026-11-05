# Linux full-curl TDGL acceleration design

- Date: 2026-11-04
- Repository: fangrh/TDGLThermal
- Branch: codex/linux-full-curl-performance
- Status: written specification approved in chat on 2026-11-04; implementation evidence and remaining gates are recorded in bd and docs/performance/README.md.

## Objective and scope

Accelerate the existing Julia KWT TDGL–thermal calculation on the designated Linux server, then compare its results and performance with an explicitly frozen, corrected full-curl reference. Optimize both time for one trajectory and throughput for a complete current scan. Report these as separate quantities.

The user selected strict Float64/full-curl reference behavior first, followed by independently validated GPU, mixed-precision and alternative-integrator candidates. A speedup is accepted only at matched physical horizon, initial-state protocol, accuracy, noise policy and output requirements. No speedup factor is promised before measurement.

Keep the same coupled psi, Ax, Ay and temperature equations, staggered grid, supported boundaries, conductivity averaging, thermal feedback and voltage definitions. Removing mixed derivatives, changing geometry, suppressing production noise or shortening physical evolution is outside an acceleration claim. A physically reformulated stochastic model is a separate scientific project.

## Verified starting point and correctness prerequisite

The performance branch was created from master commit 6db4c3035e28dea7c0031741722c4fd5796f0773. The local reviewer branch at 1c2979b3311a7f864f5d119a8aa5aa59181b7707 contains a later correction, commit 1969372be4da34eb73462eef66f823a57e64b622, to the sign of the magnetic full-curl evolution.

For homogeneous positive conductivity, zero supercurrent, Ay = 0 and a transverse Ax(y) mode, the interior corrected magnetic term is +kappa^2 Dyy Ax / sigma, which damps a resolved nonconstant sinusoidal mode. The opposite sign in the legacy master grows this mode. This is an interior sign diagnostic; complete boundary-conditioned operator behavior requires additional tests.

The first implementation change must therefore be a separately identified correctness commit:

- Port the corrected full-curl signs and corresponding operator tests.
- The source correction commit also changes reviewer packaging tooling absent from master. Port only the applicable solver/test changes with explicit original-commit attribution; do not import unrelated reviewer infrastructure.
- Retain the legacy commit and its provenance unchanged.
- Freeze and identify the corrected solver before optimizing it. Reference manifests record the exact resulting Git commit and source hashes.
- Run both existing symbolic/stencil checks and the expanded complete-RHS checks described below.

Legacy-versus-corrected timing differences are consequences of changing the equation implementation, not optimization speedups. Existing saved results are classified by their generating source hashes before reuse.

Reference source: [full-curl sign correction](https://github.com/fangrh/TDGLThermal/commit/1969372be4da34eb73462eef66f823a57e64b622).

## Target machine and execution isolation

Read-only inspection found 24 physical / 48 logical Threadripper CPU cores, about 128 GB RAM, an NVIDIA RTX PRO 6000 Blackwell Workstation GPU with 96 GB memory, and Julia 1.12.1 available by absolute path. GPU memory is shared with an existing service; available memory and utilization must be recorded for each benchmark. Hardware utilization observed during design is not a resource reservation.

Use a dedicated project directory, Julia environment and output root. Do not alter running services or existing scientific run directories. Keep SSH addresses and credentials out of public repository files. Start with one benchmark worker; scale to 6, 12 and 24 physical-core workers only when resource use and memory measurements justify it. Keep BLAS threads at one per worker for the worker-scaling comparison and record the setting.

Pin Project.toml, Manifest.toml, Julia version, CPU target and relevant package/driver versions. Separate compilation/startup time from warmed execution time, while also reporting a cold-launch measurement for practical use. Do not upgrade the user's global Julia environment.

The GPU's FP64 throughput is only 1/64 of its FP32 throughput. Its memory capacity alone does not justify GPU execution for a 60 by 30 grid. This constrains experiment order rather than excluding GPU candidates.

## Architecture and interfaces

### Frozen reference

Preserve an executable corrected reference and its configuration/source manifest. Reference results and candidate results have separate directories; candidate code cannot overwrite the reference dataset. The reference stays available even after a candidate becomes the recommended backend.

The existing command-line/configuration entry point remains usable. New execution/backend settings default to the corrected reference. Existing scientific YAML keys, continuation input interpretation and HDF5 dataset names/shapes remain compatible. Backend selection, precision and numerical method are explicit metadata.

### Compatible CPU backend

Add a focused optimized RHS implementation and reusable workspace. Preserve the original packed complex-state interpretation and adaptive error-norm weighting initially. Avoid a simultaneous change of storage representation and integration algorithm.

Candidate work includes:

- Replace per-call slices and real-array temporaries in state unpacking with allocation-free indexed copies or verified views.
- Precompute invariant grid factors, conductivity edge averages, region coefficients and boundary masks where their values are constant.
- Improve column-major stencil access and reuse intermediate quantities without changing boundary update order.
- Specialize concrete parameter/workspace types and measure inference/allocation behavior.
- Evaluate SIMD and fused loops, initially without relaxed fast-math semantics.
- Preserve legacy RNG consumption order and diagnostic call schedules wherever compatibility is claimed.
- Measure independent-current/process parallelism separately from single-trajectory speed.

The currently inferred bottlenecks are hypotheses, not profile results. Existing source shows temporary arrays in unpack_state!, several stencil passes, per-RHS random draws, diagnostic RHS reevaluation and HDF5/animation work. Profile the normal solver before selecting the next optimization.

### Benchmark and comparison harness

Use a separate benchmark/comparison component to launch frozen-reference and candidate runs with immutable input manifests. It records configuration, source hashes, initial-state hashes, environment, timing, allocations, solver statistics, resources and output hashes.

The harness produces machine-readable JSON/CSV plus English figures and a concise report. A comparison fails explicitly on solver failure, NaN/Inf, incomplete physical horizon, missing datasets, wrong state provenance or incompatible noise/output settings. A failed candidate remains experimental and cannot replace the default backend.

### GPU candidate

Use device-resident arrays and fused full-curl/TDGL/thermal stencil kernels; transfer only required output batches. Evaluate independent-state batching and larger grids before expecting a small single trajectory to benefit. Cache reuse and launch reduction precede precision relaxation.

CUDA Graphs are a candidate for a repeated, validated stepping pattern. They are not a drop-in replacement for adaptive Tsit5 with rejected steps and changing host control flow. Device-graph capture must not accidentally freeze drive values or replay stale noise.

Float64 is tested first. Float32 or mixed precision is an explicit experimental mode that must pass its own accuracy and production gates. Higher-precision observable reductions do not by themselves validate lower-precision field evolution.

### Alternative integrators

Only pursue this route if profiling/step statistics identify a meaningful stiffness or step-count bottleneck after CPU work. Evaluate a coupled sparse IMEX split first, then exponential or FFT-preconditioned alternatives when supported by measurements.

Construct a split with L from the actual discrete full-curl and thermal operator and N(u) = f_reference(u) - L*u. Keep affine boundary forcing in the residual and verify that L*u + N(u) reproduces every full-RHS component.

Preserve Ax/Ay coupling, mixed derivatives, variable conductivity and boundary reconstruction. Do not substitute independent scalar Laplacians. Ordinary Fourier diagonalization is not assumed valid for partial electrode boundaries or heterogeneous coefficients. An FFT preconditioner must converge against the original operator. Do not add gauge damping or remove null modes merely to simplify the solve. Exponential phi functions require their finite zero-eigenvalue limits.

Changing the integrator is an approximation path with independent validation, not a compatible CPU optimization.

## Random forcing and diagnostic compatibility

The current source draws fresh randn values inside every RHS evaluation. These draws enter dpsi and its squared contribution to thermal production. This is solver-call-dependent forcing, not an already specified solver-independent Wiener process.

Snapshot diagnostics also evaluate the noisy RHS and consume random draws. Snapshot targets influence integration stops. Therefore a fixed seed alone does not establish equivalence after changing stage count, rejection behavior, loop ordering, output timing or thread scheduling.

Maintain two distinct comparison modes:

1. Deterministic validation: noise disabled explicitly for operator and finite-time convergence tests, with nontrivial initial fields. These results do not establish production equivalence with configured noise.
2. Legacy-noise production validation: preserve existing amplitude, draw/call semantics, diagnostic schedule and heat coupling for compatible CPU changes. Initialize the default RNG once per trajectory from its recorded seed in an isolated worker and pin versions; record call counts. For single-RHS noisy fixtures, reset that RNG to the same fixture seed before each reference/candidate call and compare both the RHS and the next 16 RNG draws. Preserve the existing conditional draw order (two Gaussian draws per active interior cell in the original traversal order). If adaptive paths diverge, do not claim bitwise trajectory reproducibility.

Do not silently replace the legacy forcing with pre-generated physical-time noise or SDEProblem. Either change defines a different stochastic model, particularly because heat production contains derivatives squared. A physical-noise reformulation requires a separate design.

Use explicit result labels. A **compatible acceleration** preserves the corrected discrete equations, existing integrator/error norm, legacy draw policy and output/diagnostic semantics. An **approximation candidate** changes precision, integration or another numerical policy and is compared at a declared error target; its speedup is reported as approximation-at-matched-accuracy, never as legacy-compatible acceleration. Deterministic-only validation carries a **deterministic-only** label. New-integrator/GPU production promotion requires all applicable statistical and tolerance-sensitivity gates; unresolved solver-call noise makes that result inconclusive, even if the deterministic candidate is fast.

## Reference cases and comparison protocol

The root config.yaml records a 60 by 30 grid with hx = hy = 0.5, gamma = 40, kappa = 2, thermal coupling enabled, k_eff = 5, T0 = 0.18, periodic x, Neumann y, electrode coverage 0.88, noise amplitude 0.001, 99 nonnegative current points up to 0.3, stable_time = 5000, snapshot interval 1 and skip_ratio = 0.9. MP4 generation is enabled there.

The paper archive describes 99 positive-current return points and manually continued 5000-time-unit segments to a cumulative 50000-time-unit horizon, using a different documented retained window. It is a distinct protocol. Its required prior states are not stored in Git.

Use the following finite benchmark suite:

| Case | Configuration and purpose |
|---|---|
| Complete-RHS tests | The finite fixture matrix below, with nontrivial seeded fields, heterogeneous conductivities and noise-off / RNG-reset noisy checks. |
| Short deterministic trajectories | 60 by 30, currents 0.06, 0.18 and 0.27, thermal coupling on, horizon 100; nonzero deterministic complex initial fields; both cool superconducting-like and hot normal-like preparations. Retain same snapshots and initial-state bytes for both paths. |
| Standard production points | Same three currents, configured noise 0.001, horizon 5000, snapshot interval 1 and skip 0.9. Both runs use the same preparation protocol and complete field-output requirements. |
| Worker scaling and full scan | First use the 12 current-grid indices 1,10,19,28,37,46,55,64,73,82,91,99 to compare worker counts 1/6/12/24. Only finalists run all 99 points, with run_up=false and run_down=true, at one selected worker count shared by reference and candidate. Also retain an 8-worker normal-run comparison when that differs from the selected count. If the reference identifies a switching region missed by the three production currents, validate its two adjacent grid points before promotion. |
| Continuation | Reuse identical saved starting states, continue ten 5000-unit segments to 50000, and compare the same cumulative final observation window. Missing historical states require a documented freshly generated reference preparation, not a claim to reproduce the historical paper run. |
| Scaling | 120 by 60 and 240 by 120 with domain size held fixed and spacing reduced accordingly; paired runs at each grid. These are scaling/convergence cases, not speedups obtained by changing the baseline grid. |
| Output accounting | Solver plus identical HDF5 output on both paths; render MP4 separately from the same fields. Also report end-to-end wall time with matching animation settings. |

Independent fresh current-point calculations and sequential ramp/continuation calculations are separate protocols. Parallelize only independent jobs; preserve dependencies for histories that carry state between current points.

The operator fixture grid is Nx=7, Ny=5, hx=0.5, hy=0.75. Use four saved state fixtures generated with Julia seeds 20261104 through 20261107: psi has independent real/imaginary normal values scaled by 0.3, Ax/Ay normal values scaled by 0.1, and T=0.2+0.2*uniform. Save the fixture arrays and hashes before candidate evaluation. Use sigma1/2/3=0.7/1.0/1.4 with region gaps 0.2/0.2, Je=0.12 and He=0.03. Other parameters come from the frozen root configuration. For each seed, exercise these ten configurations; this is a finite coverage matrix, not an exhaustive Cartesian product:

| Fixture | bc_x | bc_y | Coverage | loop | field_side | Other difference |
|---|---|---|---|---|---|---|
| 1 | periodic | neumann | 0.88 | false | both | Standard |
| 2 | dirichlet | neumann | 1.0 | false | both | Standard |
| 3 | neumann | neumann | 0.5 | false | upper | Standard |
| 4 | periodic | dirichlet | 0.88 | false | lower | Standard |
| 5 | dirichlet | dirichlet | 1.0 | false | both | Standard |
| 6 | neumann | dirichlet | 0.5 | false | both | Standard |
| 7 | periodic | neumann | 0.5 | true | both | Boundary loop |
| 8 | periodic | neumann | 0.88 | false | both | psi_y_zero=true |
| 9 | periodic | neumann | 0.88 | false | both | epsilon_type=bcs |
| 10 | periodic | neumann | 0.88 | false | both | thermal.enabled=false |

Each configuration is checked with noise=0 and with strength=0.001 under the RNG-reset contract. Save corrected-reference RHS fixtures for all four components, with source/runtime hashes. Additional manufactured transverse/null-mode tests use homogeneous conductivity and zero Je/He with boundaries chosen to admit the manufactured mode.

Bound the initial campaign to 24 elapsed server hours, including reference work. Schedule cases in the order in this document and record completed, failed and deferred cases. Do not launch a heavy case whose measured pilot estimate exceeds the remaining budget. A timed-out run is incomplete and cannot pass a physical-horizon gate; retain its ordinary checkpoint if available. Reaching the campaign cap produces a partial, explicitly qualified report and a budget-extension decision, not an assertion of full production validation.

## Acceptance gates

### 1. Operator and implementation correctness

For any field or observable q define E_q(a,b)=max(|a-b|)/(1+max(|b|)), using native model units. The maxima include every stored entry of that component, including boundary/corner rows; for time series they also include every specified common time. Complex psi uses complex absolute values, not separate normalization of its parts. Derived magnetic field and heat arrays use their full defined grid domains; scalar observables use all common times.

Compare psi, Ax, Ay and T RHS components individually against the corrected-reference fixtures. For the Float64 compatible kernel, require E_q no larger than 1e-11 for every field in the finite fixture matrix. No boundary mask or post-hoc phase/gauge alignment may hide a mismatch.

Retain the symbolic identity/stencil tests. Add homogeneous transverse damping and compatible null-mode checks where the boundary setup admits those modes. The interior damping test is not evidence for the entire boundary spectrum.

### 2. Deterministic integration accuracy

Use reference abstol=reltol values 1e-5, 1e-6 and 1e-7 on the short deterministic cases, denoted b5,b6,b7. Compare reconstructed fields and directly recomputed voltage/heat diagnostics at S={0,1,...,100}, with exact integration stops at those times on both paths; do not mix different interpolation conventions. For each q, estimate residual reference error as e_ref=E_q(b6,b7). Require e_ref <= 0.5*E_q(b5,b6), or require both differences <=1e-12. If convergence fails, add 1e-8 and then 1e-9 references using the same last-three-level rule; unresolved cases are inconclusive rather than triggering unlimited tightening.

Let b4 be the corrected original solver with abstol=reltol=1e-4, b_star the tightest accepted reference and e_ref its last-level residual. Require E_q(candidate,b_star) <= max(1.1*E_q(b4,b_star),5*e_ref,1e-9) for every component and observable. This prevents a candidate from receiving a looser accuracy budget merely because its solver uses a different tolerance convention.

Check continuation loading and HDF5 shapes/time coordinates independently. Adaptive packed-state error weighting must remain equivalent or be tested as an explicit integrator change.

### 3. Production and long-time equivalence

Noisy and dynamically sensitive trajectories are not required to remain pointwise identical indefinitely. Compare finite-time agreement before separation and then block-averaged voltage, temperature and order-parameter magnitude, their fluctuations, spectra and switching/branch behavior.

For each representative production regime, start with eight independently seeded reference/candidate pairs, with seeds 2026110401 through 2026110408. Extend only once, to at most 16 pairs using the next eight integer seeds, when uncertainty is too large to decide and the campaign budget permits it. Reuse these runs for timing where their settings match. Estimate autocorrelation-aware uncertainty from retained samples, using a batch length at least ten estimated integrated autocorrelation times and at least ten complete batches; otherwise that statistic is inconclusive. Use a paired, trajectory-level Student-t 95% confidence interval on the differences in trajectory statistics. Acceptance requires that interval to lie within a predeclared equivalence margin, not merely overlap of two confidence intervals.

For a scalar signal q, freeze its scale s_q=sqrt(mean(q_reference^2)) from the pooled retained reference samples before inspecting candidate results. The equivalence margin for its mean is max(0.01*s_q,5*epsilon_q), where epsilon_q is the native-unit uncertainty from that regime's reference tolerance study. For its nonzero standard deviation the margin is max(0.05*std(q_reference),5*epsilon_std), with epsilon_std estimated by the same reference-only study. Apply these to voltage, temperature and order-parameter magnitude. Dominant resolved frequencies have a 5% margin plus the larger of one Fourier bin and measured reference numerical frequency uncertainty; absent/unresolved peaks do not receive a frequency-equivalence claim. Switching-current displacement must be within one original current-grid interval plus reference numerical uncertainty. Spectral comparisons use the same retained observation window and sampling. If the reference uncertainty itself exceeds the scientific margin being tested, classify that statistic as inconclusive rather than enlarge an accuracy claim.

If margins cannot be resolved within the run cap, accuracy or sample duration, the result is inconclusive and the candidate remains experimental. For legacy-noise sensitivity use exactly three settings: the baseline 1e-4 tolerance, 1e-5, and 1e-6, on the same three production currents and first eight seeds, subject to the campaign budget. A fixed-step approximation instead uses its proposed step, half-step and quarter-step. An unresolved change with stage/call policy blocks a legacy-equivalence claim. A new integrator does not inherit production approval from deterministic tests.

### 4. Performance measurement

Warm each timed execution path once without including warmup in its measured samples. Perform exactly five paired short-case repetitions and three paired standard-case repetitions. Use three paired pilot-subset repetitions per worker count. Run three paired complete 99-point scans only for the final accepted candidate at the selected matched worker count; if a separate 8-worker comparison is needed, run three additional pairs there. Long continuation uses the production ensemble cap and its completed pairs for timing, without extra timing-only repeats. All work is subject to the 24-hour campaign cap. Alternate reference/candidate order and record background CPU/GPU contention. GPU wall times include synchronization.

Report median, spread and paired speedup uncertainty. An accepted speedup must exceed measured benchmark variability; publish inconclusive and slower cases as such. Separate:

- RHS time and allocations;
- accepted/rejected steps and RHS calls;
- linear-solve cost for implicit candidates;
- warm solver wall time;
- matched-output and animation cost;
- single-trajectory latency;
- scan throughput and total CPU/GPU resources;
- cold compilation/startup overhead.

No production promotion is based solely on a synthetic kernel benchmark.

## Delivery sequence and failure behavior

1. Establish the separately attributed sign correction and reference tests.
2. Freeze source/configuration/initial states and record a normal-solver baseline on Linux.
3. Profile and implement compatible CPU improvements one at a time; retain each only after gates pass.
4. Evaluate GPU candidates and, when justified by stiffness evidence, new integrators through the same gates.
5. Publish reproducible comparison artifacts and a recommendation. Keep unsuccessful experimental backends explicit rather than silently changing production defaults.

Store source, small configurations, tests and reports in Git; store large HDF5 trajectories outside Git with hashes and an artifact manifest. Use the repository's bd issue workflow for implementation tracking; this design document is a specification, not a parallel task tracker.

This specification was approved before implementation. Detailed execution plans were then written using the writing-plans skill; measured outcomes and unresolved gates belong in the performance report and bd.

## Primary technical sources

- [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/): allocation measurement, concrete types, preallocation, memory-order access and thread oversubscription.
- [CUDA.jl performance tips](https://cuda.juliagpu.org/stable/tutorials/performance/): profiling, kernel fusion, transfer minimization, memory access and precision/fast-math tradeoffs.
- [CUDA.jl graphs](https://cuda.juliagpu.org/stable/usage/graphs/): repeated-operation submission overhead and capture semantics.
- [SciML split ODE solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/): IMEX and semilinear/exponential interfaces.
- [SciML noise-process documentation](https://docs.sciml.ai/DiffEqNoiseProcess/stable/): explicit stochastic-process handling for a separately specified noise model.
- [NVIDIA RTX PRO Blackwell architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf): FP64 throughput and memory architecture.
