# Linux Full-Curl CPU Acceleration Implementation Plan

> **For agentic workers:** Execute with controller-owned integration and bounded Luna workers as required by the user's AGENTS instructions. Track progress with bd; numbered steps here describe implementation, not a parallel checkbox tracker.

**Goal:** Deliver a corrected full-curl reference, a compatible faster CPU backend and reproducible Linux comparisons.

**Architecture:** Preserve the packed ComplexF64 state, Float64 physical fields, Tsit5 and legacy noise/diagnostic semantics. Add an optional allocation-reduced RHS and a separate comparison harness. GPU and alternative-integrator execution plans follow measured bottleneck evidence; their implementation is not claimed by this CPU plan.

**Tech Stack:** Julia 1.12.1, OrdinaryDiffEq/Tsit5, HDF5, YAML, JSON3, BenchmarkTools, Python plotting, Git and SSH.

**Spec:** docs/superpowers/specs/2026-11-04-linux-full-curl-performance-design.md, approved in chat on 2026-11-04.

## Global Constraints

- Preserve the complete coupled equations, staggered grid, boundaries, conductivity averaging, heat feedback and voltage definitions.
- Correct magnetic damping separately and freeze that reference commit before optimizing.
- Preserve packed-state error weighting, legacy RNG order and diagnostic calls for compatible acceleration.
- Keep the corrected reference as default; no relaxed fast-math.
- Match grid, initial bytes, physical horizon, precision, noise, tolerances and output requirements.
- Use an isolated Linux environment/output directory and preserve existing services.
- Keep SSH addresses and credentials out of public Git files.
- Bound the initial server campaign to 24 hours; classify incomplete/inconclusive gates explicitly.
- Controller owns integration/review; at most two subagents, no grandchildren.
- Use bd for progress rather than Markdown checkboxes.

## File boundaries

| Files | Responsibility |
|---|---|
| current_sweep_thermal_nofastscan.jl | Correct reference, optional backend integration, existing entry point. |
| test/test_full_curl_operator.jl, test/test_magnetic_damping.jl | Stencil regression and independent damping/null-mode checks. |
| src/cpu_compatible.jl, test/test_cpu_compatible.jl | Allocation-reduced RHS, selector and finite equivalence/RNG/allocation tests. |
| Project.toml, Manifest.toml, bench/run_linux.sh | Isolated pinned runtime and caller-configured launcher. |
| bench/fixtures.jl, bench/benchmark_cpu.jl | Reproducible states and matched timing/error/provenance data. |
| bench/compare_results.py | Manifest validation and English figures/report. |
| docs/performance/README.md | Reproduction commands, measurements, qualification and next-route decision. |

## Task 1: Correct and freeze the reference

**Consumes:** legacy master6db4c30 and correction1969372.
**Produces:** a separate correctness commit and immutable source/config hashes.

1. Add a failing independent test before changing the equation:
~~~julia
using Test
ENV["NOFASTSCAN_SKIP_MAIN"]="1"
ENV["NOFASTSCAN_SKIP_WORKERS"]="1"
include(joinpath(@__DIR__,"..","current_sweep_thermal_nofastscan.jl"))
p=TDGLThermalParams(Nx=17,Ny=17,hx=0.5,hy=0.5,kappa=2.0,
                    sigma=1.0,Je=0.0,He=0.0)
c=TDGLThermalCache(p)
fill!(c.state.Ay,0.0)
for j in axes(c.state.Ax,2), i in axes(c.state.Ax,1)
    c.state.Ax[i,j]=sin(2pi*(j-1)/p.Ny)
end
fill!(c.Jsx,0.0); fill!(c.Jsy,0.0)
compute_dA_dt!(c.dAx,c.dAy,c.state,p,c.Jsx,c.Jsy,c.sigma)
@test sum(c.state.Ax[2:end-1,2:end-1].*c.dAx[2:end-1,2:end-1])<0
fill!(c.state.Ax,1.0)
compute_dA_dt!(c.dAx,c.dAy,c.state,p,c.Jsx,c.Jsy,c.sigma)
@test maximum(abs,c.dAx)==0
@test maximum(abs,c.dAy)==0
~~~
2. On Task 2's Linux runtime, demonstrate the sign assertion fails on legacy code; a dependency error is not the expected failure.
3. Port only solver/operator-test changes from1969372. Its packaging-tool change targets an absent reviewer file and is excluded with attribution.
4. Run damping, existing stencil and symbolic tests:
~~~bash
NOFASTSCAN_SKIP_WORKERS=1 julia --project=. test/test_magnetic_damping.jl
NOFASTSCAN_SKIP_WORKERS=1 julia --project=. test/test_full_curl_operator.jl
python3 test/verify_full_curl_sympy.py
~~~
5. Commit the correctness change separately; save source/config SHA-256 and full commit in reference manifests.

## Task 2: Isolated Linux runtime

**Consumes:** dependency declarations in paper/nbse2_fig3/Project.toml.
**Produces:** root Project.toml/Manifest.toml, dedicated server checkout, explicit launcher.

1. Resolve the four archived dependency names (HDF5, OrdinaryDiffEq, Plots, YAML) plus benchmark dependencies from the official Julia registry; set julia compatibility to1.12. The archived OrdinaryDiffEq UUID is malformed, so do not copy its UUID declaration. Resolve into this project only:
~~~julia
using Pkg
Pkg.activate(".")
Pkg.add(["HDF5","OrdinaryDiffEq","Plots","YAML","JSON3","BenchmarkTools"])
Pkg.instantiate()
Pkg.precompile()
~~~
Use JULIA_NUM_PRECOMPILE_TASKS=4 and OPENBLAS_NUM_THREADS=1. Retain the resolved manifest; do not guess current package versions or alter global environments.
2. Clone/fetch the authorized branch in a dedicated server directory. Verify HEAD/source hashes. Existing unrelated Python simulations are not this reference.
3. Set GKSwstype=100 for headless imports. Tests/bench imports set NOFASTSCAN_SKIP_MAIN=1 and NOFASTSCAN_SKIP_WORKERS=1.
4. Create this caller-configured launcher:
~~~bash
#!/usr/bin/env bash
set -euo pipefail
test -n "$TDGL_JULIA"
test -n "$TDGL_OUTPUT_ROOT"
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=4 GKSwstype=100
exec "$TDGL_JULIA" --project=. --startup-file=no \
  bench/benchmark_cpu.jl --out "$TDGL_OUTPUT_ROOT" "$@"
~~~
5. Record Julia/package/CPU/GPU/driver metadata, run reference checks and commit small environment files. Keep addresses and trajectories out of Git.

## Task 3: Compatible CPU RHS

**Consumes:** TDGLThermalCache, TDGLThermalParams and reference component functions.
**Produces:** vector_to_state_compatible!(state,u,p), fdm_rhs_compatible!(du,u,cache,t), resolve_rhs_backend(name::AbstractString).

1. Add tests using all ten configurations/four seeds from the spec. For every packed field, include boundary rows and require normalized max error<=1e-11:
~~~julia
Random.seed!(seed)
fdm_rhs_thermal!(du_ref,u,cache_ref,0.0)
tail_ref=randn(16)
Random.seed!(seed)
fdm_rhs_compatible!(du_fast,u,cache_fast,0.0)
tail_fast=randn(16)
@test maximum(abs,du_fast-du_ref)/(1+maximum(abs,du_ref))<=1e-11
@test tail_fast==tail_ref
~~~
Run before implementation and confirm missing-function failure.
2. Implement indexed copies with cumulative offsets; keep all component equations and ordered calls unchanged. A helper for real fields is:
~~~julia
function copy_real_field!(dest,u,offset)
    @inbounds for k in eachindex(dest)
        dest[k]=real(u[offset+k])
    end
    return nothing
end
~~~
Copy psi directly, call existing link/boundary/dpsi/supercurrent/dA/heat/dT functions in reference order, and pack derivatives by indexed loops.
3. Implement:
~~~julia
function resolve_rhs_backend(name::AbstractString=get(ENV,"TDGL_RHS_BACKEND","reference"))
    name=="reference" && return fdm_rhs_thermal!
    name=="cpu_compatible" && return fdm_rhs_compatible!
    throw(ArgumentError("unknown TDGL_RHS_BACKEND: "*name))
end
~~~
Controller includes the file after reference RHS types/methods inside worker definitions. Resolve once per ODEProblem. Keep the same number/order of diagnostic RHS calls, using the selected backend.
4. Warm both functions and measure @allocated. Require less candidate allocation on60x30 and120x60; avoid wall-time assertions in unit tests.
5. Run finite operator/RNG/dispatch/allocation tests and commit kernel separately from later measured refinements.

## Task 4: Matched comparison harness

**Consumes:** build_params_from_config(load_config_yaml(path)), with_current, state_to_vector, both RHS functions, voltage and streamed-output helpers.
**Produces:** immutable manifests, JSON/CSV errors/timings and English plots.

1. Define:
~~~julia
make_case(config_path::String; Je::Float64,noise::Float64,seed::Int,
          preparation::Symbol,Nx::Int=60,Ny::Int=30)
run_case(case,backend::String; horizon::Float64,tolerance::Float64,
         output_dir::String,save_fields::Bool=true)
normalized_error(a,b)=maximum(abs,a-b)/(1+maximum(abs,b))
~~~
make_case returns parameters, packed initial state and metadata. Persist initial arrays/hash before timing. Scaling preserves physical domain.
2. run_case uses separate caches, recorded trajectory seeds, Tsit5 and exact snapshot stops. Retain original diagnostic reevaluation and full streamed HDF5 layout for production comparisons. Record V/T/|psi|, fields/heat and actual accepted/rejected/RHS statistics.
3. Add tests rejecting different initial hashes, horizons, noise or output settings, and failed/nonfinite/incomplete solves. Never compare such runs as speedups.
4. Alternate backend order. Record all repetitions, warm allocations/RHS/solver time, HDF5/animation cost, cold launch and resource contention.
5. Python report validates manifests, plots paired runtime/speedup, traces/errors and available I–V curves in native units; label passed/failed/inconclusive gates. Commit harness and smoke-test before longer runs.

## Task 5: Linux measurements and optimization

**Consumes:** corrected frozen reference, accepted CPU backend and harness.
**Produces:** measured bottlenecks, qualified CPU speedup and reproducible comparison artifacts.

1. Execute finite operator matrix and short deterministic cases at Je0.06/0.18/0.27 with cool/hot nontrivial states. Apply exact convergence/error formulas and finite reference tolerance ladder from the spec.
2. Profile noisy and deterministic normal RHS/solves; save allocation/time fractions. Select further changes only from measured bottlenecks.
3. For cached coefficients/fused repeated quantities/temporary removal: add a covering failing equivalence test, implement, run covering tests, benchmark and retain only supported improvements. Preserve solver/noise policy.
4. Run configured-noise5000 cases and finite paired ensembles/tolerance sensitivity. Noise-off success alone does not promote production code.
5. Run12-point worker pilots; only finalists enter full99-point or50000 continuation gates within the24-hour cap. Respect state dependencies and label deferred/incomplete work.
6. Publish English commands, source hashes, errors, uncertainty, resources and practical speedups. Commit/push small reports; preserve large trajectories outside Git.

## Task 6: Next-route decision

**Consumes:** Task5 profiling evidence.
**Produces:** separate concrete GPU/integrator plan when justified, with a published reason.

1. Independent-trajectory or large-grid dominance triggers a device-resident fused full-curl GPU plan, Float64 first, reusing operator fixtures and manifest comparison interfaces.
2. Step-count/stability dominance triggers a coupled sparse IMEX plan whose prerequisite test is:
~~~julia
@test normalized_error(L*u+nonlinear_residual(u),reference_rhs(u))<=1e-11
~~~
That plan defines the complete boundary-conditioned coupled L/residual including affine forcing; scalar independent magnetic Laplacians are not acceptable.
3. If neither condition holds, continue measured CPU work. Conditional routes are decisions to implement from evidence, not claims of completed GPU/IMEX code.

## Execution and self-review

The user approved implementation and specified controller/Luna ownership. Apply that division directly: controller handles reference/runtime/integration/campaign, one Luna owns Task3's new files, another Task4's new files. No concurrent shared-solver edits. Controller reviews spec compliance/code quality and runs focused Linux validation; consult Astra only for new consequential numerical uncertainty or persistently blocked debugging.

Self-review checks: Tasks1–2 cover correctness/runtime; Task3 covers compatible kernel; Task4 covers artifacts/failure handling; Task5 covers all finite validation/performance gates; Task6 schedules evidence-based later routes without pretending they are implemented. Interfaces above are binding; issue progress lives in bd. Commit bounded changes and push the authorized performance branch; no force-push or master merge.
