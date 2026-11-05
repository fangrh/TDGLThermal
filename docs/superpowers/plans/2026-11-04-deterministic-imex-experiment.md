# Deterministic Coupled IMEX Experiment Implementation Plan

> **For agentic workers:** Use the approved controller/Luna division and bd progress tracking. This is an experimental solver path, not a production default.

**Goal:** Test whether a coupled magnetic/thermal implicit block accelerates the corrected deterministic full-curl problem at matched achieved accuracy.

**Architecture:** Assemble the actual affine A/T spatial map by basis probes through existing boundary and component functions. Keep its constant linear part L implicit and use N(u)=F_reference(u)-L*u explicitly. Preserve complete derivative-based heat production in F_reference before subtraction.

**Tech Stack:** Julia 1.12.1, SparseArrays, OrdinaryDiffEqSDIRK/KenCarp4, existing CPU reference/compatible kernel and JSON3.

**Spec:** docs/superpowers/specs/2026-11-04-linux-full-curl-performance-design.md, alternative-integrator section.

## Global constraints

- Preserve full coupled curl-curl mixed derivatives, all packed boundary coordinates, affine forcing and thermal feedback.
- Packed state remains ComplexF64; A/T imaginary parts remain zero on the physical subspace.
- Reference/default production backend stays unchanged.
- Noise must be zero for this first experiment; it cannot claim configured-noise or production equivalence.
- Report operator assembly, factorization/solver time, accepted/rejected steps, errors and numerical limitations.
- No added gauge damping, removed null modes, independent scalar magnetic Laplacians or fast-math.
- Use an isolated experiment environment and the existing 24-hour campaign cap.
- No shared solver/runtime edits by the numerical worker; no subagents spawned by workers.

## Files and interfaces

- src/imex_experiment.jl: IsolatedSpatialCache, isolated_spatial!(out,u,cache), build_spatial_split(p), ImexContext, imex_residual!(du,u,context,t).
- test/test_imex_split.jl: independent affine/operator, background/forcing, recombination/input-immutability and imaginary-leakage checks.
- bench/benchmark_imex.jl: deterministic Tsit5/KenCarp4 experiment, no CLI default change.
- experiments/imex/Project.toml and Manifest.toml: controller-owned pinned environment with OrdinaryDiffEqSDIRK added to existing dependencies.
- docs/performance/imex_experiment.md: measured result and promotion/retirement decision.

## Task 1: Exact spatial map and split

1. Add tests before implementation. For small anisotropic grids with heterogeneous sigma/eta and the approved boundary options:
~~~julia
split=build_spatial_split(p)
isolated_spatial!(g,u,split.cache)
@test maximum(abs,g-(split.L*u+split.affine))/(1+maximum(abs,g))<=1e-11
~~~
Check two probe amplitudes (1 and0.5), all packed A/T columns including boundary coordinates, several independent psi backgrounds, and nonzero Je/He. The isolated map must ignore psi when Js and W arezero.
2. Implement the isolated map using separate scratch state/cache: unpack state, apply production boundaries, zero supercurrents/heat, call compute_dA_dt! and compute_dT_dt!, and pack zero psi plus A/T derivatives. Honor thermal_enabled=false.
3. Assemble columns from G(h*e_j)-G(0), divided byh. Reset complete scratch state on each probe; retain actual boundary rows and all nonzero coefficients in a sparse ComplexF64 matrix. Never use candidate numerical results to fit L.
4. Validate current/field slope invariance. Affine forcing stays entirely in the residual:
~~~julia
function imex_residual!(du,u,ctx,t)
    fdm_rhs_compatible!(du,u,ctx.reference_cache,t)
    mul!(ctx.linear_work,ctx.L,u)
    du .-= ctx.linear_work
    return nothing
end
~~~
Test recombination field by field against F_reference on independent states, and assert u unchanged. This algebraic identity is separate from the independent G affine test.
5. Run focused tests and commit the experimental module/tests.

## Task 2: Isolated solver experiment

1. Controller copies the pinned CPU Project/Manifest to experiments/imex and adds OrdinaryDiffEqSDIRK by package name. Do not mutate the CPU benchmark environment while its runs execute.
2. Confirm current SciML's linear-operator split API from official documentation and installed methods. Use analytic constant L for the implicit part, avoiding differentiation of the nonholomorphic residual:
~~~julia
prob=SplitODEProblem(ctx.L,imex_residual!,copy(u0),(0.0,horizon),ctx)
solution=solve(prob,KenCarp4();abstol=tolerance,reltol=tolerance)
~~~
If the installed API needs a MatrixOperator/explicit constant Jacobian wrapper, use that documented interface and keep the same L and residual.
3. Add a small test that completes with real A/T to1e-12 and agrees with a tightened explicit reference at common exact stops. A completed solve alone is not a pass.
4. Benchmark both algorithms after full-path warmup on Je0.06/0.18/0.27, cool/hot preparations, horizon100, noise0. Use root60x30 physical parameters, identical initial arrays and common integer-time stops. Apply the approved finite reference-tolerance ladder and achieved-error budget.
5. Record assembly and solve time separately; include assembly in cold/amortized comparisons. Repeat accepted cases five times in alternating algorithm order.
6. Label result deterministic-only. If it fails accuracy, is slower, or leaves explicit psi/heating step restrictions dominant, record that result and retain the compatible CPU backend. Configured-noise validation requires a separately identified experiment, not silent promotion.

## Review and delivery

Controller reviews the module's operator proof, boundaries, residual heating and benchmark truthfulness. Selective Astra consultation already confirmed the algebraic split and highlighted explicit psi/heating restrictions; measurements decide benefit. Commit/push small code and receipts; large trajectories remain outside Git. This plan tests a justified candidate and does not guarantee acceleration.
