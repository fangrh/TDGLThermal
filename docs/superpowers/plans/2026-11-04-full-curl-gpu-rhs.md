# Device-Resident Full-Curl GPU RHS Implementation Plan

> **For agentic workers:** Use controller integration plus one bounded Luna implementer; track progress in bd. This is an opt-in deterministic experiment, not a production default.

**Goal:** Port the full corrected deterministic RHS to CUDA and measure device-resident batch performance after fieldwise equivalence.

**Architecture:** Keep the original packed ComplexF64 coordinates, Float64 physical fields and host-generated sigma/eta/Tc maps. Separate unpack/original-link, ordered boundary reconstruction, updated links, coupled psi/A and heat/temperature kernels. The first supported geometry is exactly the root configuration: periodic x, Neumann y, no loop, both field sides, linear epsilon and psi_y_zero=false. Unsupported modes and nonzero noise are rejected explicitly.

**Tech Stack:** Julia1.12.1, CUDA.jl, existing root TDGL types and coefficient helpers, JSON3.

**Spec:** docs/superpowers/specs/2026-11-04-linux-full-curl-performance-design.md GPU section.

## Global constraints

- Preserve full curl-curl mixed derivatives and corrected damping sign.
- Preserve exact ordering of boundary reconstruction and link updates, including any old-link use during boundary reconstruction.
- No scalar CPU fallback, fast-math, lower precision or noise suppression in a claimed production run.
- First experiment is noise-off only; it cannot claim configured-noise/production equivalence.
- Keep production source/default unchanged.
- GPU arrays stay resident during RHS timing; transfers and allocation/setup are separate.
- Use less than8GiB of available GPU memory and preserve existing services.
- Reject unsupported geometry explicitly rather than silently substituting boundaries.

## Files and interfaces

- src/gpu_full_curl.jl: GPUFullCurlCache, GPU-safe scalar parameter bundle, kernel pipeline.
- test/test_gpu_full_curl.jl: actual CUDA complete-RHS comparison and unsupported-option tests.
- bench/benchmark_gpu_rhs.jl: device-resident batch RHS timing and source/environment receipts.

Required interfaces:

~~~julia
GPUFullCurlCache(p::TDGLThermalParams, currents::Vector{Float64})
gpu_full_curl_rhs!(du, u, cache::GPUFullCurlCache, t)
~~~

u and du are CuArray{ComplexF64,2}, with shape(packed_state_length,batch_count). Columns correspond to independently supplied states/currents. The cache owns separate psi/A/T/link/derivative scratch arrays; constant material maps are computed by the host reference helpers then uploaded unchanged.

## Task 1: Full spatial RHS equivalence

1. Write CUDA tests before the module. Root configuration and anisotropic7x5 variants use deterministic nontrivial psi/A/T, currents0.06/0.18/0.27 and heterogeneous sigma/Tc/eta maps from host helpers. Compare every packed field, including all boundary rows:
~~~julia
fdm_rhs_thermal!(cpu_du, cpu_u, cpu_cache, 0.0)
gpu_full_curl_rhs!(gpu_du,gpu_u,gpu_cache,0.0)
CUDA.synchronize()
@test maximum(abs,Array(gpu_du)[:,b]-cpu_du)/(1+maximum(abs,cpu_du))<=1e-11
~~~
The actual test normalizes psi/Ax/Ay/T separately. Assert input u unchanged. Test unsupported boundaries/nonzero noise produce clear errors.
2. Implement complete packed unpack and zeroing of output entries. Precompute original links using the reference phase convention.
3. Translate the existing boundary routines for the supported geometry, preserving x-then-y ordering, partial electrode masks, current-induced field and corner behavior. Recompute links at the same place as the CPU routine. Inspect the actual code rather than assume Neumann/periodic formulas.
4. Translate KWT dpsi, supercurrent and coupled magnetic stencil. Preserve local mobilities and real projections. All node/edge indexing follows existing staggering.
5. Compute full derivative-based heat and temperature diffusion/cooling with the same sampled derivative arrays. Do not replace full derivatives with a residual or finite-difference approximation.
6. Initialize/overwrite every output component each call; do not rely on allocator-zero memory. Set CUDA.allowscalar(false).
7. Run actual Linux CUDA tests; retain errors and supported-configuration metadata. Fix any component mismatch before timing.

## Task 2: Batched RHS feasibility

1. Warm the complete GPU pipeline; measure batches1,8,32,99 on60x30, with five paired CPU/GPU samples and explicit synchronization. CPU reference uses the same input columns, ordered scalar Float64 calls, and preallocated caches.
2. Save fieldwise errors, per-batch times, transfers, source hashes, memory and device/runtime.
3. Report scope='full-RHS-only'. Compare one-core aggregate CPU throughput separately from estimated/observed multi-core throughput. A GPU RHS speedup is not a full solver speedup.
4. Only after this gate, write the integration plan for a complete GPU solve and common-time achieved-accuracy comparison. Keep output/legacy-noise validation separate from this deterministic kernel gate.
