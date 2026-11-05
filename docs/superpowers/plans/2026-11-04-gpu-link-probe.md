# GPU Link-Kernel Feasibility Implementation Plan

> **For agentic workers:** Controller owns environment/results; one bounded Luna worker owns the probe file. Use bd tracking.

**Goal:** Measure whether the remaining link-phase bottleneck benefits from device-resident GPU batching before attempting a complete GPU RHS port.

**Architecture:** Benchmark the exact link operation on preallocated CPU/GPU arrays at batches1,8,32,99, with Float64 first. This probe is not a TDGL solver and cannot claim end-to-end simulation acceleration. Data remains resident during timing; report host/device transfer cost separately.

**Tech Stack:** Julia1.12.1, CUDA.jl, JSON3, Statistics, host reference cis loops.

**Spec:** docs/superpowers/specs/2026-11-04-linux-full-curl-performance-design.md, GPU candidate and performance sections.

## Global constraints

- No production/default changes, reduced physical precision or full-model claim.
- Compare exactly the same input values, complex link convention and grid.
- Synchronize timed GPU work; warm compilation outside measured samples.
- Keep existing GPU services unchanged; allocate less than1GiB and record device/memory/driver information.
- Keep addresses/credentials out of Git.
- Preserve the main CPU environment; pin a separate experiments/gpu_probe environment.

## Task 1: Implement bench/gpu_link_probe.jl

1. Define cpu_links!(Ux,Uy,Ax,Ay,hx,hy) using the compatible cis operation and a fused GPU kernel:
~~~julia
i=(blockIdx().x-1)*blockDim().x+threadIdx().x
if i<=length(Ax)
    Ux[i]=cis(-hx*Ax[i])
end
if i<=length(Ay)
    Uy[i]=cis(-hy*Ay[i])
end
~~~
2. Host fixtures use seed20261104, grid60x30, spacings0.5, and three-dimensional arrays with the batch as final axis. Ax shape=(60,31,B), Ay=(61,30,B).
3. Compare every GPU link entry against the host reference with normalized maximum error<=1e-11. Record maximum error, not only a Boolean.
4. Warm each path, then time at least five paired samples of repeated operations. Include CUDA synchronization in elapsed GPU time. Report per-call time and full batch throughput; do not confuse batch speedup with one-trajectory speedup.
5. Measure explicit upload/download cost separately. Include Float64 only in this first probe; any Float32 experiment is separately labelled approximate.
6. Save JSON to an explicit output argument, including source hash, device/runtime, shapes/batch, precision, timing samples, error and scope='link-kernel-only'.

## Task 2: Environment and decision

Controller copies the CPU Project/Manifest into experiments/gpu_probe and adds CUDA without mutating the CPU environment. Run on the authorized server, using existing available GPU capacity.

Use measured link fraction (roughly40% of compatible CPU RHS after cis optimization) to report Amdahl's bound for a link-only improvement. A favorable kernel result justifies a later fully resident fused RHS plan; it does not justify transferring arrays every CPU RHS call. A slower result is retained as evidence and does not replace the working CPU path.
