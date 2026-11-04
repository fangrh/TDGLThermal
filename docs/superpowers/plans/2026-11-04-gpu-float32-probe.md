# GPU Float32 Feasibility Probe

The approved design permits independently validated lower precision after strict baselines. The compatible Float64 CPU campaign is exact on all stored fields in 24 production pairs, and the actual Float64 CUDA RHS passes all component and reconstructed-boundary checks. GPU FP64 throughput and profiling now motivate a bounded precision probe.

The worker owns only `src/gpu_float32_experiment.jl`, `test/test_gpu_float32_experiment.jl` and `bench/gpu_float32_probe.jl`; the controller owns interpretation and integration. Keep the existing full-curl CPU and Float64 GPU source unchanged. Track progress in bd.

Use distinct Float32 scalar/cache types and ComplexF32 packed storage, explicit material-map conversion and the same kernel equations/order. Reuse the generic kernel functions where possible. Any promoted intermediates must be described as mixed precision rather than claiming all-Float32 arithmetic. Keep the same deterministic geometry guards and reject configured nonzero noise.

Measure fieldwise complete-RHS errors against CPU Float64 on small anisotropic and root grids, input immutability, boundary fields and finiteness. The 1e-11 Float64 equivalence threshold is not a Float32 acceptance claim. Then warm and synchronize batches 1/8/32/99, alternating the Float64 and lower-precision pipelines for five repetitions. Record actual errors, timings, source hashes and resources. This first probe is RHS-only and does not establish achieved trajectory accuracy or production compatibility.

Only a sufficiently accurate and faster probe justifies a complete integration experiment with the approved CPU tolerance ladder, voltage/heat/magnetic observables, common stops, noise policy and production gates. Otherwise retain it as a rejected or inconclusive experiment.

The actual probe subsequently measured a 3.20-fold improvement over Float64 GPU RHS at batch 99, with maximum normalized derivative error 8.92e-6. This motivates `bench/benchmark_gpu_precision_solve.jl`: three alternating synchronized F64/F32-storage Tsit5 pairs, 99 matched states over 100 time units, maximum per-column RMS control and the saved per-state CPU reference ladders. Warm each path only over two time units. Verify initial-state/reference hashes and apply every predeclared field/observable budget to every state; unresolved reference convergence or any failure prevents batch promotion. Keep this deterministic experiment separate from production noise and HDF5-output performance.
