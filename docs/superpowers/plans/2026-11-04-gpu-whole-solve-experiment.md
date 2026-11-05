# Deterministic GPU Whole-Solve Experiment

The full Float64 GPU RHS passed fieldwise comparisons on the target CUDA device. Its 99-state batch takes about 0.186 ms versus 6.16 ms for one CPU core processing those states. This justifies an integration experiment under the approved GPU design; those numbers are not solver speedups.

## Scope and ownership

The controller owns integration, target execution and interpretation. A bounded Luna worker owns `bench/benchmark_gpu_solve.jl` and any focused norm test. Existing production source, defaults and the legacy random forcing remain unchanged. Track execution in bd.

## Integration

Use the existing supported deterministic geometry and ComplexF64 packed columns. CUDA state and RHS scratch remain resident. Run adaptive Tsit5 with an error norm equal to the maximum of the individual column RMS norms. A global RMS over the entire batch could dilute an inaccurate trajectory and is not acceptable. GPU reductions and scalar adaptive-control synchronization are included in timed solves; compilation, allocation and explicit transfers are separately identified.

Reject nonzero noise. A shared adaptive step sequence differs from independent CPU solves, so compare achieved errors rather than assume trajectory identity. Save at common integer times and match stop schedules. Record status, final time, step/RHS counts, nonfinite values, real-field imaginary leakage and temperature range. Do not describe this as an HDF5/animation or configured-noise production benchmark.

## Accuracy and timing

An optional RHS graph probe replays only the autonomous deterministic kernel pipeline. Key executable graphs by the exact input/output buffer pair, test changed input contents, and count capture/instantiation separately. The complete adaptive integrator is never captured. Current/geometry scalars stay fixed for one cache, and nonzero noise remains rejected. This follows the [CUDA.jl graph API](https://cuda.juliagpu.org/stable/usage/graphs/), which fixes captured array addresses and scalar arguments.

First smoke-test a 7 by 5 grid over 2 time units. Then use the root 60 by 30 grid, horizon 100, currents cycling through 0.06/0.18/0.27, nontrivial seeded cool states, and batches 1/8/32/99. Evaluate CPU Tsit5 tolerance ladders through at least 1e-8 and 1e-9 for matched states, with explicit reference convergence. Compare all psi/Ax/Ay/T coordinates at common stored times using normalized maximum errors and conservative budgets derived from the CPU ladder. Failure of convergence or a component gate prevents acceptance.

Warm both complete paths. Alternate at least three timing repetitions, synchronize GPU work, and retain each sample. Report total CPU one-core aggregate time, GPU batch time and measured multi-process CPU throughput separately. Keep memory below 8 GiB and preserve existing GPU services. A faster deterministic solve remains experimental until the approved output, long-time and legacy-noise gates are resolved.
