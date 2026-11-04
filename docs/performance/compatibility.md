# Compatible CPU procedure

The CPU path changes storage copies and the link-phase evaluation; it reuses every original physics component, including boundary reconstruction, full mixed magnetic derivatives, derivative-based heating and the legacy random draws. It keeps the same integrator, packed coordinates, stop times and retained-snapshot diagnostic calls.

For the pinned Julia 1.12.1 build, let `theta = (-hx)*A`. The original complex product has real part signed zero and imaginary part `theta`. For finite phase products, complex `exp` evaluates `exp(0)` and `sincos(theta)`; `cis(theta)` uses the same `sincos`. Multiplication by exactly one preserves those finite results. See the pinned [complex multiplication and cis implementation](https://github.com/JuliaLang/julia/blob/ba1e628ee49351af0b704afd2b2903d253bd3564/base/complex.jl#L333-L343) and [complex exp implementation](https://github.com/JuliaLang/julia/blob/ba1e628ee49351af0b704afd2b2903d253bd3564/base/complex.jl#L701-L720).

This is a source-level finite-phase identity, rather than a relaxed-accuracy trigonometric approximation. Nonfinite phase products have different failure behavior and are outside this compatibility argument. Indexed unpack/pack copies preserve the stored values and traversal-independent fields; they do not move or add random draws.

The executable checks additionally cover 80 complete-RHS cases with RNG-tail comparisons, 30 short full-output pairs, 24 configured-noise 5000-time-unit pairs, 540 worker-pilot pairs, and six tighter-tolerance pairs. Their stored numerical values and solver-call counts agree exactly. HDF5 file bytes need not match because artifacts contain distinct metadata. These checks establish implementation compatibility for the tested procedures, not experimental validity or convergence of the underlying physical model.

GPU, lower-precision and new-integrator candidates do not inherit this identity argument; their separate achieved-accuracy gates remain necessary.
