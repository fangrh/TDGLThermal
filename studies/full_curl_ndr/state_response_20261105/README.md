# Local current-step response from prepared full-curl states

The user selected physical parameters and measurement protocols as the optimization direction. This bounded exploratory diagnostic returns to the original `T0 = 0.18` region, where the independently prepared points showed a reproducible 14.32% voltage decrease. The preceding [thermal-parameter search](../enhancement_20261105/README.md) did not enhance that feature. Its null result and the original full I–V remain unchanged.

## Frozen question and budget

Does a small current increase lower the late voltage when starting from a common prepared dynamical state, or does the independently prepared decrease reflect different accessible responses? A same-current sham arm measures continuing relaxation. The original eight-seed finite-window drift checks passed at both inner currents, `Je = 0.20918367346938774` and `0.21020408163265306`; this does not establish their indefinite stability.

TDGL studies distinguish dissipative vortex/phase-slip states from evolving hot spots. This motivates examining voltage, temperature and condensate fields together, without assuming the mechanism in this geometry: [Berdiyorov et al., Phys. Rev. B 90, 054506 (2014)](https://doi.org/10.1103/PhysRevB.90.054506).

The fixed matrix uses the first four original seeds, `2026110411`–`2026110414`, at both saved 50000τ inner-current states. These seeds are reused for exploration; this is not a fresh confirmatory cohort. Each source/seed is forked into three arms:

| Arm | Target current |
|---|---|
| Minus | `source Je − delta` |
| Sham | `source Je` |
| Plus | `source Je + delta` |

Here `delta = (0.21020408163265306 − 0.20918367346938774) / 4`, approximately `0.000255102`. Each arm first receives the same 5000τ conditioning at its source current, followed by four 5000τ segments at the target. All three conditioning trajectories must have identical saved-array bytes, packed initial vectors and RNG checkpoints within each source/seed. The perturbation response lasts 20000τ after conditioning, reaching cumulative time 75000τ.

There are 24 histories and exactly **120 segments / 600000τ**. No additional physical-parameter search or fresh-seed confirmation is automatically added to this diagnostic. The jobs, source identities and protocol are frozen and pushed before launch.

## Measurement and interpretation

All arms retain corrected full mixed curl, the exact original physical configuration, Float64, Tsit5, tolerance `1e-4`, in-RHS Gaussian forcing `0.001`, grid `60 × 30`, spacing `0.5`, and snapshots every 1τ. Every 5000τ segment retains its final 3000–5000τ window. Fixed segment/restart and diagnostic schedules are shared across arms; this is not equivalent to an uninterrupted 20000τ solve or a calibrated Brownian SDE.

The four response windows are 3000–5000τ, 8000–10000τ, 13000–15000τ and 18000–20000τ after the current step. The final comparison uses the last window. Drift is the last window minus 13000–15000τ. Independent seed trajectories, rather than frames or the three arms, determine uncertainty.

For each of the two source states, paired neighboring-arm differences are `V(sham) − V(minus)` and `V(plus) − V(sham)`, with four seeds per contrast. A descriptive negative-step candidate requires its one-sided Student 95% upper bound, Bonferroni-adjusted over all four contrasts, below `−0.01` native voltage units. Nominal two-sided 95% drift intervals are compared with `±0.0025`. Centered plus-minus differences are also reported descriptively. No selection of a different source, seed subset or window replaces these rules.

Even a negative local step response is not proof of a continuous stationary branch: each fork can switch dynamical states. Temperature/condensate changes are correlations, and four reused seeds give limited population evidence. Current-step transients, later relaxation and spatial fields are retained for interpretation. Spatial convergence, calibrated thermal noise, and an experimentally fitted physical parameter set remain outside this diagnostic.

## Provenance

The existing `bench/reproduce_ndr_history.jl` runner supports abrupt current changes with full field/RNG continuation. Its source/configuration guards are retained. The validated streaming progress guard and explicit `TDGL_STREAM_MAXITERS=10000000` protect long integration. No PDE or RHS source is changed.

Protocol preparation verifies the eight source HDF5/RNG and chain/sidecar receipts. Final analysis verifies the complete 24 × 5 matrix, source/initial-state/RNG lineage, hashes, saved schedules, all nine numeric datasets and common conditioning. English figures and seed-level CSVs report every arm.

The [protocol](plan/protocol.json) and [24 jobs](plan/state_response_jobs.json) were pushed in `4600220` before launch on 2026-11-05 at 06:53:56 UTC. The [launch receipt](results/launch.json) records the separate runtime pinned to `5c80eba`, every Julia harness/model hash, dependency-lockfile hash, process ID, exact arguments and environment. Python analysis overlays are recorded separately and do not change the frozen Julia simulation sources.

The raw-data root is `/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_state_response_20261105_v1`. The runner is `bench/reproduce_ndr_history.jl --horizon 5000 --skip 0.6 --backend cpu_compatible --tolerance 1e-4`, with the frozen jobs and 24 persistent one-thread workers. Analysis uses `bench/analyze_ndr_state_response.py --protocol .../plan/protocol.json --campaign-root .../campaign --out .../analysis --validate-artifacts`.
