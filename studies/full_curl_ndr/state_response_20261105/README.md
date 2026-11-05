# Local current-step response from prepared full-curl states

The user selected physical parameters and measurement protocols as the optimization direction. This bounded exploratory diagnostic returns to the original `T0 = 0.18` region, where the independently prepared points showed a reproducible 14.32% voltage decrease. The preceding [thermal-parameter search](../enhancement_20261105/README.md) did not enhance that feature. Its null result and the original full I–V remain unchanged.

## Verified outcome: locally positive response

All 120 segments / 24 histories completed. Hashes, source/initial-state/RNG lineage, schedules and all nine saved numeric datasets passed validation. Common conditioning has identical numeric dataset bytes, packed initial vectors, RNG checkpoints and solver counts across the three arms in every source/seed group. The [final audit receipt](results/final_validation_receipt.json) also matches all 120 individual receipts to the aggregate campaign and confirms that the owned simulation session has ended.

The [English response figure](results/state_response.png) ([PDF](results/state_response.pdf)), [complete analysis](results/state_response_analysis.json), [final seed rows](results/state_response_rows.csv) and [120 segment rows](results/state_response_segment_rows.csv) preserve every arm. Final 18000–20000τ means are:

| Source current | V minus | V sham | V plus | Mean sham-minus | Mean plus-sham |
|---|---:|---:|---:|---:|---:|
| 0.2091836735 | 2.930854 | 2.942102 | 2.952490 | +0.011248 | +0.010387 |
| 0.2102040816 | 2.505568 | 2.515040 | 2.524560 | +0.009472 | +0.009520 |

All **16 seed-level neighboring-arm differences are positive**. None meets the frozen negative-step criterion. The four Bonferroni-adjusted one-sided upper bounds are `+0.013371`, `+0.012415`, `+0.012770` and `+0.015474`. Centered finite-response secants are approximately `+42.41` and `+37.22` native units; these are descriptive differences between fork means, not stationary derivatives. The maximum final-window sampled-versus-Ax-integrated voltage discrepancy is `0.001532` native units.

All three inner1 arm groups pass the nominal drift-equivalence rule. All three inner2 arm groups fail: minus CI `[-0.006425, 0.002844]`, sham `[-0.004242, 0.002713]`, plus `[-0.005129, 0.008188]`. Positive finite-window responses do not make these latter trajectories stationary.

The [spatial figure](results/fields/state_response_fields.png) ([PDF](results/fields/state_response_fields.pdf), [provenance](results/fields/field_provenance.json)) illustrates the prespecified first seed at cumulative time 75000τ. Its condensate and temperature patterns differ across the two source neighborhoods. These single-time snapshots do not identify vortex/phase-slip events or establish the cause of the original decrease.

This protocol did not enhance NDR. It resolves locally positive finite-time response around both original prepared states, while the intervening current gap and a continuous stationary negative branch remain unresolved. A subsequent physical-parameter choice should be informed by resolved state dynamics rather than interpreting the original prepared-point jump as an established negative derivative. Eleven preparation and twelve analysis fixture/tamper tests pass; final Linux artifact checks and both real-data figure runs also pass.

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

The two local current ranges are approximately `0.208929–0.209439` and `0.209949–0.210459`. They leave an untested gap inside the original decrease. Positive responses in both local neighborhoods would not exclude a narrower negative interval or switching within that gap. These separate source-state forks do not trace one branch through the entire original interval.

## Provenance

The existing `bench/reproduce_ndr_history.jl` runner supports abrupt current changes with full field/RNG continuation. Its source/configuration guards are retained. The validated streaming progress guard and explicit `TDGL_STREAM_MAXITERS=10000000` protect long integration. No PDE or RHS source is changed.

Protocol preparation verifies the eight source HDF5/RNG and chain/sidecar receipts. Final analysis verifies the complete 24 × 5 matrix, source/initial-state/RNG lineage, hashes, saved schedules, all nine numeric datasets and common conditioning. English figures and seed-level CSVs report every arm.

The [protocol](plan/protocol.json) and [24 jobs](plan/state_response_jobs.json) were pushed in `4600220` before launch on 2026-11-05 at 06:53:56 UTC. The [launch receipt](results/launch.json) records the separate runtime pinned to `5c80eba`, every Julia harness/model hash, dependency-lockfile hash, process ID, exact arguments and environment. Python analysis overlays are recorded separately and do not change the frozen Julia simulation sources.

The raw-data root is `/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_state_response_20261105_v1`. The runner is `bench/reproduce_ndr_history.jl --horizon 5000 --skip 0.6 --backend cpu_compatible --tolerance 1e-4`, with the frozen jobs and 24 persistent one-thread workers. Analysis uses `bench/analyze_ndr_state_response.py --protocol .../plan/protocol.json --campaign-root .../campaign --out .../analysis --validate-artifacts`.
