# Full-curl NDR enhancement exploration

This is a bounded physical-parameter exploration requested on 2026-11-05. The delivered [101-point baseline](../full_iv/README.md) remains the comparison dataset. The screen, independent 50000τ holdout, and guarded ordered histories are verified. The declared enhancement target failed. All tested ordered neighboring-current contrasts are positive; a stationary continuous negative branch is not established.

The [English final overview](results/final_overview/ndr_final_overview.png) ([PDF](results/final_overview/ndr_final_overview.pdf)) puts the original prepared-point response, new candidate, ordered histories, and frozen primary contrast together. The original approximately 14.3% prepared-point decrease remains the stronger verified reference in this comparison.

## Long-horizon result: enhancement target not met

All 56 fresh holdout chains and 560 segments passed output, initial-state, RNG, source/configuration, retained-schedule, and nine-dataset finite-value checks. The [verified holdout analysis](results/holdout/holdout_analysis.json) and [English comparison figure](results/comparison/ndr_enhancement_comparison.png) preserve every seed.

At the frozen primary contrast `Je = 0.19 → 0.20`, all eight seeds decrease. The mean paired decrease is **2.978%**, with nominal one-sided 95% lower bound **2.393%**. Mean `V_hi − V_lo = −0.07409`, with upper bound `−0.05950`. This confirms a small finite-time decrease, but fails the declared ≥20% mean and >15% lower-bound enhancement criteria.

| Contrast | Mean voltage decrease | Mean paired fraction | Current interval width | Secant of point means |
|---|---:|---:|---:|---:|
| Original `T0 = 0.18`, local comparison | 0.42041 | 14.32% | 0.0010204 | about −412.0 |
| Frozen primary `T0 = 0.22`, `0.19 → 0.20` | 0.07409 | 2.98% | 0.01000 | −7.41 |
| Exploratory local `T0 = 0.22`, `0.1975 → 0.20` | 0.35456 | 12.81% | 0.00250 | −141.83 |

The refined local contrast was identified after inspecting the holdout curve. It is exploratory and does not replace the frozen primary endpoint. The secants are finite-interval differences between independently prepared point means, not established derivatives along a continuous stationary branch. A smaller spacing across a jump can increase the secant magnitude.

The local decrease coincides with `T_avg` changing from 0.30220 to 0.29789 and mean `|ψ|²` from 0.44438 to 0.47415. This is consistent with a cooler, more superconducting response; the correlations do not determine causal direction or spatial uniformity. Six of seven currents pass the declared finite-window drift bound. The highest current, `0.2025`, fails, with drift CI `[-0.003601, 0.004948]`.

The 10000τ screen did not recover an eligible baseline decrease at that same horizon. It therefore cannot exclude slower NDR behavior in the other six configurations. `T0 = 0.22` is the screen-selected candidate, not an established optimum. The stronger original baseline is retained.

## Verified ordered-history outcome

All eight histories traversed seven currents with 20000τ per dwell. All 56 dwell outputs passed source/configuration/guard checks, HDF5 and initial-vector hashes, RNG and field-source continuity, retained-time schedules, and all nine saved-dataset finite scans. The [analysis](results/history_v2/history_analysis.json), [seed rows](results/history_v2/history_seed_rows.csv), [adjacent contrasts](results/history_v2/history_deltas.csv), and [V/temperature/condensate figure](results/history_v2/history_V_T_psi.png) preserve the complete result.

Every one of the 48 seed-level neighboring-current differences `V(higher Je) − V(lower Je)` is positive. Ascending mean differences range from 0.10176 to 0.10656; descending mean differences range from 0.08457 to 0.39819. None of the twelve direction/interval comparisons satisfies the declared negative-difference bound, and there are no consecutive confirmed negative intervals.

Five of fourteen direction/current combinations fail the nominal finite-window drift-equivalence criterion: ascending `0.1925`; descending `0.1925`, `0.195`, `0.20`, and `0.2025`. The remaining nine pass. These tests describe the recorded windows and do not rule out rare switching. The largest discrepancy between sampled voltage and the Ax-integrated voltage over the same 18000–20000τ window is 0.001315 native voltage units.

The ordered curves show different response levels for the two directions while retaining positive neighboring-current increments. Together with the prepared-point results, this supports sensitivity to preparation and scan protocol in this model. The comparison does not isolate preparation from dwell, restart, or the legacy forcing/evaluation schedule, and it does not identify a unique physical cause. It does not establish that full-curl NDR is impossible under other parameters or protocols.

The declared scientific campaign completed 252 screen segments + 560 holdout segments + 224 equivalent history segments = **1036 equivalent 5000τ segments**. Invalid ordered v1 output and separate software-validation probes are excluded from scientific results. The conditional refinement was not triggered because the frozen enhancement target failed. A follow-on should resolve local responses on identified dynamical states around the stronger original `T0=0.18` region, rather than silently widening this search or relabeling a post-hoc peak as confirmation. Spatial convergence and a calibrated stochastic thermal model remain outside this study.

## Verified screening result and frozen holdout

All 126 chains and 252 segments completed successfully. Output/initial-state/RNG hashes, chain continuity, retained schedules, and all nine saved numeric datasets passed validation with no errors. See [the screen analysis](results/screen/enhancement_analysis.json) and [all seven I–V/temperature/condensate curves](results/screen/enhancement_screen.png).

Only `T0 = 0.22` qualified at the declared 10000τ screen horizon: `Je = 0.19 → 0.20` decreased voltage by 2.89% and 1.19% in the two seeds. No other configuration, including the baseline at this same horizon, had an eligible decrease in both seeds. This is a small exploratory decrease, not evidence that the original 50000τ baseline drop has been improved.

The [frozen holdout protocol](holdout_plan/holdout_protocol.json) fixes `T0 = 0.22`, currents `0.1875, 0.19, 0.1925, 0.195, 0.1975, 0.20, 0.2025`, and eight fresh seeds. The primary contrast is exclusively `0.19 → 0.20`; choosing a different interval from the holdout results is not allowed. Each of the 56 chains reached 50000τ. No new-seed simulation was launched before freezing these files.

The [original ordered-history plan](history_plan/history_protocol.json) fixes the first four holdout seeds and both directions, starting from their saved outer-current fields/RNG states. All 56 × 10 holdout metadata/continuity records and all sixteen outer HDF5/RNG source hashes were checked before freezing this plan. Each of the eight histories traverses all seven currents with 20000τ dwell per point. The completed replacement run assessed branch accessibility without changing the failed primary enhancement result.

The first ordered-history attempt is [invalid and excluded](results/history_invalid_v1/invalid_run.json). Adaptive Tsit5 defaults to one million iterations. A 20000τ dwell needs more, and the original manual target-advance loop ignored `MaxIters`, repeatedly calling `step!` without advancing time. The owned processes were stopped and partial output preserved. Completed 5000τ segments have about 300000 accepted steps and their verified results remain valid.

The benchmark harness now checks return codes and strict time advancement, and records an explicit `TDGL_STREAM_MAXITERS` override. The PDE/model and RHS files are unchanged. Twenty-nine pinned-Linux tests and a distributed failure canary pass. The [guarded-versus-frozen noisy comparison](results/streaming_guard_validation/differential_summary.json), with iteration budget 10000000, has exact saved-array bytes, initial vectors and RNG states in all six point-segment pairs. The replacement ordered run retained the original currents, seeds, initial checkpoints, dwell, output windows and statistical gates and completed successfully.

The [replacement v2 protocol](history_plan_v2/history_protocol.json) and unchanged [eight jobs](history_plan_v2/history_jobs.json) were pushed in `b9cbebc` before the 2026-11-05 03:06 UTC launch. The separate runtime is pinned to `5c80eba`; the recorded guard source hash is `9c764ecaf9a58078394abc2f39651a7a5a07f054761e55c3ceeccfade9b6fa70`. Only this replacement run contributes ordered-history results.

For comparison, the original four-current/eight-seed 50000τ study has a 14.32% mean paired middle-interval drop. Its [English detail figure](results/baseline_detail/ndr_enhancement_comparison.png) retains every seed and the higher-current spread; this prepared-point response does not establish a stationary continuous negative branch. Connecting lines guide the eye between independently prepared points and do not represent an ordered current sweep.

## Full-curl identity

The model retains the full mixed magnetic operator:

```text
dAx/dt = [Jsx + kappa^2 (Dyy Ax - Dxy Ay)] / sigma_x
dAy/dt = [Jsy + kappa^2 (Dxx Ay - Dxy Ax)] / sigma_y
```

The implemented staggered-grid conductivity averages and boundary conditions are retained. The model SHA256 is `4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422`; the compatible RHS SHA256 is `900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe`. No magnetic term is removed or rescaled to enhance the response.

## Declared screen

Seven one-factor configurations test the thermal feedback in both directions:

| Configuration | Changed parameter |
|---|---|
| Baseline | None |
| Lower / higher conduction | `k_eff = 2.5 / 10` (baseline 5) |
| Lower / higher suspended-region cooling | `eta_hole = 0.05 / 0.2` (baseline 0.1) |
| Lower / higher bath temperature | `T0 = 0.14 / 0.22` (baseline 0.18) |

All cases retain Float64, Tsit5, tolerance `1e-4`, original in-RHS noise `0.001`, the 60 × 30 grid with spacing 0.5, `gamma = 40`, `kappa = 2`, `C_eff = 1`, `eta_env = 10`, and the original geometry and contacts. Changing bath temperature also changes the state-dependent noise amplitude, so it is a change to the coupled model rather than an isolated cooling perturbation.

Each configuration has nine declared currents and two fresh seeds, `2026110501` and `2026110502`, with independent hot preparations (`psi = 0`, `T = 1.2`). Two 5000τ segments give a common 10000τ screen horizon. The final 8000–10000τ means are compared with the earlier 3000–5000τ window. There are 126 chains and 252 new point-segments. See [the exact protocol](screen_protocol.json), [jobs](screen_jobs.json), and [configurations](configs/).

For each adjacent current pair, screening eligibility requires both seed voltages at the lower current to be at least 0.05 and both absolute voltage decreases to exceed 0.01. The score is the smaller of the two seed-specific fractional decreases. All configurations and seed-level outcomes will be shown. One nonbaseline configuration and current bracket can advance only if its score exceeds the baseline screen score. A null result closes this bounded search without silently widening parameters or changing thresholds.

Thermal transport is a justified first control: coupled TDGL/heat modeling and the suspended-region thermal response are discussed in [Fang et al., Advanced Electronic Materials](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aelm.70592). Stronger feedback need not improve NDR; it can instead enlarge hysteresis or prevent superconducting recovery. The ranges here are exploratory numerical settings, not calibrated material parameters.

## Confirmation and interpretation

Before examining new holdout seeds, freeze one candidate, one bracket and the analysis rules. The holdout uses eight new seeds (`2026110511`–`2026110518`) at seven currents: five equally spaced within the selected bracket and one shoulder one such spacing outside each end. Every chain reaches 50000τ; the comparison window is 48000–50000τ. Four holdout seeds also undergo ascending and descending ordered histories with 20000τ dwell per point, using their saved endpoint fields and RNG states.

The stronger-drop target is a mean fractional decrease of at least 20%, a nominal one-sided 95% lower bound above 15%, and an absolute voltage-difference upper bound below −0.01. A stronger prepared-point decrease can satisfy this target while stationary continuous NDR remains unresolved. Mixed response distributions will be reported rather than hidden by their mean.

Evidence for a stationary negative-slope branch additionally requires at least two consecutive negative ordered-current intervals with simultaneous uncertainty bounds, late-window drift-equivalence intervals inside ±0.0025, and persistence under longer dwell and conditional current refinement. Frames are correlated time samples; independent seeds are the replicates. This native-grid study does not establish spatial convergence or a calibrated thermal-noise model.

The declared ceiling is 1036 equivalent 5000τ point-segments, with at most 160 more for the explicitly conditional two-current refinement. Wall time will be measured at each stage because changed parameters can change integration cost. Gamma, heat capacity and contact geometry are not automatically added to this search.

## Reproduction records

[Environment and source hashes](provenance/environment.json), the exact [Julia project](provenance/Project.toml), and its [dependency lockfile](provenance/Manifest.toml) preserve the Linux environment. The screen and independent holdout use the physics and Julia harnesses at `3adb0005af07d9485fe56c041958b2d75c76f50d`. The replacement ordered history uses the recorded iteration guard with the same PDE/RHS source identities. Python analysis tools are separately versioned and hashed in their output receipts. The holdout plan was pushed in `0b25fd6` before its 2026-11-05 01:17 UTC launch.

The existing Linux raw-data roots are:

```text
/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_enhance_screen_20261105_v1
/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_enhance_holdout_20261105_v1
/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_enhance_history_20261105_v2
```

The screen was run with `bench/reproduce_ndr.jl --segments 2 --horizon 5000 --skip 0.6 --backend cpu_compatible`, and the holdout with the same arguments except `--segments 10`. Both use 24 persistent Julia workers, one BLAS thread, and the declared immutable job file. The runner refuses to overwrite an existing segment. Original absolute Linux paths are retained in the receipts; relocation requires an explicit new job/protocol snapshot with updated hashes.

Analysis commands are `bench/analyze_ndr_enhancement.py` for the screen, `bench/analyze_ndr_holdout.py` for the frozen 50000τ contrast, and `bench/analyze_ndr_enhancement_history.py` for the ordered histories. Each accepts `--protocol`, `--campaign-root`, `--out`, and `--validate-artifacts`. The history analysis uses only 18000–20000τ of each 20000τ dwell for its final mean; its voltage-integral check uses the same window. Independent seed trajectories, not saved frames, determine uncertainty.

The [checkpoint timing observation](provenance/checkpoint_receipt_timing.json) identifies a separate parent-process receipt cost. A later receipt-cache change is validated separately for future campaigns; it is not used by these frozen physical campaigns. Fourteen pure hash-guard tests passed on Windows and Linux. The [noisy differential test](results/receipt_cache_validation/differential_summary.json) covers fresh starts, continuation, imported checkpoints, and zero-segment imports: six point-segment pairs have identical saved values, initial-vector bytes and RNG states. An additional [array-byte check](results/receipt_cache_validation/array_byte_check.json) confirms exact bytes for all 54 saved-dataset pairs. Each run retains and validates its own whole-file HDF5 hashes.

The [isolated receipt-component benchmark](results/receipt_cache_validation/receipt_metadata_component_20261105_v1.json), under concurrent simulation load, compares 56 predecessor files containing 8.81 GB. Three trials gave an old median of 38.25 s and checked-cache median of 0.0000233 s. This measures the removed serial rereading/hash work only; it is not a whole-campaign speedup measurement.
