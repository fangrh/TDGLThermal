# Full-curl NDR enhancement exploration

This is a bounded physical-parameter exploration requested on 2026-11-05. The delivered [101-point baseline](../full_iv/README.md) remains the comparison dataset. The complete screen is verified; independent long-horizon confirmation is pending.

## Verified screening result and frozen holdout

All 126 chains and 252 segments completed successfully. Output/initial-state/RNG hashes, chain continuity, retained schedules, and all nine saved numeric datasets passed validation with no errors. See [the screen analysis](results/screen/enhancement_analysis.json) and [all seven I–V/temperature/condensate curves](results/screen/enhancement_screen.png).

Only `T0 = 0.22` qualified at the declared 10000τ screen horizon: `Je = 0.19 → 0.20` decreased voltage by 2.89% and 1.19% in the two seeds. No other configuration, including the baseline at this same horizon, had an eligible decrease in both seeds. This is a small exploratory decrease, not evidence that the original 50000τ baseline drop has been improved.

The [frozen holdout protocol](holdout_plan/holdout_protocol.json) fixes `T0 = 0.22`, currents `0.1875, 0.19, 0.1925, 0.195, 0.1975, 0.20, 0.2025`, and eight fresh seeds. The primary contrast is exclusively `0.19 → 0.20`; choosing a different interval from the holdout results is not allowed. Each of the 56 chains will reach 50000τ. No new-seed simulation was launched before freezing these files.

For comparison, the original four-current/eight-seed 50000τ study has a 14.32% mean paired middle-interval drop. Its [English detail figure](results/baseline_detail/ndr_enhancement_comparison.png) retains every seed and the higher-current spread; this prepared-point response does not establish a stationary continuous negative branch.

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

[Environment and source hashes](provenance/environment.json), the exact [Julia project](provenance/Project.toml), and its [dependency lockfile](provenance/Manifest.toml) preserve the Linux environment. Physics and Julia harnesses remain at `3adb0005af07d9485fe56c041958b2d75c76f50d`; Python analysis tools are separately versioned and hashed in their output receipts. The holdout plan was pushed in `0b25fd6` before its 2026-11-05 01:17 UTC launch.

The existing Linux raw-data roots are:

```text
/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_enhance_screen_20261105_v1
/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_enhance_holdout_20261105_v1
```

The screen was run with `bench/reproduce_ndr.jl --segments 2 --horizon 5000 --skip 0.6 --backend cpu_compatible`, and the holdout with the same arguments except `--segments 10`. Both use 24 persistent Julia workers, one BLAS thread, and the declared immutable job file. The runner refuses to overwrite an existing segment. Original absolute Linux paths are retained in the receipts; relocation requires an explicit new job/protocol snapshot with updated hashes.

Analysis commands are `bench/analyze_ndr_enhancement.py` for the screen, `bench/analyze_ndr_holdout.py` for the frozen 50000τ contrast, and `bench/analyze_ndr_enhancement_history.py` for the ordered histories. Each accepts `--protocol`, `--campaign-root`, `--out`, and `--validate-artifacts`. The history analysis uses only 18000–20000τ of each 20000τ dwell for its final mean; its voltage-integral check uses the same window. Independent seed trajectories, not saved frames, determine uncertainty.
