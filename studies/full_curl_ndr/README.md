# Corrected full-curl NDR study

## Current result

At reduced bath temperature `T0 = 0.18`, corrected full curl produces a sharp voltage decrease from `2.9350` to `2.5146` between `Je = 0.20918` and `0.21020` after 50000τ. All eight seed pairs decrease. The ordered-current controls show positive slopes and distinct history-dependent responses, so stable continuous negative differential resistance (NDR) remains unresolved.

The primary endpoint comparison uses independent hot starts at

- A: `Je = 0.20816326530612245`
- B: `Je = 0.21122448979591835`

with eight fresh seeds (`2026110411`–`2026110418`) at each endpoint. Each chain runs ten 5000τ segments to 50000τ, carries its saved field and serialized RNG checkpoint between segments, and uses the final 48000–50000τ window (2001 samples). These are independent prepared points, not a descending-current history.

The paired endpoint contrast is:

- mean `V_B − V_A = −0.2097174726`
- one-sided 95% upper bound `−0.0754827511`
- predeclared threshold `−0.01`

The nominal Student-based effect-size component passes. Endpoint A passes the declared stationarity check (drift CI `[-0.0013109, 0.0004798]`); endpoint B fails it (CI `[-0.0312416, 0.0131128]`). Late trajectories show markedly different responses and changes between observation windows. Not every seed has the same late drop. With eight samples and a mixed distribution, the Student bound has nominal coverage; it does not establish a stationary population mean. The joint stationary-NDR acceptance requirement is **not met**.

The endpoint result is not a claim of a resolved steady NDR interval, dissipationless transport, or a unique mechanism. The finite current resolution, stochastic branch behavior, and preparation dependence remain material.

## Completed four-current result

All four currents have eight complete field/RNG chains to 50000τ. Output and initial-state hashes, RNG continuity, retained times and HDF5 dimensions were checked for all 32 chains. The final 48000–50000τ means are:

| Je | Mean V | Range across eight seeds |
|---:|---:|---:|
| 0.2081632653 | 2.917436 | 2.898863–2.927321 |
| 0.2091836735 | 2.935010 | 2.922038–2.963014 |
| 0.2102040816 | 2.514596 | 2.506030–2.539120 |
| 0.2112244898 | 2.707719 | 2.544082–2.983888 |

The middle adjacent step drops by `0.420414` (about `14.3%`); every one of the eight seed pairs decreases. Its nominal one-sided Student/Bonferroni upper bound is `−0.409068`. The adjacent changes have signs **positive, negative, positive**. The decrease is therefore concentrated in one sampled current interval, rather than two consecutive resolved negative-slope intervals. The declared interval gate fails, and the highest-current stationarity test also fails.

These observations support a **finite-time, preparation-dependent voltage-drop signature**. The ordered histories below support history-dependent state selection as a contributor. They do not establish switching as the sole cause or stationary NDR on one continuous branch. A narrower unresolved interval, slower changes, or another accessible branch are not excluded.

See the [four-current figure](results/interval/ndr_interval.png), [PDF](results/interval/ndr_interval.pdf), [verified analysis](results/interval/analysis.json), and [per-seed data](results/interval/interval_points.csv).

## Complete full-range I–V

The [full-range campaign](full_iv/README.md) is complete: 99 base currents over `Je = 0–0.3` plus two refinements, all at 50000τ with fixed seed `2026110411`. All 1010 field/RNG segments and saved numeric datasets passed artifact validation. The [English figure](full_iv/results/full_iv.png), [PDF](full_iv/results/full_iv.pdf), and [CSV](full_iv/results/full_iv_seed_rows.csv) use the final 48000–50000τ window. The local eight-seed overlay includes the global seed; it does not add a ninth sample or establish eight-seed full-range coverage. These independent hot preparations provide the full finite-time I–V context and do not alter the unresolved stationary-NDR conclusion above.

## Model and protocol

All study states use the corrected full mixed magnetic curl in Float64 with Tsit5, the CPU-compatible backend, tolerance `1e-4`, original in-RHS Gaussian noise `0.001`, `T0 = 0.18`, `Nx × Ny = 60 × 30`, and `hx = hy = 0.5`. The archived coefficients include `gamma = 40` and `kappa = 2` in the study configuration. The corrected damping term is the full-curl form, including the mixed contribution `+κ²(Dyy Ax − Dxy Ay)/σ`.

The primary chains begin from independent hot starts (`psi = 0`, `T = 1.2`) at their own current. They are not continuation along a descending current sweep. The current protocol saves all fields and restores the RNG checkpoint at every 5000τ segment. The interval protocol predeclares the 48000–50000τ window, stationarity bound `0.0025`, and one-sided slope/drop comparisons. Frames are time samples, not independent replicates.

The historical Figure 3 script uses a component-diagonal Laplace approximation without the mixed curl. Its data are not treated as corrected-full-curl evidence here.

## Controls and observable checks

The normal-state control is positive Ohmic behavior: `R_eff = 27.8240`, maximum residual `4.97e−14`, and exactly zero voltage at zero current. The noise-off tolerance control starts from identical 5000τ fields and compares tolerances `1e−4`, `1e−5`, and `1e−6` at six cases. Maximum voltage shifts are `0.000645894` (`1e−4` vs `1e−6`) and `0.000103333` (`1e−5` vs `1e−6`). This is a deterministic tolerance control, not stochastic convergence evidence.

The voltage-observable pilot compares sampled window means with `−Lx Δ⟨Ax⟩/Δt` over the same stored window; differences are below `0.00041` in native model units. In the final endpoint cohort the maximum discrepancy is `0.000359`. Continuation, history, and analysis-chain smoke receipts verify the serialization and analysis paths.

## Ordered-history control and reference replay

Six ordered-history jobs are complete: three seeds (`2026110411`–`2026110413`), ascending and descending directions, four currents, and 5000τ dwell after the 50000τ endpoint state/RNG checkpoint. The voltage sequences (native units) are:

| seed | ascending, as `Je` increases | descending, as `Je` decreases |
|---|---|---|
| 411 | 2.91945, 2.96184, 3.00288, 3.04496 | 2.89146, 2.50688, 2.47311, 2.43464 |
| 412 | 2.92002, 2.96136, 3.00287, 3.04399 | 2.55597, 2.51911, 2.47803, 2.43917 |
| 413 | 2.92678, 2.96827, 3.01098, 3.05126 | 2.81831, 2.53660, 2.50166, 2.46426 |

Every adjacent change has positive `dV/dJe` along these sampled ordered histories. The large differences between preparations show history-dependent separation over the tested 5000τ dwell. This dwell is shorter than the 50000τ independent preparation, so the control does not exclude slower changes or all other branches. Together with endpoint B bimodality and late changes, it supports a history-sensitive ensemble contrast. See the [verified history receipt](results/history_analysis.json), [history figure](results/history/history.png), and [finite-difference slopes](results/history/history_slopes.json).

Selected final [field maps](results/fields/fields.png) compare condensate density, temperature and magnetic field for the two responses. These are illustrative snapshots; the [provenance receipt](results/fields/provenance.json) records the selected states and their hashes.

The original reference replay of the final 45000–50000τ window from identical field and RNG checkpoints matches the compatible path at both critical endpoints: all nine datasets have exact zero error, solver counts are equal, and RNG checkpoints are equal. This validates the replay/comparison path, not the physical stability of the branch. See [`reference_replay.json`](results/reference_replay.json).

## Reproduction

Use Julia 1.12.1 with the isolated depot described in [the performance documentation](../../docs/performance/README.md). From the repository root, launch with 16 Julia workers:

```bash
JULIA_BIN=/path/to/julia-1.12.1/bin/julia
export JULIA_DEPOT_PATH=/path/to/isolated_depot:/path/to/existing_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 GKSwstype=100
NOFASTSCAN_SKIP_MAIN=1 NOFASTSCAN_SKIP_WORKERS=1 \
  "$JULIA_BIN" -p 16 --project=. --startup-file=no bench/reproduce_ndr.jl \
  --jobs studies/full_curl_ndr/endpoint_jobs.json \
  --out /caller/path/full_curl_ndr \
  --segments 10 --horizon 5000 --skip 0.6 \
  --backend cpu_compatible
```

The job JSON paths are repository-relative when launched from the repository root. Results are in native model units, not experimental volts or seconds. Primary analysis files are [`analysis.json`](results/endpoint/analysis.json), [`endpoint_pairs.csv`](results/endpoint/endpoint_pairs.csv), [`ndr_summary.png`](results/endpoint/ndr_summary.png), and [`ndr_summary.pdf`](results/endpoint/ndr_summary.pdf). Protocols are [`protocol.json`](protocol.json) and [`interval_protocol.json`](interval_protocol.json).

## Scope

The declared four-current study, ordered histories and reference replay are complete. The evidence supports a finite-time voltage decrease with strong preparation/history dependence; stable continuous NDR has not been established in this case. Only `T0 = 0.18` was simulated in this study.

These are native-grid results, without a spatial-convergence claim. Saved-Ax agreement checks consistency of the implemented temporal-gauge voltage observable; it does not independently calibrate physical voltage or imposed current. The adaptive-RHS random draws define the specified numerical forcing protocol. The deterministic tolerance control does not establish stochastic convergence or a fluctuation-dissipation calibrated noise model, and no experimental-unit conversion is implied.
