# Small gamma changes in corrected full-curl TDGL

The user requested small physical changes that could make the NDR-like decrease more pronounced. This study tests `gamma = 38, 40, 42`, only ±5% about the original value. It retains the complete mixed-curl operator, Float64, original stochastic forcing, thermal coefficients, geometry, tolerance and measurement rules. It seeks a stronger reproducible **prepared-point voltage decrease**; a continuous stationary negative branch is a separate question.

The [previous thermal search](../enhancement_20261105/README.md) failed its enhancement target. The [local step-response control](../state_response_20261105/README.md) found positive local responses around both original prepared states, leaving a central current gap unresolved. Their outcomes are retained. The original 14.3237% eight-seed decrease is historical context, rather than the only control for a new parameter choice.

## Final result: enhancement target not met

All **720 declared unconditional segments** completed: 240 in the two-seed screen and 480 in the prospectively amended fresh eight-seed validation. Both gamma perturbations failed the fixed primary comparison. No candidate was selected, so the conditional 160-segment ordered control was not triggered. No extra bracket, seed subset, statistical window or parameter search was substituted after observing the outcomes.

The [fresh English figure](results/validation_v2/gamma_validation.png) ([PDF](results/validation_v2/gamma_validation.pdf)), [complete analysis](results/validation_v2/gamma_validation_analysis.json), [48 seed/current rows](results/validation_v2/gamma_validation_rows.csv) and [480 segment rows](results/validation_v2/gamma_validation_segments.csv) retain all observations. Thin lines/circles are seed results; heavier marks are means, with the bounds described on each panel. Lines joining independently prepared endpoints do not represent a continuously followed current sweep.

| Gamma | Mean V at low J | Mean V at high J | Mean seedwise decrease D | Seeds with D > 0 | Mean paired improvement over gamma 40 |
|---|---:|---:|---:|---:|---:|
| 40 | 2.821318 | 2.522348 | +10.074% | 6/8 | Reference |
| 38 | 2.455057 | 2.869761 | -16.896% | 0/8 | -26.969 percentage points |
| 42 | 2.986946 | 2.997392 | -0.350% | 3/8 | -10.424 percentage points |

Negative D means voltage increases. The V columns are arithmetic means, whereas D is the arithmetic mean of seedwise fractions, **not the fraction calculated from the two mean voltages**. The primary pair remains `0.20918367346938774 -> 0.21020408163265306` for every row.

The per-comparison one-sided alpha=0.025 lower bounds for mean D are 3.533%, -22.309% and -0.858% for gamma 40, 38 and 42. Candidate paired-improvement bounds are [-34.380, -19.558] and [-16.597, -4.251] percentage points. Here each endpoint is an alpha=0.025 one-sided t bound; the superiority rule uses the lower endpoint, with Bonferroni correction for two candidate claims. These constituent intervals are not jointly simultaneous 95% intervals. Neither candidate approaches the declared 20% mean-drop target or a positive improvement over the contemporaneous baseline.

The fresh gamma-40 control is also less reproducible than the historical eight-seed set. Seeds `2026110611` and `2026110614` have low-current voltages 2.473005 and 2.470805, compared with 2.922065-2.953483 for the other six seeds; their voltage changes have the opposite sign. They are retained, without exclusion. Their low-current mean temperature is about 0.2635 and mean |psi|^2 about 0.5076, versus about 0.2733-0.2744 and 0.4631-0.4662 for the other six. These are distinct late-time prepared outcomes in this solver/noise protocol, rather than evidence identifying their microscopic cause. The previous 14.3237% result must not be generalized to all fresh seeds or advertised as a stable continuous NDR branch.

| Gamma | Endpoint | Late-window drift mean | Nominal two-sided 95% CI | Inside +/-0.0025 |
|---|---|---:|---|---|
| 40 | Low J | -0.000131 | [-0.000647, +0.000385] | Yes |
| 40 | High J | -0.000342 | [-0.002577, +0.001893] | No |
| 38 | Low J | +0.004299 | [-0.004679, +0.013276] | No |
| 38 | High J | -0.001554 | [-0.012807, +0.009698] | No |
| 42 | Low J | +0.000427 | [+0.000015, +0.000838] | Yes |
| 42 | High J | -0.000064 | [-0.000924, +0.000796] | Yes |

Gamma 42 passes its own endpoint drift checks but has no mean voltage decrease at the fixed primary pair. Gamma 40's high-current drift interval narrowly exceeds the frozen margin. The failure is therefore not solely a drift-gate issue: both perturbations also have worse primary drops and fail the amplitude/sign/improvement requirements. A stationary negative branch remains unconfirmed.

The [final audit](results/validation_v2/final_validation_receipt.json) confirms 48 jobs, 480 aggregate and individual receipts, 480 matching analysis receipt hashes, unchanged numerical and analysis sources, complete finite-field/initial-state/RNG/schedule checks, and the end of the owned simulation process session. The [local archive check](results/validation_v2/local_archive_verified.json) confirms downloaded analysis, campaign and launch bytes match that receipt, and the local frozen plan and all analysis helpers match Linux provenance. Scoped Git attributes retain LF endings for the frozen plans and text receipts across checkouts. Raw fields and RNG checkpoints remain on the server under the run roots below. The simulation finished at approximately 16:55 UTC on 2026-11-06; final validation and audit completed at 16:59 UTC (18:55 and 18:59 Europe/Kyiv).

## Verified two-seed screen

All 24 fresh chains / 240 segments reached 50000τ and passed source/configuration, field/initial-vector/RNG hash and continuity, retained-schedule and nine-dataset finite checks. The [final audit](results/screen/final_validation_receipt.json) matches all individual receipts to the campaign, with the owned screen process session ended. The [English figure](results/screen/gamma_screen.png) ([PDF](results/screen/gamma_screen.pdf)), [analysis](results/screen/gamma_screen_analysis.json) and [seed rows](results/screen/gamma_screen_rows.csv) report all variants and currents.

| Gamma | Mean paired decrease at the fixed primary pair | V1 candidate |
|---|---:|---|
| 40 | +13.665% | Reference |
| 38 | −19.671% (voltage increase) | No |
| 42 | −0.690% (voltage increase) | No |

Neither perturbation strengthens the original primary decrease in these two seeds. The v1 strict drift/selection gates also fail. Gamma 42 has an exploratory decrease at the **next** interval, `0.2102040816 → 0.2112244898`, with seedwise fractions 14.481% and 14.731% (mean 14.606%). This indicates a shifted feature at screening scale, rather than a confirmed amplitude enhancement. It cannot replace the primary endpoint comparison.

The amendment and fresh allocation were frozen before these outcomes were viewed. All three gamma values subsequently completed the fixed eight-seed validation reported above. The raw-data roots are `/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_gamma_screen_20261106_v1` and `/data/User/fangr1/projects/TDGLThermal-linux-performance-runs/ndr_gamma_validation_20261106_v2`.

In the exploratory shifted gamma-42 interval, the two-seed mean temperature decreases from 0.275922 to 0.267362 while mean |psi|^2 increases from 0.464784 to 0.503177. This is a descriptive correlation with the voltage decrease, not a causal explanation or fresh eight-seed confirmation of that neighboring interval.

## Outcome-blind amendment v2

The [prospective amendment](amendment_v2/amendment.json) changes the fresh validation allocation before any current-screen voltage/temperature/condensate outcomes or drift estimates were inspected. Its saved server context records 144/240 segment receipts and no final campaign or analysis file. The original v1 screen protocol and its strict selection verdict are preserved and reported separately.

The reason comes solely from historical eight-seed drift variation. Its estimated SDs are 0.001073 and 0.001718 at the two primary currents. Using these for precision planning gives approximate N=2 nominal 95% half-widths 0.00964 and 0.01543, much larger than the ±0.0025 equivalence margin. These are planning calculations, not results from the running screen or proof of stationarity. Requiring the two-seed gate before any fresh validation would provide a poor screening bottleneck.

Under v2, **all three fixed gamma values** run both fixed primary endpoints with eight fresh seeds, regardless of the v1 screening verdict: `3 × 2 × 8 = 48` fresh chains / 480 segments. This replaces the original candidate-plus-baseline, three-current allocation within the same budget. No parameter, current bracket, statistical window or seed subset is chosen from current-screen outcomes. Missing or failed runs disable complete inference; no exclusion or imputation is permitted, and there is no outcome-based early stopping.

The estimand is the arithmetic mean of seedwise fractional drops, with seedwise improvement relative to the same-label gamma-40 control. The two candidate-versus-baseline claims use Bonferroni correction, alpha 0.025 per candidate for one-sided fraction, improvement and voltage-difference bounds. A successful candidate must satisfy every original amplitude/sign requirement and the original nominal two-sided 95% endpoint drift-equivalence gates. These constituent intervals are not advertised as jointly simultaneous. The drift gate remains conservative, and eight seeds may still lack adequate precision.

Among candidates meeting every criterion, the winner is the larger paired-improvement lower bound, then the larger mean fractional drop, then gamma 38. Successful confirmation triggers the original ordered control. Two independent prepared endpoints confirm a drop and a parameter comparison, rather than curve shape or a continuous stationary negative branch. The total ceiling remains **240 + 480 + 160 = 880** equivalent segments. Any successful claim is explicitly under amended v2, rather than a retrospective v1 success.

The [server freeze acknowledgement](amendment_v2/freeze_ack.json) records that amendment commit `d1bff3b` was already pushed while only 144 screen receipts existed and final outcomes were unavailable. The immutable [fresh validation protocol](amendment_v2/plan/validation_protocol.json), [48 jobs](amendment_v2/plan/validation_jobs.json) and [configurations](amendment_v2/plan/configs/) use source/config identities checked against v1. The preparer does not read screen analysis or voltage outcomes.

## Physical motivation

In the implemented KWT equation, gamma changes the relative radial and tangential condensate dynamics and also enters GL heat production. At a fixed state, it does not change the full-curl magnetic stencil. This makes small gamma changes a focused test of the coupled dynamics and state selection. It does not guarantee enhancement or isolate a microscopic mechanism. The original generalized TDGL work includes phase-slip solutions: [Kramer and Watts-Tobin, Physical Review Letters 40, 1041 (1978)](https://doi.org/10.1103/PhysRevLett.40.1041).

## Frozen screen

The immutable [protocol](plan/screen_protocol.json), [jobs](plan/screen_jobs.json) and [three configurations](plan/configs/) define 24 fresh configured-hot chains: three gamma values × four original currents × two new seeds (`2026110601`, `2026110602`). Every chain reaches 50000τ in ten 5000τ segments, starting at `psi = 0`, `T = 1.2`. This equal horizon is necessary because the earlier 10000τ screen did not reliably recover the original baseline feature.

The four currents are `0.20816326530612245`, `0.20918367346938774`, `0.21020408163265306` and `0.21122448979591835`. The **only primary pair** is the original middle interval, `0.20918367346938774 → 0.21020408163265306`. Final means use 48000–50000τ; drift compares that window with 43000–45000τ. Frames are correlated samples; seeds are the replicates. All three neighboring pairs are reported, but a new bracket cannot replace this primary pair within the declared study.

For each seed define `D(gamma) = (V_low − V_high) / V_low` and paired improvement `D(gamma) − D(40)`. A screen candidate requires positive drops larger than 0.01 native voltage units, low-current voltage at least 0.05, and positive fractional improvement in **both** seeds. Both primary endpoints of candidate and contemporaneous baseline must pass nominal two-sided 95% drift intervals inside ±0.0025. Candidates are ranked by the smaller paired improvement, with gamma 38 breaking exact ties. The result may be that neither candidate advances; two seeds provide only a limited screen.

## Original v1 conditional plan, retained for audit

If a candidate advances, its gamma is frozen before examining eight fresh seed labels `2026110611`–`2026110618`. Candidate and matched gamma-40 baseline both run the primary endpoints and their midpoint to 50000τ: 48 chains / 480 segments. Matching seed labels defines statistical blocks; adaptive RHS forcing does not produce identical physical noise histories across gamma values.

Confirmation requires all eight candidate drops positive, mean fractional decrease at least 20%, nominal one-sided 95% fraction lower bound above 15%, a positive one-sided 95% lower bound on paired improvement over the new baseline, voltage-difference upper bound below −0.01, and primary-endpoint drift gates passed for both configurations. Every outcome and uncertainty is reported even if these rules fail.

Successful confirmation additionally triggers the predeclared ordered-history control: first four holdout seeds, both directions, five equally spaced currents spanning the fixed primary interval, 20000τ per dwell, starting from candidate endpoint field/RNG states. This costs 160 equivalent 5000τ segments. Positive prepared-point evidence alone does not establish a stationary branch; ordered jumps and resolved negative intervals have different interpretations.

The ceiling is **240 screen + 480 conditional confirmation + 160 conditional ordered = 880 equivalent 5000τ scientific segments**. No automatic parameter widening, new bracket selection or extra seed campaign is added if a stage fails.

## Numerical and provenance boundaries

The PDE/model source SHA256 remains `4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422`, and the compatible RHS source is `900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe`. Only the gamma line differs between YAML configurations. The progress guard uses the recorded source and explicit iteration limit 10000000. Raw fields, packed initial states, hashes and serialized RNG checkpoints are retained.

Fresh initial-vector bytes and initial RNG states are checked across the three gamma values for each current/seed. Accepted/rejected steps and RHS counts are reported because gamma changes can change the adaptive evaluation and forcing schedule. This is a controlled comparison within the legacy solver/noise protocol, not a calibrated thermal SDE or a material-parameter determination. Spatial convergence remains outside this study.
