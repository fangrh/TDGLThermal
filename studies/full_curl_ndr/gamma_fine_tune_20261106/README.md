# Small gamma changes in corrected full-curl TDGL

The user requested small physical changes that could make the NDR-like decrease more pronounced. This study tests `gamma = 38, 40, 42`, only ±5% about the original value. It retains the complete mixed-curl operator, Float64, original stochastic forcing, thermal coefficients, geometry, tolerance and measurement rules. It seeks a stronger reproducible **prepared-point voltage decrease**; a continuous stationary negative branch is a separate question.

The [previous thermal search](../enhancement_20261105/README.md) failed its enhancement target. The [local step-response control](../state_response_20261105/README.md) found positive local responses around both original prepared states, leaving a central current gap unresolved. Their outcomes are retained. The original 14.3237% eight-seed decrease is historical context, rather than the only control for a new parameter choice.

## Physical motivation

In the implemented KWT equation, gamma changes the relative radial and tangential condensate dynamics and also enters GL heat production. At a fixed state, it does not change the full-curl magnetic stencil. This makes small gamma changes a focused test of the coupled dynamics and state selection. It does not guarantee enhancement or isolate a microscopic mechanism. The original generalized TDGL work includes phase-slip solutions: [Kramer and Watts-Tobin, Physical Review Letters 40, 1041 (1978)](https://doi.org/10.1103/PhysRevLett.40.1041).

## Frozen screen

The immutable [protocol](plan/screen_protocol.json), [jobs](plan/screen_jobs.json) and [three configurations](plan/configs/) define 24 fresh configured-hot chains: three gamma values × four original currents × two new seeds (`2026110601`, `2026110602`). Every chain reaches 50000τ in ten 5000τ segments, starting at `psi = 0`, `T = 1.2`. This equal horizon is necessary because the earlier 10000τ screen did not reliably recover the original baseline feature.

The four currents are `0.20816326530612245`, `0.20918367346938774`, `0.21020408163265306` and `0.21122448979591835`. The **only primary pair** is the original middle interval, `0.20918367346938774 → 0.21020408163265306`. Final means use 48000–50000τ; drift compares that window with 43000–45000τ. Frames are correlated samples; seeds are the replicates. All three neighboring pairs are reported, but a new bracket cannot replace this primary pair within the declared study.

For each seed define `D(gamma) = (V_low − V_high) / V_low` and paired improvement `D(gamma) − D(40)`. A screen candidate requires positive drops larger than 0.01 native voltage units, low-current voltage at least 0.05, and positive fractional improvement in **both** seeds. Both primary endpoints of candidate and contemporaneous baseline must pass nominal two-sided 95% drift intervals inside ±0.0025. Candidates are ranked by the smaller paired improvement, with gamma 38 breaking exact ties. The result may be that neither candidate advances; two seeds provide only a limited screen.

## Conditional confirmation and ordered control

If a candidate advances, its gamma is frozen before examining eight fresh seed labels `2026110611`–`2026110618`. Candidate and matched gamma-40 baseline both run the primary endpoints and their midpoint to 50000τ: 48 chains / 480 segments. Matching seed labels defines statistical blocks; adaptive RHS forcing does not produce identical physical noise histories across gamma values.

Confirmation requires all eight candidate drops positive, mean fractional decrease at least 20%, nominal one-sided 95% fraction lower bound above 15%, a positive one-sided 95% lower bound on paired improvement over the new baseline, voltage-difference upper bound below −0.01, and primary-endpoint drift gates passed for both configurations. Every outcome and uncertainty is reported even if these rules fail.

Successful confirmation additionally triggers the predeclared ordered-history control: first four holdout seeds, both directions, five equally spaced currents spanning the fixed primary interval, 20000τ per dwell, starting from candidate endpoint field/RNG states. This costs 160 equivalent 5000τ segments. Positive prepared-point evidence alone does not establish a stationary branch; ordered jumps and resolved negative intervals have different interpretations.

The ceiling is **240 screen + 480 conditional confirmation + 160 conditional ordered = 880 equivalent 5000τ scientific segments**. No automatic parameter widening, new bracket selection or extra seed campaign is added if a stage fails.

## Numerical and provenance boundaries

The PDE/model source SHA256 remains `4464df90825a7afefdddbe90ae3b314724e6da951d1bc7da1ee2cf845eb2e422`, and the compatible RHS source is `900b2a7ab277030b96454fd4563f926b0be983cc349f64d5fa07a0381ef287fe`. Only the gamma line differs between YAML configurations. The progress guard uses the recorded source and explicit iteration limit 10000000. Raw fields, packed initial states, hashes and serialized RNG checkpoints are retained.

Fresh initial-vector bytes and initial RNG states are checked across the three gamma values for each current/seed. Accepted/rejected steps and RHS counts are reported because gamma changes can change the adaptive evaluation and forcing schedule. This is a controlled comparison within the legacy solver/noise protocol, not a calibrated thermal SDE or a material-parameter determination. Spatial convergence remains outside this study.
