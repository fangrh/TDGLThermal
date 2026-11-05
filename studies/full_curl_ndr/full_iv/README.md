# Full-range corrected-full-curl I–V

All **101/101 points are complete and artifact-validated at 50000τ**. The curve contains 99 uniformly spaced base currents from `Je = 0` to `0.3`, plus the two previously tested refinement currents inside the voltage-drop interval. All points use seed `2026110411`. The four-current, eight-seed study remains a local validation cohort; it includes this global seed and is not an additional ninth realization or eight-seed coverage of the full curve.

## Results

- [English four-panel figure](results/full_iv.png) and [vector PDF](results/full_iv.pdf): full I–V, voltage-drop zoom, temperature and condensate density.
- [Per-point CSV](results/full_iv_seed_rows.csv), [voltage summary CSV](results/full_iv_summary.csv), and [verified analysis JSON](results/full_iv_analysis.json).
- [Complete chain registry](results/chain_registry.json), [campaign receipt](results/campaign.json), and [source-checkout verification](results/source_checkout_verification.json).

The fixed-seed curve drops from `V = 2.93302994` to `2.50833359` between `Je = 0.20918367` and `0.21020408`: a decrease of `0.42469635`, approximately `14.48%`. These are final-window means from independently prepared currents. The local eight-seed mean decrease is approximately `14.3%`, as reported in the [parent study](../README.md).

| Final 48000–50000τ observable | Minimum | Maximum |
|---|---:|---:|
| V | −0.000004082 | 6.84990993 |
| T_avg | 0.18107509 | 0.45649826 |
| mean \|ψ\|² | 0.24037249 | 0.82310464 |

The large late-window drift occurs at `Je = 0.21122449`: its 48000–50000τ mean is `0.07460780` below the 43000–45000τ mean. The other 100 fixed-seed points have absolute drift below `0.0025`; this descriptive comparison is not an ensemble stationarity test. The maximum difference between sampled mean voltage and the saved-`Ax` endpoint estimator is `0.00101435` in native units. The completed curve retains the finite-time voltage-drop signature and does not establish stationary continuous NDR.

## Protocol

Every current is an independent hot preparation (`psi = 0`, `T = 1.2`) with bath temperature `T0 = 0.18`, corrected full mixed magnetic curl, `60 × 30` grid, `hx = hy = 0.5`, Float64, adaptive Tsit5, tolerance `1e-4`, and the original in-RHS noise strength `0.001`. Each trajectory reaches 50000τ in ten 5000τ segments. Saved fields and serialized RNG states are carried between segments. The reported averages use 48000–50000τ, with 2001 retained samples per point. Units are native model units.

The archived [catalog](input/catalog.json) and [job plan](input/jobs.json) record the measured server paths and source hashes. Four previously validated 50000τ chains are reused in full, and two 5000τ shoulder checkpoints are continued. The remaining 95 points start fresh. This requires 968 new 5000τ point-segments. Imported and new receipt IDs are mapped by declared current and seed, rather than by folder-name conventions.

## Verification and interpretation

The [mixed-continuation smoke receipt](results/mixed_smoke.json) checks jobs with zero, one and two new segments in the same campaign. All nine saved datasets and the RNG checkpoints agree exactly with the established continuation path; the zero-segment checkpoint remains unchanged.

The full-curve analyzer checked all 1010 physical segments: 968 newly computed and 42 imported. Grid coverage, seed, cumulative intervals, model/configuration/RHS hashes, solver settings, retained schedules, output hashes, initial fields and RNG continuity passed. All nine saved numeric datasets were scanned for nonfinite values. The JSON records the analysis-code hashes and software versions as well as physical-source provenance. Fourteen focused software checks and the Linux mixed-continuation smoke passed. The clean Linux branch and measured runtime match byte for byte in seven critical source/configuration files.

The analyzer reports late-window drift and the saved-`Ax` integrated-voltage comparison. Incomplete curves are explicitly labeled and have gaps. Condensate density is the saved `psi_avg = mean(|psi|²)` observable.

A complete finite-time I–V does not establish a stationary continuous NDR branch. The existing local study shows a preparation-dependent voltage decrease and history-dependent responses, with unresolved stationarity at one endpoint. See the [parent study](../README.md) for the eight-seed statistics and controls.

## Reproduction

From the Linux repository root, prepare a new plan. A source-run directory containing the compatible earlier campaigns enables checkpoint reuse; an empty directory produces fresh ten-segment jobs at all 101 currents.

```bash
python3 bench/prepare_full_iv.py \
  --source-run-root /path/to/earlier_runs \
  --out /path/to/new_full_iv/input

export JULIA_DEPOT_PATH=/path/to/isolated_depot:/path/to/existing_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 GKSwstype=100
NOFASTSCAN_SKIP_MAIN=1 NOFASTSCAN_SKIP_WORKERS=1 \
  /path/to/julia-1.12.1/bin/julia -p 24 --project=. --startup-file=no \
  bench/reproduce_ndr.jl \
  --jobs /path/to/new_full_iv/input/jobs.json \
  --out /path/to/new_full_iv/campaign \
  --segments 10 --horizon 5000 --skip 0.6 --backend cpu_compatible

python3 bench/analyze_full_iv.py \
  --catalog /path/to/new_full_iv/input/catalog.json \
  --campaign-root /path/to/new_full_iv/campaign \
  --focused-analysis /path/to/ndr_interval_analysis_v1/analysis.json \
  --out /path/to/new_full_iv/analysis --validate-artifacts
```

The focused overlay is optional. `--validate-artifacts` hashes and checks the raw outputs; run it after the campaign has finished. The analyzer exits with status 1 while coverage is incomplete or validation fails. It writes an English four-panel figure, CSV data, and a JSON receipt without promoting a finite-time voltage drop to stationary NDR.

## Runtime context

The campaign used the validated CPU-compatible backend with 24 workers. Its observed wall time was approximately **106.82 minutes**, including startup, inferred from the run-log creation time and final campaign timestamp. This is not a new matched speedup measurement. Approximately 159 GB of raw HDF5 remain on the server; the registry records their paths and hashes.

The earlier matched production-noise benchmark measured 1.4644× speedup over the reference at the same worker count, including HDF5 output. The separate 26.80× deterministic GPU result compares against one CPU core and does not validate this legacy-noise production protocol. See the [performance receipts](../../../docs/performance/README.md) and [noise-acceleration investigation](../../../docs/performance/noise_acceleration.md).
