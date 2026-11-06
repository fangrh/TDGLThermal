# Full-curl I–V: full range and local decrease

The [English figure](ndr_full_and_zoom.png) ([PDF](ndr_full_and_zoom.pdf), [SVG](ndr_full_and_zoom.svg)) uses archived, verified simulation data. Panel A shows 101 independent prepared currents over `Je = 0–0.3`, one seed `2026110411`. Panel B shows four local currents, eight original seeds, their individual values and mean ± nominal two-sided Student 95% seed intervals (`df = 7`). The full-range seed belongs to the local cohort and does not add a ninth replicate.

Between `Je = 0.2091836735` and `0.2102040816`, the mean voltage changes from `2.935010` to `2.514596`. All eight seed pairs decrease; their mean paired fractional decrease is `14.3237%`. Lines guide comparisons between independently prepared points. This finite-time signature does not establish a continuous stationary negative branch; the ordered-history and local-response controls retain their separately reported limitations.

The original corrected full-curl equations, `T0 = 0.18`, 50000τ preparations and final 48000–50000τ window are retained. No simulation, smoothing or point exclusion was added for this figure. [Zoom source values](ndr_zoom_source.csv) and [input/code hashes and software versions](figure_provenance.json) accompany the export. Full-range values remain in [the existing CSV](../full_iv/results/full_iv_seed_rows.csv).

Reproduce from the repository root with `python bench/plot_ndr_showcase.py --out studies/full_curl_ndr/figures_20261106`.
