# Result layout

The packaged results are limited to outputs from the main scCGRL workflow,
publication figures, and its formal repeated runs:

- `<dataset>/seed<seed>/`: one complete model run. For example,
  `human_myeloid/seed42/` contains pseudotime, metrics, model state, processed
  AnnData, resource measurements, manifest, and diagnostic figures.
- `<dataset>/seeds<first>-<last>/`: a multi-seed main-CLI run when `--runs` is
  greater than one.
- `figures/`: the human myeloid, mouse pancreas, and human bone marrow large
  publication figures in PNG, TIFF, EPS, and editable SVG formats.
- `repeat_50_runs/<dataset>/`: recorded 50-run metric CSVs for all five
  datasets (seeds 42--91).

Baseline runs are written to:

```text
results/benchmark/<method>/<dataset>/
results/benchmark/combined/
```

Each main-model run directory contains, where applicable:

```text
<dataset>_pseudotime.csv
<dataset>_single_run_metrics.csv
scCGRL_processed.h5ad
scCGRL_model_state.pkl
run_manifest.json
00_model_report.txt
diagnostic PNG files
```

Generated benchmark and table directories are not required to exist before a
run; the corresponding drivers create them.
