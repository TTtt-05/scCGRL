This directory is the portable scCGRL code and data package. Runtime paths are
repository-relative or supplied explicitly on the command line. The packaged
scope is the core scCGRL model, its five configured datasets, the ten-method
benchmark, repeated main-model runs, publication-figure programs, table
summaries, and data-reproducibility utilities.

## Quick start

From the repository root:

```powershell
conda env create -f environment.yml
conda activate sccgrl_20260827

# Recompute the complete human-myeloid workflow from integer raw counts.
python run_sccgrl.py --dataset human_myeloid --input-stage raw --seed 42 --episodes 10000

# Reuse the packaged, model-ready PCA/neighbors/UMAP cache.
python run_sccgrl.py --dataset human_myeloid --input-stage processed --seed 42 --episodes 10000
```

Both commands write the run to `results/human_myeloid/seed42/`. They save
the per-cell pseudotime, metrics, resource measurements, processed AnnData,
model state, audit manifest, and diagnostic figures.

## Repository map

| Path | Main contents | What it is used for |
|---|---|---|
| `configs/` | One audited YAML file per dataset | Defines labels, early state, preprocessing, UMAP dimensions, endpoint settings, biological references, and plotting dimensions. |
| `data/raw/` | Three real-data H5AD files with integer counts | Recomputes the complete common preprocessing pipeline from source counts. |
| `data/processed/` | Five dataset-named H5AD files plus checksums and manifests | Reuses or audits the unified processed inputs, labels, PCA, neighbors, and three-dimensional UMAP representations. |
| `src/sccgrl/` | Preprocessing, graph/endpoints, Q-learning, trajectory construction, graph pseudotime, RF propagation, metrics, and I/O modules | Implements the main scCGRL algorithm. |
| `run_sccgrl.py` | Main command-line entry point | Runs one or more seeds on one configured dataset from raw or processed input. |
| `benchmark/` | PAGA, DPT, Palantir, Slingshot, Monocle 1/2/3, SLICER, TSCAN, and SCORPIUS runners | Runs the ten comparison methods and applies the shared evaluation adapter. |
| `experiments/repeat_50_runs/` | The audited seeds 42--91 repetition driver | Reproduces the 50-run stability/resource experiment for the main method. |
| `figures/` | Dataset-specific publication-figure scripts | Produces the biologically distinct figures for each dataset without forcing one common layout. |
| `tables/` | Benchmark, repeated-run, and runtime/memory summaries | Converts per-run CSV files into manuscript-ready statistical tables. |
| `reproducibility/data/` | Raw-data preparation, exact verification, checksums, and bilingual provenance | Reconstructs and verifies the three real raw inputs without changing counts. |
| `reproducibility/validate_seed42_run.py` | Complete seed-42 validation | Compares a portable main-model run with the recorded reference output. |
| `results/repeat_50_runs/` | Existing five-dataset 50-run metric CSVs | Supplies the recorded scCGRL results imported by the benchmark. |
| `results/<dataset>/seed<seed>/` | Complete single-run output | Stores metrics, pseudotime, model state, processed AnnData, audit manifest, and diagnostic figures for one dataset and seed. |
| `results/figures/` | Three real-data publication figures | Preserves each large figure in PNG, TIFF, EPS, and editable SVG formats. |
| `docs/` | Data, benchmark, software, and reproduction documentation | Explains provenance, implementation choices, output locations, and software versions. |

## Data availability and input contents

The data files in this release are verified, repository-relative artifacts.
Checksums are stored in `data/checksums.sha256` and
`data/processed/processed_inputs_checksums.sha256`; source URLs, annotation
provenance, selection rules, and redistribution boundaries are recorded in
`DATA_LICENSES.md` and `docs/data_availability.md`.

The three real-data files in `data/raw` contain integer raw counts:

| Dataset | Raw shape | Required label | Runtime selection |
|---|---:|---|---|
| `human_myeloid` | 3,264 x 19,089 | `cluster` | none |
| `mouse_pancreas` | 2,780 x 27,998 | `clusters_fig6_broad_final` | E15.5 endocrine subset already fixed by the source-selection rule |
| `human_bone_marrow` | 7,439 x 17,226 | `celltype` | exclude ETP (60) and BcellPre (154), leaving 7,225 cells |

The corresponding real-data caches in `data/processed` contain 2,000 HVGs,
the required labels and metadata, 50-dimensional `X_pca`, 3-dimensional
`X_umap`, and the Scanpy neighbor matrices. The common real-data preprocessing
is: dataset-specific exclusion; `normalize_total(target_sum=10000)`; natural
`log1p`; 2,000 Seurat HVGs; `set_raw`; HVG subsetting; scale capped at 10;
PCA50; neighbors15 using PC30; and UMAP3 with `min_dist=0.3` and seed 0.

`simulation_2.h5ad` and `simulation_3.h5ad` are processed floating-point
expression inputs rather than raw-count files. They include simulated branch
labels, reference pseudotime, PCA50, neighbor matrices, and UMAP3. Their YAML
configurations deterministically regenerate PCA50, neighbors30/PC30, and UMAP3
before a raw-stage main-model run, so historical AnnData fields are not used to
tune the final graph.

Expression-count provenance and annotation provenance are reported separately
in `docs/data_availability.md`, `docs/datasets.md`, and
`reproducibility/data/datasets_cn.md`. Packaging these files does not replace or
broaden upstream licenses, terms of use, attribution requirements, or
controlled-access conditions.

## Main scCGRL runs

Available dataset keys are `human_myeloid`, `mouse_pancreas`,
`human_bone_marrow`, `simulation_2`, and `simulation_3`.

```powershell
# One complete run
python run_sccgrl.py --dataset mouse_pancreas --seed 42 --episodes 10000

# Ten paired seeds beginning at 42
python run_sccgrl.py --dataset simulation_2 --seed 42 --runs 10 --episodes 10000

# Formal 50-run protocol, seeds 42--91
python experiments/repeat_50_runs/run_repeat_50.py --dataset human_myeloid
```

## Ten-method benchmark

Python baselines are PAGA, DPT, and Palantir. R baselines are Slingshot,
Monocle 1/2/3, SLICER, TSCAN, and SCORPIUS. Each method retains its audited
official/recommended internal workflow. The same cell basis, root-information
policy, pseudotime normalization, evaluation adapter, and resource recorder are
used across methods. No terminal identity, terminal count, true branch count,
or reference pseudotime is supplied to inference.

Install `benchmark/environments/benchmark_python.yml` and the isolated R
environments described under `benchmark/environments/`. External R locations
are configured with the environment variables documented in
`docs/benchmark_methods.md`.

```powershell
python -m benchmark.run_benchmark --datasets human_myeloid --runs 10 --seed 42 --resume
```

Method-level outputs are written to
`results/benchmark/<method>/<dataset>/`. Combined outputs are written to
`results/benchmark/combined/benchmark_all_runs.csv`,
`benchmark_summary.csv`, and `benchmark_completion.csv`. Existing scCGRL
50-run rows are imported from `results/repeat_50_runs`; scCGRL is not rerun by
the benchmark driver.

## Publication figures

The three real datasets use separate figure programs because their lineage
ordering, marker panels, and heatmap layouts differ biologically. Each program
exports PNG (300 dpi), TIFF (600 dpi), EPS, and editable-text SVG.

```powershell
python figures/human_myeloid/make_figures_human_myeloid.py --seed 42
python figures/mouse_pancreas/make_figures_mouse_pancreas.py --seed 42
python figures/human_bone_marrow/make_figures_human_bone_marrow.py --seed 42
```

## Summaries and verification

```powershell
python tables/repeat_50_runs_summary.py results/repeat_50_runs results/tables/repeat_50_runs_summary.csv
python tables/benchmark_summary.py
python tables/runtime_memory_summary.py results/benchmark/combined/benchmark_all_runs.csv results/tables/runtime_memory_summary.csv
python reproducibility/data/verify_raw_inputs.py
```

See `results/README.md` for the output layout, `docs/reproduction_map.md` for
the source-to-module map, `docs/software_versions.md` for environments, and
`docs/release_validation.md` for the committed seed-42 numerical check.

## Release identity and exact commit

Every run manifest records the Git commit returned by `git rev-parse HEAD`.
Before citing or archiving a run, verify the checkout and data package with:

```powershell
git rev-parse HEAD
python reproducibility/verify_public_release.py
```

The publication release is tagged `v1.0.0`. Use the full commit hash rather
than only the mutable branch name in manuscripts and response letters.

## License and citation

Original scCGRL code is released under the MIT License. `CITATION.cff` records
the software and manuscript authors. Upstream data and comparison-method
licenses remain applicable independently; see `DATA_LICENSES.md`.
