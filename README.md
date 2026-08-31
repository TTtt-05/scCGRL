# scCGRL

This repository contains the public reproducibility resources for scCGRL:

- verified raw-count inputs for the three empirical datasets;
- model-ready processed inputs for all five datasets;
- the scCGRL model implementation and command-line runner;
- ten comparison-method runners and reproducible software environments;
- dataset-specific figure-generating scripts;
- software-version, data-provenance, licensing, and verification records.

Generated run outputs are intentionally not committed. Every command writes new
outputs under `results/`, which is excluded from Git. The curated benchmark
workbook used as Supplementary Table S11 is included at
`benchmark/results/Supplementary_Table_S11.xlsx`.

## Repository contents

| Path | Contents |
|---|---|
| `configs/` | Audited YAML configuration for each dataset. |
| `data/raw/` | Integer raw-count H5AD files for the three empirical datasets. |
| `data/processed/` | Five model-ready H5AD inputs containing labels, PCA50, KNN matrices, and UMAP3. |
| `src/sccgrl/` | Preprocessing, endpoint discovery, Q-learning, trajectory inference, pseudotime mapping, RF propagation, metrics, and I/O. |
| `run_sccgrl.py` | Main scCGRL command-line entry point. |
| `benchmark/` | PAGA, DPT, Palantir, Slingshot, Monocle 1/2/3, SLICER, TSCAN, and SCORPIUS runners, shared evaluation, and environments. |
| `benchmark/results/Supplementary_Table_S11.xlsx` | Mean, SD, median, and 95% CI benchmark statistics for scCGRL and ten comparison methods across five datasets. |
| `figures/` | Dataset-specific publication-figure scripts. |
| `reproducibility/data/` | Data preparation, exact verification, provenance, and checksums. |
| `docs/` | Dataset, benchmark, software-version, and reproduction documentation. |

## Installation

Git LFS is required because the H5AD inputs are stored as LFS objects.

```powershell
git lfs install
git clone https://github.com/TTtt-05/scCGRL.git
Set-Location scCGRL
git lfs pull

conda env create -f environment.yml
conda activate sccgrl_20260827
```

The main environment pins the versions used for release verification. The
comparison methods use the isolated environments in `benchmark/environments/`.

## Data

| Dataset | Raw input | Processed input | Processed shape | Label column |
|---|---|---|---:|---|
| `human_myeloid` | yes | yes | 3,264 x 2,000 | `cluster` |
| `mouse_pancreas` | yes | yes | 2,780 x 2,000 | `clusters_fig6_broad_final` |
| `human_bone_marrow` | yes | yes | 7,225 x 2,000 | `celltype` |
| `simulation_2` | no raw-count artifact | yes | 2,000 x 1,000 | `branch` |
| `simulation_3` | no raw-count artifact | yes | 3,000 x 1,000 | `branch` |

The bone-marrow processed input excludes ETP (60 cells) and BcellPre (154
cells) during configured preprocessing; the bundled raw input retains all
7,439 cells. The two simulation files are processed floating-point expression
inputs generated with PROSSTT and include simulation truth.

For empirical data, the common preprocessing is dataset-specific exclusion,
`normalize_total(target_sum=10000)`, natural `log1p`, 2,000 Seurat HVGs,
`set_raw`, HVG subsetting, scaling capped at 10, PCA50, neighbors15 using PC30,
and UMAP3 (`min_dist=0.3`, seed 0). Exact sources, selection rules, annotation
provenance, transformations, and redistribution boundaries are documented in
`DATA_LICENSES.md`, `docs/data_availability.md`, and
`reproducibility/data/`.

## Run scCGRL

Run a complete seed-42 analysis from packaged raw counts:

```powershell
python run_sccgrl.py --dataset human_myeloid --input-stage raw --seed 42 --episodes 10000
```

Reuse the verified model-ready processed input:

```powershell
python run_sccgrl.py --dataset human_myeloid --input-stage processed --seed 42 --episodes 10000
```

Both commands write to `results/human_myeloid/seed42/` and record pseudotime,
metrics, resource measurements, processed AnnData, model state, figures, run
parameters, software versions, and the Git commit.

For Random Forest propagation, trajectory/path cells are split reproducibly
into 80% training cells and 20% held-out testing cells using the run seed. MSE
and R2 are calculated only on the held-out path cells. The same model fitted on
the 80% training subset is then applied to all cells without refitting on the
test subset. Each single-run directory records the exact split in
`<dataset>_rf_path_cell_split.csv`; `<dataset>_pseudotime.csv` identifies every
cell as `train_path`, `test_path`, or `mapped_non_path`.

Q-learning receives the graph, coordinates, model-selected start node, and
model-selected endpoint indices. Cell-type labels are not passed into the
Q-learning object, and its diagnostic plots identify endpoints only by node
index; biological endpoint types are evaluated after inference.

## Run the ten comparison methods

Create the Python baseline environment:

```powershell
conda env create -f benchmark/environments/benchmark_python.yml
```

Create each required R environment as described by the YAML files and
`benchmark/environments/README.md`. The benchmark uses the same-seed scCGRL
root without rerunning the main model internally, so first generate the
matching local scCGRL rows (or point `SCCGRL_REPEAT_RESULTS` to existing formal
rows), then run the baselines:

```powershell
python run_sccgrl.py --dataset human_myeloid --input-stage processed --seed 42 --runs 10 --episodes 10000
python -m benchmark.run_benchmark --datasets human_myeloid --runs 10 --seed 42 --skip-sccgrl-import --resume
```

The benchmark does not supply terminal identity, terminal count, true branch
count, or reference pseudotime to inference. Unsupported branch/topology
metrics remain `NaN`; failures are recorded and do not terminate later runs.
Outputs are generated under `results/benchmark/`.

The benchmark statistics reported in the manuscript are also provided as the
tracked workbook `benchmark/results/Supplementary_Table_S11.xlsx`, with separate
worksheets for mean, SD, median, and 95% confidence intervals.

## Generate publication figures

The three empirical datasets deliberately use separate layouts and marker
panels. Each script exports PNG, TIFF, EPS, and editable SVG.

```powershell
python figures/human_myeloid/make_figures_human_myeloid.py --seed 42
python figures/mouse_pancreas/make_figures_mouse_pancreas.py --seed 42
python figures/human_bone_marrow/make_figures_human_bone_marrow.py --seed 42
```

Simulation-specific figure runners are under `figures/simulation_2/` and
`figures/simulation_3/`. Generated files are written to `results/figures/`.

## Verify the public package

```powershell
python reproducibility/data/verify_raw_inputs.py
python reproducibility/verify_public_release.py
git rev-parse HEAD
```

The publication release is identified by the immutable Git tag `v1.0.0` and
its exact 40-character commit hash. Cite that hash rather than only the mutable
`main` branch. See `CITATION.cff` for software citation metadata.

## License

Original scCGRL code is released under the MIT License. Third-party datasets
and comparison methods retain their own licenses and attribution requirements;
see `DATA_LICENSES.md` and the upstream sources before redistribution.
