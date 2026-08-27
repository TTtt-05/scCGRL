# Raw data

This directory contains the three verified, dataset-named real-data inputs:

- `human_myeloid.h5ad`: 3,264 cells x 19,089 genes; label `cluster`.
- `mouse_pancreas.h5ad`: 2,780 cells x 27,998 genes; labels `day` and
  `clusters_fig6_broad_final`.
- `human_bone_marrow.h5ad`: 7,439 cells x 17,226 genes; label `celltype`.

Their `X` matrices contain integer raw counts. No normalization, log transform,
HVG restriction, scaling, PCA, neighbors, or UMAP is stored as raw input.
Human-bone-marrow ETP and BcellPre cells remain in the raw file and are removed
only by the configured preprocessing pipeline.

Use `reproducibility/data/verify_raw_inputs.py` and `data/checksums.sha256` for
exact verification. Expression and annotation provenance are documented in
`reproducibility/data/datasets_cn.md`, `datasets_en.md`, and
`docs/data_availability.md`.
