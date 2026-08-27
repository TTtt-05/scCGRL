# Processed data

This directory contains the five authoritative processed H5AD inputs used by
the release:

| Dataset | Processed shape | Label/reference columns | Stored representations |
|---|---:|---|---|
| human_myeloid | 3,264 x 2,000 | `cluster` | PCA50, neighbors, UMAP3 |
| mouse_pancreas | 2,780 x 2,000 | `day`, `clusters_fig6_broad_final` | PCA50, neighbors, UMAP3 |
| human_bone_marrow | 7,225 x 2,000 | `sample`, `batch`, `celltype` | PCA50, neighbors, UMAP3 |
| simulation_2 | 2,000 x 1,000 | simulated `branch`, reference `pseudotime` | PCA50, neighbors, UMAP3 |
| simulation_3 | 3,000 x 1,000 | simulated `branch`, reference `pseudotime` | PCA50, neighbors, UMAP3 |

The human-bone-marrow cache is generated after excluding ETP (60 cells) and
BcellPre (154 cells), reducing 7,439 raw cells to 7,225 model cells. The three
real-data caches contain 2,000 HVGs and preserve the normalized all-gene matrix
in `.raw`.

The simulation H5AD files are processed floating-point expression inputs, not
raw-count matrices. Their configured main-model raw-stage workflow
deterministically regenerates PCA50, neighbors30/PC30, and UMAP3 before graph
construction.

Use `processed_inputs_audit.csv`, `processed_inputs_manifest.json`, and
`processed_inputs_checksums.sha256` to verify dimensions, identifiers,
preprocessing settings, and exact packaged files.
