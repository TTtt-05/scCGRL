# Data layout

`raw/` contains the three verified real-data integer count matrices.
`processed/` contains all five configured model inputs, including required cell
labels/reference metadata, PCA50, neighbor matrices, and UMAP3.

The real processed inputs are deterministic seed-42 preprocessing caches. The
human-bone-marrow cache contains 7,225 cells after excluding ETP and BcellPre;
the raw file retains the complete 7,439 cells. The two simulation datasets are
distributed only as processed floating-point expression inputs and include
their simulated branch/reference-pseudotime metadata.

Verify the packaged files with `checksums.sha256`,
`processed/processed_inputs_checksums.sha256`, and the programs under
`reproducibility/data`.

Data inclusion does not alter upstream terms. See `DATA_LICENSES.md`,
`docs/data_availability.md`, and `docs/datasets.md`.
