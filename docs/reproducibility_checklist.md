# Reproducibility checklist

- [x] Five dataset configurations are stored as YAML.
- [x] Three empirical datasets are bundled as verified integer raw-count H5AD files.
- [x] Five processed inputs retain labels, PCA50, KNN matrices, and UMAP3.
- [x] Expression and annotation sources are documented separately.
- [x] Raw input shapes, cell IDs, gene IDs, totals, nnz, provenance, and SHA256 are verifiable.
- [x] The main scCGRL method has a portable command-line entry point.
- [x] The ten comparison methods have method-specific runners and environment definitions.
- [x] Baseline failures remain auditable and unavailable metrics are not fabricated.
- [x] Dataset-specific figure scripts export PNG, TIFF, EPS, and editable SVG.
- [x] Main and benchmark software versions are recorded.
- [x] Original code and third-party data licensing boundaries are explicit.
- [x] H5AD artifacts are tracked with Git LFS.
- [x] `reproducibility/verify_public_release.py` checks package scope, files, checksums, H5AD structure, portable paths, Git identity, and Git LFS.
- [x] The public release is identified by tag `v1.0.0` and an exact Git commit hash.
