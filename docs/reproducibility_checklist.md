# Reproducibility checklist

- [x] Five dataset configurations are stored as YAML.
- [x] Three real datasets are bundled as verified integer raw-count H5AD files.
- [x] Five processed inputs retain their required labels and low-dimensional representations.
- [x] Human bone marrow excludes ETP and BcellPre only during configured preprocessing.
- [x] PCA, neighbor, UMAP, K, Q-learning, RF, metric, seed, runtime, and memory settings are recorded.
- [x] The main method has one portable command-line entry point.
- [x] The ten baselines have method-specific runners and isolated environment descriptions.
- [x] Baseline failures are recorded without fabricating unavailable branch or topology outputs.
- [x] Existing scCGRL 50-run rows can be imported without rerunning scCGRL.
- [x] Publication figures use dataset-specific scripts and four output formats.
- [x] Raw input shapes, integer counts, cell IDs, gene IDs, totals, nnz, provenance, and SHA256 can be verified.
- [x] Expression and annotation sources are documented separately.
