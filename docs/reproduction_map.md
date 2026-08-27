# Reproduction map

| Repository target | Audited source | Purpose |
|---|---|---|
| `configs/*.yaml` | `2026-08-17_scCGRL_five_datasets_v1.ipynb`, configuration cell | Final dataset parameters, labels, preprocessing, references, and plotting dimensions |
| `src/sccgrl/preprocessing.py` | final five-dataset notebook preprocessing functions | Raw/processed input validation and configured transformations |
| `src/sccgrl/graph_endpoints.py` | final five-dataset notebook graph/end-point functions | Adaptive graph construction and endpoint selection |
| `src/sccgrl/q_learning.py` | final five-dataset notebook Q-learning class | Episode training, rewards, paths, and Q-value state |
| `src/sccgrl/trajectory.py` | final five-dataset notebook orchestration cells | Complete single-run and repeated-run pipeline |
| `src/sccgrl/pseudotime_mapping.py` | final five-dataset notebook pseudotime cells | Graph/global pseudotime mapping |
| `src/sccgrl/rf_mapping.py` | final five-dataset notebook RF cells | All-cell pseudotime propagation and held-out RF metrics |
| `src/sccgrl/metrics.py` | trajectory metric v2.0 and final notebook evaluation cells | Unified ordering, branch, topology, feature, endpoint, and resource metrics |
| `benchmark/` | official-preprocessing benchmark v1.0 under `response/method_official_preprocessing_v1.0/8.12` | Ten official/recommended baseline workflows and shared evaluation |
| `experiments/repeat_50_runs/` | final five-dataset repetition cell | Seeds 42--91 main-model runs |
| `figures/<dataset>/` | final dataset-specific comprehensive-figure cells | Publication figure generation without homogenizing biological layouts |
| `reproducibility/data/` | verified raw-public preparation programs | Exact source selection, annotation provenance, matrix checks, and processed caches |
| `tables/` | final benchmark/repetition summary utilities | Statistical and runtime/memory summaries |

All runtime input paths are resolved relative to the repository unless an
explicit command-line path or documented environment variable is provided.
