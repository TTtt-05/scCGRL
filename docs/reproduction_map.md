# Reproduction map

| Repository target | Audited source | Purpose |
|---|---|---|
| `configs/*.yaml` | final five-dataset configuration cells | Labels, preprocessing, references, endpoint settings, and plotting dimensions |
| `src/sccgrl/preprocessing.py` | final common preprocessing functions | Raw/processed input validation and configured transformations |
| `src/sccgrl/graph_endpoints.py` | final graph and endpoint functions | Adaptive graph construction and endpoint selection |
| `src/sccgrl/q_learning.py` | final Q-learning class | Episode training, rewards, paths, and Q-value state |
| `src/sccgrl/trajectory.py` | final orchestration cells | Complete scCGRL inference pipeline |
| `src/sccgrl/pseudotime_mapping.py` | final pseudotime functions | Graph/global pseudotime mapping |
| `src/sccgrl/rf_mapping.py` | final RF functions | All-cell pseudotime propagation and RF metrics |
| `src/sccgrl/metrics.py` | trajectory metric v2.0 | Unified ordering, branch, topology, feature, and endpoint metrics |
| `benchmark/` | official-preprocessing benchmark implementation | Ten native/recommended baseline workflows, resource recording, and shared evaluation |
| `figures/<dataset>/` | final dataset-specific comprehensive-figure functions | Publication figures without forcing one biological layout across datasets |
| `reproducibility/data/` | verified raw-public preparation programs | Source selection, annotation provenance, exact matrix checks, and processed inputs |

All runtime paths are repository-relative unless the user supplies an explicit
input or output path. Runtime-generated outputs are written under `results/`
and are not part of the committed public package.
