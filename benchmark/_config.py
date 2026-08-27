"""Portable paths and fixed settings for the official-method benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


VERSION = "v1.0"
ARTIFACT_DATE = "2026-08-12"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("SCCGRL_PROJECT_ROOT", REPOSITORY_ROOT)).resolve()
BENCHMARK_ROOT = REPOSITORY_ROOT
RESULTS_ROOT = Path(
    os.environ.get("SCCGRL_BENCHMARK_RESULTS", REPOSITORY_ROOT / "results" / "benchmark")
).resolve()
SCCGRL_RESULTS_ROOT = Path(
    os.environ.get("SCCGRL_REPEAT_RESULTS", REPOSITORY_ROOT / "results" / "repeat_50_runs")
).resolve()

# R installations are intentionally external to the repository. Set these
# environment variables to the isolated environments created from the YAML files.
SYSTEM_RSCRIPT = Path(os.environ.get("SCCGRL_RSCRIPT", shutil.which("Rscript") or "Rscript"))
CURRENT_R_LIBRARY = Path(os.environ.get("SCCGRL_R_LIBRARY", REPOSITORY_ROOT / "benchmark" / "environments" / "r_library"))
MONOCLE2_ENV = Path(os.environ.get("SCCGRL_MONOCLE2_ENV", REPOSITORY_ROOT / "benchmark" / "environments" / "monocle2"))
MONOCLE2_R_LIBRARY = Path(os.environ.get("SCCGRL_MONOCLE2_LIBRARY", MONOCLE2_ENV / "lib" / "R" / "library"))
MONOCLE2_LEGACY_DPLYR_LIBRARY = Path(os.environ.get("SCCGRL_MONOCLE2_DPLYR_LIBRARY", MONOCLE2_R_LIBRARY))
MONOCLE3_R_LIBRARY = Path(os.environ.get("SCCGRL_MONOCLE3_LIBRARY", REPOSITORY_ROOT / "benchmark" / "environments" / "monocle3" / "library"))
MONOCLE1_ENV = Path(os.environ.get("SCCGRL_MONOCLE1_ENV", REPOSITORY_ROOT / "benchmark" / "environments" / "monocle1"))
MONOCLE1_RSCRIPT = Path(os.environ.get("SCCGRL_MONOCLE1_RSCRIPT", MONOCLE1_ENV / "runtime" / "R-3.1.3" / "bin" / "x64" / "Rscript.exe"))
MONOCLE1_R_LIBRARY = Path(os.environ.get("SCCGRL_MONOCLE1_LIBRARY", MONOCLE1_ENV / "library"))
CONDA_EXECUTABLE = Path(os.environ.get("SCCGRL_CONDA", shutil.which("conda") or "conda"))
R_RUNNER = BENCHMARK_ROOT / "benchmark" / "common" / "r_methods_runner.R"
MONOCLE1_RUNNER = BENCHMARK_ROOT / "benchmark" / "monocle1.R"

DATASETS = (
    "human_myeloid", "mouse_pancreas", "human_bone_marrow",
    "simulation_2", "simulation_3",
)
EXTERNAL_METHODS = (
    "PAGA", "DPT", "Palantir", "Slingshot", "Monocle3", "Monocle2",
    "Monocle1", "SLICER", "TSCAN", "SCORPIUS",
)
ALL_METHODS = ("scCGRL",) + EXTERNAL_METHODS
R_METHODS = frozenset({"Slingshot", "Monocle3", "Monocle2", "SLICER", "TSCAN", "SCORPIUS"})
BASE_SEED = 42
REPEAT_RUNS = 10
SEEDS = tuple(range(BASE_SEED, BASE_SEED + REPEAT_RUNS))

# The shared root is supplied only where the official method supports it.
# No benchmark method receives terminal identities, counts, or branch labels.
ROOT_USAGE = {
    "PAGA": "root-free PAGA topology; shared root used by Scanpy DPT ordering",
    "DPT": "shared exact cell as adata.uns['iroot']",
    "Palantir": "shared exact cell as early_cell",
    "Slingshot": "unsupervised cluster containing shared root as start.clus",
    "Monocle3": "shared exact cell as root_cells",
    "Monocle2": "Monocle state containing shared root as root_state",
    "Monocle1": "shared exact cell passed to historical orderCells root_cell",
    "SLICER": "shared exact cell as start cell",
    "TSCAN": "unsupervised TSCAN cluster containing shared root as start cluster",
    "SCORPIUS": "shared root used only for direction after root-free inference",
}
