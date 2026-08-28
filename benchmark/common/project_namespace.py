"""Load the audited scCGRL modules for shared benchmark evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sccgrl.graph_endpoints as graph_endpoints
import sccgrl.metrics as metrics
import sccgrl.preprocessing as preprocessing
import sccgrl.pseudotime_mapping as pseudotime_mapping
import sccgrl.q_learning as q_learning
import sccgrl.rf_mapping as rf_mapping
import sccgrl.trajectory as trajectory
from run_sccgrl import DATASET_KEYS, load_dataset_config, resolve_input


def dataset_configs(project_root=None):
    return {
        key: load_dataset_config(key, project_root=project_root)
        for key in DATASET_KEYS
    }


def build_namespace(project_root=None):
    """Expose the final model/evaluation functions without executing notebooks."""
    namespace = {"DATASET_CONFIGS": dataset_configs(project_root)}
    for module in (
        graph_endpoints,
        metrics,
        preprocessing,
        pseudotime_mapping,
        q_learning,
        rf_mapping,
        trajectory,
    ):
        namespace.update(
            {
                name: value
                for name, value in vars(module).items()
                if not name.startswith("__")
            }
        )

    def _resolve(config):
        return resolve_input(copy.deepcopy(config))

    namespace["resolve_input"] = _resolve
    required = (
        "load_prepared_dataset",
        "find_start_and_endpoints",
        "train_one_q_learning",
        "compute_branch_pseudotimes",
        "compute_graph_distance_global_pseudotime",
        "compute_rf_pseudotime_with_validation",
        "compute_trajectory_benchmark_metrics",
        "PeakRSSMonitor",
    )
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(f"Benchmark namespace missing: {missing}")
    return namespace
