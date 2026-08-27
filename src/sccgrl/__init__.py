"""Public API for the audited scCGRL implementation (code revision 1.6)."""

from .preprocessing import load_prepared_dataset
from .graph_endpoints import find_start_and_endpoints, find_minimal_connected_k
from .q_learning import QLearningPathFinder
from .trajectory import (
    CODE_REVISION,
    compute_pseudotime_outputs,
    prepare_model_context,
    run_repeated_experiments,
    run_single_experiment,
    set_reproducible_seed,
    train_one_q_learning,
)

__version__ = CODE_REVISION

__all__ = [
    "CODE_REVISION",
    "QLearningPathFinder",
    "compute_pseudotime_outputs",
    "find_minimal_connected_k",
    "find_start_and_endpoints",
    "load_prepared_dataset",
    "prepare_model_context",
    "run_repeated_experiments",
    "run_single_experiment",
    "set_reproducible_seed",
    "train_one_q_learning",
]
