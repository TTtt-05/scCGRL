# Exact audited project/input adapter from the 8.12 benchmark.
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._config import PROJECT_ROOT, SCCGRL_RESULTS_ROOT, SEEDS
from experiments.common import build_namespace
import scanpy as sc


def read_existing_csv(path):
    """Read project result CSVs without rewriting their original encoding."""
    last_error = None
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def load_project_namespace():
    """Build the benchmark namespace from repository modules, not a notebook."""
    namespace = build_namespace(PROJECT_ROOT)
    namespace["sc"] = sc
    if namespace.get("METRIC_VERSION") != "v2.0":
        raise RuntimeError("The formal trajectory metric v2.0 layer was not loaded")
    return namespace


def stable_digest(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def matrix_digest(values):
    digest = hashlib.sha256()
    if sp.issparse(values):
        matrix = values.tocsr(copy=False)
        for item in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.ascontiguousarray(item).tobytes())
        digest.update(str(matrix.shape).encode())
    else:
        array = np.ascontiguousarray(np.asarray(values))
        digest.update(array.tobytes())
        digest.update(str(array.shape).encode())
    return digest.hexdigest()


def resolve_dataset(namespace, dataset_key):
    config = copy.deepcopy(namespace["DATASET_CONFIGS"][dataset_key])
    input_path = Path(namespace["resolve_input"](config)).resolve()
    return config, input_path


def evaluation_context(namespace, input_path, config, seed):
    """Exact main-Notebook cells and reference information, not method input."""
    return namespace["load_prepared_dataset"](input_path, copy.deepcopy(config), int(seed))


def select_same_cells(adata, cell_ids):
    requested = pd.Index(np.asarray(cell_ids).astype(str))
    available = pd.Index(adata.obs_names.astype(str))
    positions = available.get_indexer(requested)
    if np.any(positions < 0):
        missing = requested[positions < 0].tolist()[:5]
        raise KeyError(f"Input does not contain evaluation cells: {missing}")
    result = adata[positions].copy()
    if not np.array_equal(result.obs_names.astype(str), requested.to_numpy()):
        raise AssertionError("Cell order differs after baseline input subsetting")
    return result


def best_expression_source(adata):
    """Return the least transformed available matrix without altering cells."""
    if "counts" in adata.layers:
        return adata.layers["counts"], np.asarray(adata.var_names).astype(str), "layers[counts]", True
    if adata.raw is not None:
        matrix = adata.raw.X
        data = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
        is_count = bool(data.size and np.all(data >= 0) and np.mean(np.isclose(data, np.round(data))) > 0.98)
        return matrix, np.asarray(adata.raw.var_names).astype(str), "adata.raw.X", is_count
    matrix = adata.X
    data = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    is_count = bool(data.size and np.all(data >= 0) and np.mean(np.isclose(data, np.round(data))) > 0.98)
    return matrix, np.asarray(adata.var_names).astype(str), "adata.X", is_count


def build_baseline_input(namespace, input_path, config, evaluation):
    original = namespace["sc"].read_h5ad(input_path)
    original = select_same_cells(original, evaluation["adata"].obs_names)
    matrix, genes, source, is_count = best_expression_source(original)
    matrix = matrix.tocsr().copy() if sp.issparse(matrix) else np.asarray(matrix).copy()
    return {
        "adata": original,
        "expression": matrix,
        "gene_ids": genes,
        "expression_source": source,
        "expression_is_counts": is_count,
        "cell_ids": np.asarray(original.obs_names).astype(str),
        "cell_ids_sha256": stable_digest(np.asarray(original.obs_names).astype("U")),
        "expression_sha256": matrix_digest(matrix),
    }


def build_root_anchor(namespace, dataset_key, config, input_path, seed, context=None):
    # Read the same-seed root from the existing scCGRL result.  Do not rerun
    # scCGRL preprocessing, endpoint discovery, Q-learning, or pseudotime.
    metrics_path = (
        SCCGRL_RESULTS_ROOT / f"{dataset_key}_repeat_50" / f"{dataset_key}_50_runs.csv"
    )
    existing = read_existing_csv(metrics_path)
    seed_column = "random_seed" if "random_seed" in existing else "seed"
    existing = existing[pd.to_numeric(existing[seed_column], errors="coerce") == int(seed)]
    if len(existing) != 1 or "start_cell" not in existing:
        raise RuntimeError(f"Cannot read the unique seed-{seed} scCGRL root from {metrics_path}")
    start_index = int(existing.iloc[0]["start_cell"])
    if context is None:
        context = evaluation_context(namespace, input_path, config, int(seed))
    if not 0 <= start_index < context["adata"].n_obs:
        raise IndexError(f"Stored scCGRL root index {start_index} is outside the prepared cells")
    return {
        "start_cell_id": str(context["adata"].obs_names[start_index]),
        "scCGRL_start_index": start_index,
        "K": int(existing.iloc[0]["K"]),
        "source": str(metrics_path),
    }


def import_sccgrl_rows(dataset_key, runs=10):
    path = SCCGRL_RESULTS_ROOT / f"{dataset_key}_repeat_50" / f"{dataset_key}_50_runs.csv"
    frame = read_existing_csv(path)
    seed_column = "random_seed" if "random_seed" in frame else "seed"
    frame = frame[pd.to_numeric(frame[seed_column], errors="coerce").isin(SEEDS[:runs])].copy()
    frame["dataset"] = dataset_key
    frame["method"] = "scCGRL"
    frame["status"] = "success"
    frame["error_message"] = ""
    frame["scCGRL_result_source"] = str(path)
    frame["scCGRL_rerun"] = False
    # Normalize historical scCGRL column names into the benchmark-wide schema
    # without changing or recomputing source values.
    simulated = str(frame.get("evaluation_mode", pd.Series([""])).iloc[0]) == "gold_reference_simulation"
    aliases = {
        "Pearson": "reference_pseudotime_Pearson" if simulated else "global_Pearson",
        "Spearman": "reference_pseudotime_Spearman" if simulated else "global_Spearman",
        "Kendall": "reference_pseudotime_Kendall" if simulated else "global_Kendall",
    }
    for target, source in aliases.items():
        if source in frame:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")
    frame["comparison_reference_source"] = (
        "stored simulation generating pseudotime"
        if simulated else "curated annotation depth; consistency, not absolute ground truth"
    )
    return frame
