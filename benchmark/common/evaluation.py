# Exact audited evaluation adapter; uses trajectory metric v2.0.
import json

import numpy as np
import pandas as pd
from scipy import stats


def normalize01(values):
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No finite pseudotime values")
    values[~finite] = np.nanmedian(values[finite])
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high <= low else (values - low) / (high - low)


def endpoints_from_branches(pseudotime, branches):
    values = np.asarray(pseudotime, dtype=float)
    branches = np.asarray(branches).astype(str)
    indices = []
    for branch in pd.unique(branches):
        members = np.flatnonzero((branches == branch) & np.isfinite(values))
        if members.size:
            indices.append(int(members[np.argmax(values[members])]))
    return sorted(set(indices))


def paths_from_prediction(pseudotime, branches, root_index):
    values = np.asarray(pseudotime, dtype=float)
    branches = np.asarray(branches).astype(str)
    paths = {}
    for branch in pd.unique(branches):
        members = np.flatnonzero((branches == branch) & np.isfinite(values))
        if not members.size:
            continue
        ordered = members[np.argsort(values[members], kind="mergesort")].tolist()
        ordered = [int(value) for value in ordered if int(value) != int(root_index)]
        if ordered:
            paths[int(ordered[-1])] = [int(root_index)] + ordered
    if not paths:
        raise ValueError("Prediction has no usable ordered path")
    return paths


def _safe_correlation(function, left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return np.nan
    result = function(left[mask], right[mask])
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def reference_correlations(context, config, pseudotime):
    if config.get("trajectory_reference_kind") == "simulated_tree":
        reference = pd.to_numeric(
            context["adata"].obs[config["reference_pseudotime_column"]], errors="coerce"
        ).to_numpy(float)
        source = "stored simulation generating pseudotime"
    else:
        depth = {str(key): float(value) for key, value in config.get("depth_map", {}).items()}
        reference = np.asarray([depth.get(str(label), np.nan) for label in context["labels"]])
        source = "curated annotation depth; consistency, not absolute ground truth"
    return {
        "comparison_reference_source": source,
        "Pearson": _safe_correlation(stats.pearsonr, pseudotime, reference),
        "Spearman": _safe_correlation(stats.spearmanr, pseudotime, reference),
        "Kendall": _safe_correlation(stats.kendalltau, pseudotime, reference),
    }


def evaluate(namespace, context, config, seed, root_index, pseudotime, branches,
             has_native_branches=True, has_native_topology=True):
    pseudotime = normalize01(pseudotime)
    paths = paths_from_prediction(pseudotime, branches, root_index)
    bundle = {
        "global_pseudotime": pseudotime,
        "global_mask": np.isfinite(pseudotime),
        "rf_pseudotime": pseudotime,
        "rf_mask": np.isfinite(pseudotime),
    }
    metrics = namespace["compute_trajectory_benchmark_metrics"](
        context, paths, bundle, config, seed=int(seed)
    )
    if not has_native_branches:
        for column in (
            "F1_branches", "branch_recovery", "branch_relevance",
            "branch_precision", "branch_recall",
        ):
            if column in metrics:
                metrics[column] = np.nan
    if not has_native_topology:
        for column in (
            "HIM_similarity", "topology_edge_precision", "topology_edge_recall",
            "topology_edge_F1", "terminal_state_precision", "terminal_state_recall",
            "terminal_state_F1",
        ):
            if column in metrics:
                metrics[column] = np.nan
    if not has_native_branches or not has_native_topology:
        # The formal overall score is a geometric mean of heterogeneous
        # components and is undefined when a required native structure is absent.
        if "trajectory_overall_geometric_mean" in metrics:
            metrics["trajectory_overall_geometric_mean"] = np.nan
    metrics.update(reference_correlations(context, config, pseudotime))
    endpoints = endpoints_from_branches(pseudotime, branches) if has_native_branches else []
    metrics.update({
        "endpoint_count": len(endpoints) if has_native_branches else np.nan,
        "endpoint_indices": json.dumps(endpoints),
        "endpoint_cell_ids": json.dumps([str(context["adata"].obs_names[i]) for i in endpoints]),
        "branch_count": (
            int(len(pd.unique(np.asarray(branches).astype(str))))
            if has_native_branches else np.nan
        ),
        "native_branch_output_available": bool(has_native_branches),
        "native_topology_output_available": bool(has_native_topology),
        "structural_metric_missing_reason": (
            "" if has_native_branches and has_native_topology
            else "Method did not return native branch/topology output; structural metrics are NaN."
        ),
    })
    projection = namespace["_projection_from_predicted_paths"](
        paths, context["model_coords"]
    )
    return metrics, endpoints, paths, projection


def numeric_summary(frame, status_column="status"):
    rows = []
    success = frame if status_column not in frame else frame[frame[status_column] == "success"]
    excluded = {
        "dataset", "method", "status", "error_message", "traceback", "seed", "random_seed",
        "cell_id", "method_parameters_json", "endpoint_indices", "endpoint_cell_ids",
        "comparison_reference_source", "reference_trajectory_source", "reference_trajectory_note",
        "metric_reference", "metric_implementation", "preprocessing_audit_json",
    }
    for column in success.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(success[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        n = int(len(values))
        sd = float(values.std(ddof=1)) if n > 1 else 0.0
        margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
        rows.append({
            "metric": column,
            "n": n,
            "mean": float(values.mean()),
            "median": float(values.median()),
            "standard_deviation": sd,
            "confidence_interval_95_lower": float(values.mean() - margin),
            "confidence_interval_95_upper": float(values.mean() + margin),
        })
    return pd.DataFrame(rows)
