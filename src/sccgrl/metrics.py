# Canonical metric source: scCGRL reference-trajectory metrics v2.0
# Metric version: v2.0 (2026-08-09)
"""Reference-trajectory metrics for the five configured scCGRL datasets.

This module intentionally replaces the layered metric overrides previously
embedded in the notebook.  It follows the evaluation separation used by
Saelens et al. (Nature Biotechnology, 2019): trajectory accuracy is measured
against an explicit reference trajectory, while stability and resource use
remain separate evaluation dimensions.

Simulation datasets use their stored generating trajectory.  Real datasets
use the curated, annotation/literature-derived lineage graph already declared
in ``annotation_topology_edges``.  Scores for real data therefore quantify
agreement with that curated reference and are not absolute biological truth.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from scipy.io import mmwrite
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


METRIC_VERSION = "v2.0"
SAELENS_REFERENCE = (
    "Saelens et al., Nature Biotechnology (2019), "
    "doi:10.1038/s41587-019-0071-9"
)
CORE_TRAJECTORY_METRICS = (
    "cor_dist",
    "F1_branches",
    "HIM_similarity",
    "wcor_features",
)
SIMULATION_ONLY_COLUMNS = (
    "reference_pseudotime_Pearson",
    "reference_pseudotime_Spearman",
    "reference_pseudotime_Kendall",
)
REAL_ONLY_COLUMNS = (
    "marker_gene_mean_abs_spearman",
    "marker_gene_median_abs_spearman",
    "marker_genes_configured",
    "marker_genes_available",
    "marker_gene_correlations_json",
)
TRAJECTORY_METRIC_COLUMNS = [
    "evaluation_mode",
    "reference_trajectory_source",
    "reference_trajectory_note",
    *CORE_TRAJECTORY_METRICS,
    "trajectory_overall_geometric_mean",
    *SIMULATION_ONLY_COLUMNS,
    *REAL_ONLY_COLUMNS,
    "official_dyneval_status",
    "official_dyneval_error",
    "metric_implementation",
    "metric_reference",
    "metric_version",
    "r_version",
    "dyneval_version",
    "dynwrap_version",
    "dynfeature_version",
    "netdist_version",
    "dyneval_commit",
    "dynwrap_commit",
    "dynfeature_commit",
    "netdist_commit",
    "global_pseudotime_min",
    "global_pseudotime_max",
    "rf_pseudotime_min",
    "rf_pseudotime_max",
    "preprocessing_runtime_seconds",
    "inference_runtime_seconds",
    "pipeline_runtime_seconds",
    "trajectory_metrics_runtime_seconds",
    "pipeline_peak_rss_mb",
    "pipeline_memory_increase_mb",
    "preprocessing_shared_across_repeated_runs",
    "preprocessing_applied",
    "preprocessing_completed_before_inference",
    "preprocessing_profile",
    "preprocessing_step_count",
    "preprocessing_steps_executed",
    "preprocessing_seed",
    "preprocessing_input_shape",
    "preprocessing_output_shape",
    "preprocessing_audit_json",
]


class PeakRSSMonitor:
    """Sample process RSS, including memory allocated by native libraries."""

    def __init__(self, interval=0.05):
        self.interval = float(interval)
        self.process = psutil.Process()
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline_bytes = self.process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        try:
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def peak_rss_mb(self):
        return self.peak_bytes / (1024.0 ** 2)

    @property
    def increase_mb(self):
        return max(0.0, self.peak_bytes - self.baseline_bytes) / (1024.0 ** 2)


def evaluation_mode_from_config(config):
    kind = str(config.get("trajectory_reference_kind", "")).strip()
    if kind == "simulated_tree":
        return "gold_reference_simulation"
    if kind == "annotation_guided" and config.get("annotation_topology_edges"):
        return "curated_reference_real"
    raise ValueError(
        "Each dataset must define either a simulated reference trajectory or "
        "a curated annotation_topology_edges reference."
    )


def reference_values_from_config(adata, cell_types, dataset_config):
    column = dataset_config.get("reference_pseudotime_column")
    if column is not None:
        if column not in adata.obs:
            raise KeyError(f"{column!r} not found in adata.obs")
        return pd.to_numeric(adata.obs[column], errors="coerce").to_numpy(dtype=float)
    depth_map = {
        str(key): float(value)
        for key, value in dataset_config.get("depth_map", {}).items()
    }
    if not depth_map:
        return np.full(len(cell_types), np.nan, dtype=float)
    return np.asarray([depth_map.get(str(label), np.nan) for label in cell_types], dtype=float)


def _safe_correlation(function, predicted, reference, absolute=False):
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(predicted) & np.isfinite(reference)
    if valid.sum() < 3:
        return np.nan
    if np.nanstd(predicted[valid]) == 0 or np.nanstd(reference[valid]) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = function(predicted[valid], reference[valid])
    value = result.statistic if hasattr(result, "statistic") else result[0]
    if not np.isfinite(value):
        return np.nan
    value = float(value)
    return abs(value) if absolute else value


def _ordering_metrics(predicted, reference):
    return {
        "Pearson": _safe_correlation(pearsonr, predicted, reference),
        "Spearman": _safe_correlation(spearmanr, predicted, reference),
        "Kendall": _safe_correlation(kendalltau, predicted, reference),
    }


RF_TEST_FRACTION = 0.20
RF_TRAIN_FRACTION = 1.0 - RF_TEST_FRACTION


def compute_rf_pseudotime_with_validation(
    global_pt,
    coordinates,
    seed=42,
    test_size=RF_TEST_FRACTION,
    return_details=False,
):
    """Validate on held-out path cells, then map pseudotime to all cells.

    Path cells with finite nonnegative global pseudotime are split once into
    80% training and 20% held-out testing cells using ``seed``.  The held-out
    cells are never used to fit the model. MSE and R2 are computed only on
    those testing cells, after which the same model trained on the 80% subset
    predicts pseudotime for every cell. The model is not refitted on the test
    cells before the all-cell mapping step.

    Set ``return_details=True`` to obtain the exact path/train/test indices,
    held-out predictions, realized fractions, and RF parameters for auditing.
    """
    global_pt = np.asarray(global_pt, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if coordinates.shape[0] != global_pt.shape[0]:
        raise ValueError("global_pt and coordinates must contain the same cells")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must lie strictly between 0 and 1")

    path_mask = np.isfinite(global_pt) & (global_pt >= 0)
    path_indices = np.flatnonzero(path_mask)
    details = {
        "status": "insufficient_path_cells",
        "split_seed": int(seed),
        "requested_train_fraction": float(1.0 - float(test_size)),
        "requested_test_fraction": float(test_size),
        "path_indices": path_indices,
        "train_indices": np.asarray([], dtype=int),
        "test_indices": np.asarray([], dtype=int),
        "test_truth": np.asarray([], dtype=float),
        "test_prediction": np.asarray([], dtype=float),
        "n_path_cells": int(path_indices.size),
        "n_train_cells": 0,
        "n_test_cells": 0,
        "realized_train_fraction": np.nan,
        "realized_test_fraction": np.nan,
        "validation_scope": "held_out_path_cells",
        "mapping_training_scope": "training_path_cells_only",
        "model_refit_after_validation": False,
        "model_parameters": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "random_state": int(seed),
            "n_jobs": -1,
        },
    }
    if path_indices.size < 20:
        result = (global_pt, path_mask, np.nan, np.nan)
        return (*result, details) if return_details else result

    train_indices, test_indices = train_test_split(
        path_indices,
        test_size=float(test_size),
        random_state=int(seed),
        shuffle=True,
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(coordinates[train_indices], global_pt[train_indices])
    test_prediction = model.predict(coordinates[test_indices])
    test_truth = global_pt[test_indices]
    mse = float(np.mean((test_prediction - test_truth) ** 2))
    r2 = float(model.score(coordinates[test_indices], test_truth))
    rf_pt = np.clip(model.predict(coordinates), 0.0, 1.0001)
    start_cell = int(path_indices[np.argmin(global_pt[path_indices])])
    rf_pt[start_cell] = 0.0
    details.update({
        "status": "ok",
        "train_indices": np.asarray(train_indices, dtype=int),
        "test_indices": np.asarray(test_indices, dtype=int),
        "test_truth": np.asarray(test_truth, dtype=float),
        "test_prediction": np.asarray(test_prediction, dtype=float),
        "n_train_cells": int(len(train_indices)),
        "n_test_cells": int(len(test_indices)),
        "realized_train_fraction": float(len(train_indices) / len(path_indices)),
        "realized_test_fraction": float(len(test_indices) / len(path_indices)),
    })
    result = (rf_pt, np.ones(len(coordinates), dtype=bool), mse, r2)
    return (*result, details) if return_details else result


def serialize_path(path):
    return json.dumps([int(cell) for cell in path], separators=(",", ":"))


def build_repeated_result_row(
    cycle,
    dijkstra_paths,
    branch_results,
    global_pt,
    global_mask,
    rf_pt,
    rf_mask,
    rf_mse,
    rf_r2,
    start_cell,
    cell_types,
    reference_values,
    rf_validation=None,
    seed=42,
):
    row = {
        "run": int(cycle),
        "random_seed": int(seed),
        "n_paths": int(len(dijkstra_paths)),
        "traj_cells": int(np.asarray(global_mask, dtype=bool).sum()),
        "rf_mse": float(rf_mse),
        "rf_r2": float(rf_r2),
        "start_cell": int(start_cell),
    }
    if rf_validation is not None:
        row.update({
            "rf_path_cells": int(rf_validation.get("n_path_cells", 0)),
            "rf_train_cells": int(rf_validation.get("n_train_cells", 0)),
            "rf_test_cells": int(rf_validation.get("n_test_cells", 0)),
            "rf_train_fraction": float(
                rf_validation.get("realized_train_fraction", np.nan)
            ),
            "rf_test_fraction": float(
                rf_validation.get("realized_test_fraction", np.nan)
            ),
            "rf_split_seed": int(rf_validation.get("split_seed", seed)),
            "rf_validation_scope": str(
                rf_validation.get("validation_scope", "held_out_path_cells")
            ),
            "rf_mapping_training_scope": str(
                rf_validation.get(
                    "mapping_training_scope", "training_path_cells_only"
                )
            ),
            "rf_model_refit_after_validation": bool(
                rf_validation.get("model_refit_after_validation", False)
            ),
        })
    for index, (_, path) in enumerate(dijkstra_paths.items(), start=1):
        row[f"path_{index}"] = serialize_path(path)
    for index, branch in enumerate(branch_results.values(), start=1):
        prefix = f"branch_{index}"
        row[f"{prefix}_paths"] = serialize_path(
            branch.get("path_cells", branch.get("path", []))
        )
        mask = np.asarray(branch["mask"], dtype=bool)
        metrics = _ordering_metrics(
            np.asarray(branch["pseudotime"], dtype=float)[mask],
            np.asarray(reference_values, dtype=float)[mask],
        )
        for name, value in metrics.items():
            row[f"{prefix}_{name}"] = value
    for prefix, values, mask in (
        ("global", global_pt, global_mask),
        ("RF", rf_pt, rf_mask),
    ):
        mask = np.asarray(mask, dtype=bool)
        metrics = _ordering_metrics(
            np.asarray(values, dtype=float)[mask],
            np.asarray(reference_values, dtype=float)[mask],
        )
        for name, value in metrics.items():
            row[f"{prefix}_{name}"] = value
    return row


def ordered_result_columns(max_paths):
    columns = [
        "run", "random_seed", "K", "evaluation_mode",
        "reference_trajectory_source", "n_paths", "traj_cells",
        "rf_mse", "rf_r2", "rf_path_cells", "rf_train_cells",
        "rf_test_cells", "rf_train_fraction", "rf_test_fraction",
        "rf_split_seed", "rf_validation_scope",
        "rf_mapping_training_scope", "rf_model_refit_after_validation",
        "start_cell",
        "preprocessing_runtime_seconds", "inference_runtime_seconds",
        "pipeline_runtime_seconds", "trajectory_metrics_runtime_seconds",
        "pipeline_peak_rss_mb", "pipeline_memory_increase_mb",
    ]
    columns.extend(f"path_{index}" for index in range(1, max_paths + 1))
    for index in range(1, max_paths + 1):
        prefix = f"branch_{index}"
        columns.extend([
            f"{prefix}_paths", f"{prefix}_Pearson",
            f"{prefix}_Spearman", f"{prefix}_Kendall",
        ])
    for prefix in ("global", "RF"):
        columns.extend([
            f"{prefix}_Pearson", f"{prefix}_Spearman", f"{prefix}_Kendall",
        ])
    columns.extend(column for column in TRAJECTORY_METRIC_COLUMNS if column not in columns)
    return columns


def _build_union_tree(paths, coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    graph = nx.Graph()
    nonempty = [list(map(int, path)) for path in paths.values() if len(path) >= 1]
    if not nonempty:
        raise ValueError("No non-empty predicted paths")
    root = int(nonempty[0][0])
    endpoints = {int(path[-1]) for path in nonempty}
    for path in nonempty:
        graph.add_nodes_from(path)
        for left, right in zip(path[:-1], path[1:]):
            weight = max(float(np.linalg.norm(coordinates[left] - coordinates[right])), 1e-12)
            graph.add_edge(int(left), int(right), weight=weight)
    if not nx.is_connected(graph):
        component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(component).copy()
        root = root if root in graph else next(iter(graph.nodes))
        endpoints &= set(graph.nodes)
    if graph.number_of_edges() >= graph.number_of_nodes():
        graph = nx.minimum_spanning_tree(graph, weight="weight")
    milestones = {root, *endpoints}
    milestones.update(node for node, degree in graph.degree() if degree != 2)
    segments, visited = [], set()
    for start in sorted(milestones):
        for neighbour in graph.neighbors(start):
            key = frozenset((start, neighbour))
            if key in visited:
                continue
            nodes = [start, neighbour]
            visited.add(key)
            previous, current = start, neighbour
            while current not in milestones:
                candidates = [node for node in graph.neighbors(current) if node != previous]
                if not candidates:
                    break
                following = candidates[0]
                visited.add(frozenset((current, following)))
                nodes.append(following)
                previous, current = current, following
            cumulative = [0.0]
            for left, right in zip(nodes[:-1], nodes[1:]):
                cumulative.append(cumulative[-1] + float(graph[left][right]["weight"]))
            length = max(cumulative[-1], 1e-12)
            segments.append({
                "u": int(nodes[0]), "v": int(nodes[-1]), "nodes": nodes,
                "length": length,
                "fractions": np.asarray(cumulative, dtype=float) / length,
            })
    if not segments:
        raise ValueError("Predicted path union has no usable segment")
    return root, sorted(milestones), segments


def _projection_from_predicted_paths(paths, coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    root, milestones, segments = _build_union_tree(paths, coordinates)
    duplicated_coords, duplicated_segment, duplicated_fraction = [], [], []
    for index, segment in enumerate(segments):
        duplicated_coords.extend(coordinates[segment["nodes"]])
        duplicated_segment.extend([index] * len(segment["nodes"]))
        duplicated_fraction.extend(segment["fractions"].tolist())
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nearest.fit(np.asarray(duplicated_coords))
    nearest_index = nearest.kneighbors(coordinates, return_distance=False).ravel()
    segment_index = np.asarray(duplicated_segment, dtype=int)[nearest_index]
    fraction = np.asarray(duplicated_fraction, dtype=float)[nearest_index]
    graph = nx.Graph()
    graph.add_nodes_from(milestones)
    for segment in segments:
        graph.add_edge(segment["u"], segment["v"], weight=segment["length"])
    root_distances = nx.single_source_dijkstra_path_length(graph, root, weight="weight")
    scale = max(root_distances.values()) if root_distances else 1.0
    scale = max(float(scale), 1e-12)
    for segment in segments:
        segment["length"] /= scale
    return {
        "graph": graph,
        "milestones": milestones,
        "segments": segments,
        "segment_index": segment_index,
        "fraction": fraction,
    }


def _projection_from_simulation_reference(adata, config):
    branch_column = config.get("reference_branch_column", config.get("label_column"))
    time_column = config.get("reference_pseudotime_column")
    edges = [tuple(map(str, edge)) for edge in config.get("reference_milestone_edges", [])]
    if not branch_column or branch_column not in adata.obs:
        raise KeyError("Simulation reference branch column is missing")
    if not time_column or time_column not in adata.obs or not edges:
        raise KeyError("Simulation reference pseudotime or milestone edges are missing")
    labels = np.asarray(adata.obs[branch_column]).astype(str)
    pseudotime = pd.to_numeric(adata.obs[time_column], errors="coerce").to_numpy(dtype=float)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    global_span = max(float(np.nanmax(pseudotime) - np.nanmin(pseudotime)), 1e-12)
    segments, lookup = [], {}
    for index, (parent, child) in enumerate(edges):
        mask = labels == child
        if not np.any(mask):
            raise ValueError(f"Reference segment {child!r} has no cells")
        low, high = float(np.nanmin(pseudotime[mask])), float(np.nanmax(pseudotime[mask]))
        length = max((high - low) / global_span, 1e-12)
        graph[parent][child]["weight"] = length
        segments.append({"u": parent, "v": child, "length": length, "label": child})
        lookup[child] = index
    assignment = np.asarray([lookup.get(label, -1) for label in labels], dtype=int)
    if np.any(assignment < 0):
        raise ValueError(f"Unmapped simulation labels: {sorted(set(labels[assignment < 0]))}")
    fraction = np.zeros(len(labels), dtype=float)
    for child, index in lookup.items():
        mask = labels == child
        values = pseudotime[mask]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        fraction[mask] = 0.5 if high <= low else (values - low) / (high - low)
    return {
        "graph": graph,
        "milestones": list(graph.nodes),
        "segments": segments,
        "segment_index": assignment,
        "fraction": fraction,
    }


def _projection_from_curated_reference(adata, config):
    """Map annotations to the configured coarse literature-derived tree."""
    label_column = config.get("label_column")
    if not label_column or label_column not in adata.obs:
        raise KeyError("Curated reference requires label_column in adata.obs")
    labels = np.asarray(adata.obs[label_column]).astype(str)
    configured_edges = [
        tuple(map(str, edge)) for edge in config.get("annotation_topology_edges", [])
    ]
    if not configured_edges:
        raise ValueError("Curated reference requires annotation_topology_edges")
    children = {child for _, child in configured_edges}
    roots = sorted({parent for parent, _ in configured_edges} - children)
    early_label = str(config.get("early_label", roots[0] if roots else "__root__"))
    edges = list(configured_edges)
    synthetic_root = "__curated_root__"
    if early_label not in children:
        edges.insert(0, (synthetic_root, early_label))
    depth_map = {str(k): float(v) for k, v in config.get("depth_map", {}).items()}
    graph = nx.Graph()
    segments, lookup = [], {}
    for index, (parent, child) in enumerate(edges):
        parent_depth = depth_map.get(parent, depth_map.get(child, 1.0) - 1.0)
        child_depth = depth_map.get(child, parent_depth + 1.0)
        length = max(abs(child_depth - parent_depth), 1.0)
        graph.add_edge(parent, child, weight=float(length))
        segments.append({"u": parent, "v": child, "length": float(length), "label": child})
        if child in lookup:
            raise ValueError(f"Curated reference is not a tree; multiple parents for {child!r}")
        lookup[child] = index
    assignment = np.asarray([lookup.get(label, -1) for label in labels], dtype=int)
    if np.any(assignment < 0):
        raise ValueError(f"Annotations absent from curated reference: {sorted(set(labels[assignment < 0]))}")
    # An annotation-derived reference only resolves coarse states.  Mapping each
    # cell to its child milestone avoids inventing unsupported within-state time.
    fraction = np.ones(len(labels), dtype=float)
    return {
        "graph": graph,
        "milestones": list(graph.nodes),
        "segments": segments,
        "segment_index": assignment,
        "fraction": fraction,
    }


def _reference_projection(context, config):
    mode = evaluation_mode_from_config(config)
    if mode == "gold_reference_simulation":
        return _projection_from_simulation_reference(context["adata"], config)
    return _projection_from_curated_reference(context["adata"], config)


def _stratified_waypoints(groups, maximum, seed):
    groups = np.asarray(groups)
    rng = np.random.default_rng(int(seed))
    chosen = []
    unique = list(pd.unique(groups))
    per_group = max(1, int(math.ceil(int(maximum) / max(1, len(unique)))))
    for group in unique:
        indices = np.where(groups == group)[0]
        chosen.extend(
            rng.choice(indices, size=min(per_group, len(indices)), replace=False).tolist()
        )
    if len(chosen) > int(maximum):
        chosen = rng.choice(np.asarray(chosen), size=int(maximum), replace=False).tolist()
    return np.asarray(sorted(chosen), dtype=int)


def _cell_geodesic_matrix(projection, cell_indices):
    cell_indices = np.asarray(cell_indices, dtype=int)
    graph = nx.Graph()
    for segment in projection["segments"]:
        graph.add_edge(segment["u"], segment["v"], weight=float(segment["length"]))
    milestone_distance = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    segments = projection["segments"]
    assignment = np.asarray(projection["segment_index"], dtype=int)
    fractions = np.asarray(projection["fraction"], dtype=float)
    result = np.zeros((len(cell_indices), len(cell_indices)), dtype=float)
    for row, left_cell in enumerate(cell_indices):
        left_segment = segments[int(assignment[left_cell])]
        left_fraction = float(fractions[left_cell])
        left_endpoints = [
            (left_segment["u"], left_fraction * left_segment["length"]),
            (left_segment["v"], (1.0 - left_fraction) * left_segment["length"]),
        ]
        for column in range(row + 1, len(cell_indices)):
            right_cell = int(cell_indices[column])
            right_segment = segments[int(assignment[right_cell])]
            right_fraction = float(fractions[right_cell])
            if int(assignment[left_cell]) == int(assignment[right_cell]):
                distance = abs(left_fraction - right_fraction) * left_segment["length"]
            else:
                right_endpoints = [
                    (right_segment["u"], right_fraction * right_segment["length"]),
                    (right_segment["v"], (1.0 - right_fraction) * right_segment["length"]),
                ]
                distance = min(
                    left_offset + milestone_distance[left_node][right_node] + right_offset
                    for left_node, left_offset in left_endpoints
                    for right_node, right_offset in right_endpoints
                )
            result[row, column] = result[column, row] = float(distance)
    return result


def _branch_overlap_similarity(left_assignment, right_assignment):
    left_assignment = np.asarray(left_assignment)
    right_assignment = np.asarray(right_assignment)
    left_groups = list(pd.unique(left_assignment))
    right_groups = list(pd.unique(right_assignment))
    similarities = np.zeros((len(left_groups), len(right_groups)), dtype=float)
    for row, left_group in enumerate(left_groups):
        left_set = set(np.where(left_assignment == left_group)[0])
        for column, right_group in enumerate(right_groups):
            right_set = set(np.where(right_assignment == right_group)[0])
            union = left_set | right_set
            similarities[row, column] = len(left_set & right_set) / len(union) if union else 0.0
    recovery = float(np.mean(similarities.max(axis=1))) if similarities.size else 0.0
    relevance = float(np.mean(similarities.max(axis=0))) if similarities.size else 0.0
    f1 = 0.0 if recovery + relevance <= 0 else 2.0 * recovery * relevance / (recovery + relevance)
    return f1, similarities, left_groups, right_groups


def _segment_adjacency(segments):
    edges = set()
    for left in range(len(segments)):
        endpoints = {segments[left]["u"], segments[left]["v"]}
        for right in range(left + 1, len(segments)):
            if endpoints & {segments[right]["u"], segments[right]["v"]}:
                edges.add((left, right))
    return edges


def _spectral_density(adjacency, grid, gamma=0.1):
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    frequencies = np.sqrt(np.clip(eigenvalues[1:], 0.0, None))
    if not len(frequencies):
        return np.zeros_like(grid)
    density = np.sum(
        gamma / ((grid[:, None] - frequencies[None, :]) ** 2 + gamma ** 2),
        axis=1,
    )
    area = np.trapz(density, grid)
    return density / area if area > 0 else density


def _him_similarity(node_labels, left_edges, right_edges, gamma=0.1):
    count = len(node_labels)
    if count < 2:
        return 1.0
    index = {label: position for position, label in enumerate(node_labels)}
    def adjacency(edges):
        matrix = np.zeros((count, count), dtype=float)
        for left, right in edges:
            if left in index and right in index and left != right:
                matrix[index[left], index[right]] = 1.0
                matrix[index[right], index[left]] = 1.0
        return matrix
    left_matrix, right_matrix = adjacency(left_edges), adjacency(right_edges)
    hamming = float(np.sum(np.abs(left_matrix - right_matrix)) / (count * (count - 1)))
    grid = np.linspace(0.0, math.sqrt(max(2, count)) + 4.0, 2000)
    left_density = _spectral_density(left_matrix, grid, gamma)
    right_density = _spectral_density(right_matrix, grid, gamma)
    im_distance = math.sqrt(float(np.trapz((left_density - right_density) ** 2, grid)))
    empty = np.zeros_like(left_matrix)
    complete = np.ones_like(left_matrix) - np.eye(count)
    normaliser = math.sqrt(float(np.trapz(
        (_spectral_density(empty, grid, gamma) - _spectral_density(complete, grid, gamma)) ** 2,
        grid,
    )))
    im_normalised = min(1.0, im_distance / normaliser) if normaliser > 0 else 0.0
    distance = math.sqrt(hamming ** 2 + im_normalised ** 2) / math.sqrt(2.0)
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def trajectory_pairwise_similarity(left, right, maximum_waypoints=120, seed=42):
    """Saelens-style stability components between two inferred models."""
    if len(left["segment_index"]) != len(right["segment_index"]):
        raise ValueError("Pairwise stability requires the same cells in both runs")
    waypoints = _stratified_waypoints(left["segment_index"], maximum_waypoints, seed)
    left_distance = _cell_geodesic_matrix(left, waypoints)
    right_distance = _cell_geodesic_matrix(right, waypoints)
    upper = np.triu_indices(len(waypoints), 1)
    if len(upper[0]) < 3 or np.std(left_distance[upper]) == 0 or np.std(right_distance[upper]) == 0:
        cor_dist = 0.0
    else:
        value = spearmanr(left_distance[upper], right_distance[upper]).correlation
        cor_dist = max(0.0, float(value)) if np.isfinite(value) else 0.0
    branch_f1, similarities, left_groups, right_groups = _branch_overlap_similarity(
        left["segment_index"], right["segment_index"]
    )
    left_labels = list(range(len(left_groups)))
    left_group_map = {int(group): index for index, group in enumerate(left_groups)}
    right_to_left = {
        column: int(np.argmax(similarities[:, column]))
        for column in range(similarities.shape[1])
    }
    left_edges = {
        (left_group_map[left_index], left_group_map[right_index])
        for left_index, right_index in _segment_adjacency(left["segments"])
        if left_index in left_group_map and right_index in left_group_map
    }
    right_edges = set()
    for left_index, right_index in _segment_adjacency(right["segments"]):
        mapped_left = right_to_left.get(left_index)
        mapped_right = right_to_left.get(right_index)
        if mapped_left is not None and mapped_right is not None and mapped_left != mapped_right:
            right_edges.add((mapped_left, mapped_right))
    him = _him_similarity(left_labels, left_edges, right_edges)
    overall = float(np.prod(np.clip([cor_dist, branch_f1, him], 0.0, 1.0)) ** (1.0 / 3.0))
    return {
        "pairwise_cor_dist": cor_dist,
        "pairwise_F1_branches": branch_f1,
        "pairwise_HIM_similarity": him,
        "pairwise_trajectory_similarity": overall,
        "pairwise_waypoints": int(len(waypoints)),
    }


def _write_dynwrap_projection(directory, prefix, projection, cell_ids):
    network_path = directory / f"{prefix}_network.csv"
    pd.DataFrame([
        {
            "from": str(segment["u"]),
            "to": str(segment["v"]),
            "length": max(float(segment["length"]), 1e-12),
            "directed": True,
        }
        for segment in projection["segments"]
    ]).to_csv(network_path, index=False, encoding="utf-8")
    progression_path = directory / f"{prefix}_progressions.csv"
    rows = []
    for cell, cell_id in enumerate(cell_ids):
        segment = projection["segments"][int(projection["segment_index"][cell])]
        rows.append({
            "cell_id": str(cell_id),
            "from": str(segment["u"]),
            "to": str(segment["v"]),
            "percentage": float(np.clip(projection["fraction"][cell], 0.0, 1.0)),
        })
    pd.DataFrame(rows).to_csv(progression_path, index=False, encoding="utf-8")
    return network_path, progression_path


def _official_dyneval_paths():
    # The original metric module ran from the project notebook and therefore
    # obtained PROJECT_ROOT from notebook globals.  In the modular repository,
    # resolve the same unmodified manifest/bridge from either the explicit
    # project-root environment or the repository's audited parent project.
    candidates = []
    environment_root = os.environ.get("SCCGRL_PROJECT_ROOT")
    if environment_root:
        candidates.append(Path(environment_root).expanduser().resolve())
    configured_root = globals().get("PROJECT_ROOT")
    if configured_root is not None:
        candidates.append(Path(configured_root).expanduser().resolve())
    module_path = Path(__file__).resolve()
    if len(module_path.parents) > 4:
        candidates.append(module_path.parents[4])
    candidates.append(Path.cwd().resolve())

    checked = []
    manifest_path = bridge_path = None
    for root in dict.fromkeys(candidates):
        candidate_manifest = (
            root / "response" / "1" /
            "official_dyneval_environment_manifest_v1.0.json"
        )
        candidate_bridge = (
            root / "response" / "1" / "official_dyneval_metrics_v1.0.R"
        )
        checked.append(str(candidate_manifest))
        if candidate_manifest.exists() and candidate_bridge.exists():
            manifest_path, bridge_path = candidate_manifest, candidate_bridge
            break
    if manifest_path is None or bridge_path is None:
        raise FileNotFoundError(
            "Official dyneval manifest or R bridge is missing; checked: "
            + "; ".join(checked)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rscript = Path(manifest["rscript"])
    r_library = Path(manifest["r_library"])
    if not rscript.exists() or not r_library.exists():
        raise FileNotFoundError(f"Invalid R environment: {rscript}; {r_library}")
    return manifest, rscript, r_library, bridge_path


def _expression_matrix(adata):
    matrix = adata.X
    if sp.issparse(matrix):
        return matrix.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(matrix, dtype=np.float64))


def _official_dyneval_metrics(context, reference, prediction, config, seed):
    manifest, rscript, r_library, bridge_path = _official_dyneval_paths()
    expression = context.get("_trajectory_metric_expression_v2")
    if expression is None:
        expression = _expression_matrix(context["adata"])
        context["_trajectory_metric_expression_v2"] = expression
    cell_ids = [f"cell_{index}" for index in range(expression.shape[0])]
    feature_ids = [f"feature_{index}" for index in range(expression.shape[1])]
    with tempfile.TemporaryDirectory(prefix="sccgrl_dyneval_v2_") as temporary:
        directory = Path(temporary)
        expression_path = directory / "expression.mtx"
        cell_ids_path = directory / "cell_ids.txt"
        feature_ids_path = directory / "feature_ids.txt"
        mmwrite(expression_path, expression)
        cell_ids_path.write_text("\n".join(cell_ids) + "\n", encoding="utf-8")
        feature_ids_path.write_text("\n".join(feature_ids) + "\n", encoding="utf-8")
        true_network, true_progressions = _write_dynwrap_projection(
            directory, "reference", reference, cell_ids
        )
        pred_network, pred_progressions = _write_dynwrap_projection(
            directory, "prediction", prediction, cell_ids
        )
        output_path = directory / "metrics.json"
        spec_path = directory / "spec.json"
        spec = {
            "random_seed": int(seed),
            "waypoint_count": int(config.get("cor_dist_waypoints", 120)),
            "cell_ids_file": str(cell_ids_path),
            "feature_ids_file": str(feature_ids_path),
            "expression_matrix_file": str(expression_path),
            "true_network_file": str(true_network),
            "predicted_network_file": str(pred_network),
            "true_progressions_file": str(true_progressions),
            "predicted_progressions_file": str(pred_progressions),
        }
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        environment = os.environ.copy()
        environment["R_LIBS_USER"] = str(r_library)
        completed = subprocess.run(
            [str(rscript), str(bridge_path), str(spec_path), str(output_path)],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(config.get("official_dyneval_timeout_seconds", 1800)),
        )
        if completed.returncode != 0 or not output_path.exists():
            message = (completed.stderr or completed.stdout or "unknown R error").strip()
            raise RuntimeError(f"Official dyneval failed: {message}")
        result = json.loads(output_path.read_text(encoding="utf-8"))
    commits = manifest.get("pinned_git_commits", {})
    result.update({
        "official_dyneval_status": "success",
        "official_dyneval_error": "",
        "dyneval_commit": commits.get("dyneval", ""),
        "dynwrap_commit": commits.get("dynwrap", ""),
        "dynfeature_commit": commits.get("dynfeature", ""),
        "netdist_commit": commits.get("netdist", ""),
    })
    return result


def _simulation_pseudotime_metrics(context, pseudotime, config):
    column = config.get("reference_pseudotime_column")
    if not column or column not in context["adata"].obs:
        raise KeyError("Simulation requires reference_pseudotime_column")
    truth = pd.to_numeric(context["adata"].obs[column], errors="coerce").to_numpy(dtype=float)
    prediction = np.asarray(pseudotime["rf_pseudotime"], dtype=float)
    scores = _ordering_metrics(prediction, truth)
    return {f"reference_pseudotime_{name}": value for name, value in scores.items()}


def _marker_metrics(context, pseudotime, config):
    configured = [str(gene) for gene in config.get("marker_genes", [])]
    result = {
        "marker_gene_mean_abs_spearman": np.nan,
        "marker_gene_median_abs_spearman": np.nan,
        "marker_genes_configured": len(configured),
        "marker_genes_available": 0,
        "marker_gene_correlations_json": "{}",
    }
    if not configured:
        return result
    adata = context["adata"]
    exact = {str(name): str(name) for name in adata.var_names}
    folded = {str(name).casefold(): str(name) for name in adata.var_names}
    available = []
    for gene in configured:
        matched = exact.get(gene, folded.get(gene.casefold()))
        if matched is not None and matched not in available:
            available.append(matched)
    result["marker_genes_available"] = len(available)
    if not available:
        return result
    matrix = adata[:, available].X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    prediction = np.asarray(pseudotime["rf_pseudotime"], dtype=float)
    correlations = {
        gene: _safe_correlation(spearmanr, matrix[:, index], prediction)
        for index, gene in enumerate(available)
    }
    finite = np.asarray([
        abs(value) for value in correlations.values() if np.isfinite(value)
    ], dtype=float)
    if finite.size:
        result["marker_gene_mean_abs_spearman"] = float(np.mean(finite))
        result["marker_gene_median_abs_spearman"] = float(np.median(finite))
    result["marker_gene_correlations_json"] = json.dumps(
        {gene: (float(value) if np.isfinite(value) else None)
         for gene, value in correlations.items()},
        ensure_ascii=False,
        sort_keys=True,
    )
    return result


def _finite_range(values, mask=None):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.nan, np.nan
    return float(values[valid].min()), float(values[valid].max())


def pseudotime_range_metrics(global_pt, global_mask, rf_pt, rf_mask):
    global_min, global_max = _finite_range(global_pt, global_mask)
    rf_min, rf_max = _finite_range(rf_pt, rf_mask)
    return {
        "global_pseudotime_min": global_min,
        "global_pseudotime_max": global_max,
        "rf_pseudotime_min": rf_min,
        "rf_pseudotime_max": rf_max,
    }


def metric_provenance(seed):
    return {
        "random_seed": int(seed),
        "metric_version": METRIC_VERSION,
        "metric_reference": SAELENS_REFERENCE,
        "metric_implementation": "official R dyneval on exported dynwrap trajectories",
    }


def compute_trajectory_benchmark_metrics(context, paths, pseudotime, config, seed=42):
    mode = evaluation_mode_from_config(config)
    reference = _reference_projection(context, config)
    prediction = _projection_from_predicted_paths(paths, context["model_coords"])
    metrics = _official_dyneval_metrics(context, reference, prediction, config, seed)
    metrics.update(metric_provenance(seed))
    metrics.update({
        "evaluation_mode": mode,
        "reference_trajectory_source": (
            "stored_simulation_generating_trajectory"
            if mode == "gold_reference_simulation"
            else "curated_annotation_literature_lineage_graph"
        ),
        "reference_trajectory_note": (
            "Simulation scores compare against the stored generating trajectory."
            if mode == "gold_reference_simulation"
            else "Real-data scores measure agreement with the configured coarse curated reference; they are not absolute biological ground truth."
        ),
    })
    if mode == "gold_reference_simulation":
        metrics.update(_simulation_pseudotime_metrics(context, pseudotime, config))
    else:
        metrics.update(_marker_metrics(context, pseudotime, config))
    metrics.update(pseudotime_range_metrics(
        pseudotime["global_pseudotime"], pseudotime["global_mask"],
        pseudotime["rf_pseudotime"], pseudotime["rf_mask"],
    ))
    audit = dict(context.get("preprocessing_audit", {}))
    records = list(audit.get("steps", []))
    metrics.update({
        "preprocessing_applied": bool(records),
        "preprocessing_completed_before_inference": bool(
            audit.get("completed_before_inference", False)
        ),
        "preprocessing_profile": audit.get("profile", "dataset_configured"),
        "preprocessing_step_count": len(records),
        "preprocessing_steps_executed": "|".join(
            str(record.get("operation", "")) for record in records
        ),
        "preprocessing_seed": int(audit.get("random_seed", seed)),
        "preprocessing_input_shape": "x".join(map(str, audit.get("input_shape", []))),
        "preprocessing_output_shape": "x".join(map(str, audit.get("output_shape", []))),
        "preprocessing_audit_json": json.dumps(audit, ensure_ascii=False, sort_keys=True),
    })
    return metrics


def add_resource_metrics(
    metrics,
    preprocessing_seconds,
    inference_seconds,
    peak_rss_mb,
    memory_increase_mb,
    metric_seconds,
    preprocessing_shared=False,
):
    metrics.update({
        "preprocessing_runtime_seconds": float(preprocessing_seconds),
        "inference_runtime_seconds": float(inference_seconds),
        "pipeline_runtime_seconds": float(preprocessing_seconds + inference_seconds),
        "trajectory_metrics_runtime_seconds": float(metric_seconds),
        "pipeline_peak_rss_mb": float(peak_rss_mb),
        "pipeline_memory_increase_mb": float(memory_increase_mb),
        "preprocessing_shared_across_repeated_runs": bool(preprocessing_shared),
    })
    return metrics


def filter_metric_frame_for_mode(frame, mode):
    frame = frame.copy()
    if mode == "gold_reference_simulation":
        return frame.drop(columns=list(REAL_ONLY_COLUMNS), errors="ignore")
    if mode == "curated_reference_real":
        return frame.drop(columns=list(SIMULATION_ONLY_COLUMNS), errors="ignore")
    raise ValueError(f"Unknown evaluation mode: {mode}")
