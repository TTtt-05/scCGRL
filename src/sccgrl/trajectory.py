# Canonical source notebook: 2026-08-17_scCGRL_five_datasets_v1.ipynb
# Notebook date/version: 2026-08-17 / CODE_REVISION 1.6
# Core source cell: index 17 / order 18; all override definitions retained in original order
# Final single-run orchestration: index 21 / order 22, final v2.0 helper chain
"""Single-run and repeated-run orchestration for scCGRL."""

from __future__ import annotations


import random
import time

from .preprocessing import load_prepared_dataset
from .graph_endpoints import find_start_and_endpoints
from .q_learning import QLearningPathFinder
from .pseudotime_mapping import (
    compute_branch_pseudotimes,
    compute_graph_distance_global_pseudotime,
)
from .rf_mapping import compute_rf_pseudotime_with_validation
from .metrics import (
    PeakRSSMonitor,
    add_resource_metrics,
    build_repeated_result_row,
    compute_trajectory_benchmark_metrics,
    evaluation_mode_from_config,
    filter_metric_frame_for_mode,
    metric_provenance,
    ordered_result_columns,
    reference_values_from_config,
)
from .io_utils import save_reference_figures

from importlib import metadata
from pathlib import Path
import json
import pickle
import platform
import numpy as np
import pandas as pd



CODE_REVISION = "1.7"


def software_versions():
    packages = [
        "scanpy", "anndata", "numpy", "pandas", "scipy", "scikit-learn",
        "matplotlib", "seaborn", "adjustText", "networkx",
    ]
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def prepare_model_context(input_path, dataset_config, seed=42):
    context = load_prepared_dataset(input_path, dataset_config, seed)
    np.random.seed(int(seed))
    context["endpoint_result"] = find_start_and_endpoints(
        context["model_coords"], context["labels"], early_cell_label=context["early_label"]
    )
    return context


def train_one_q_learning(context, episodes=10000, seed=42, verbose=True):
    set_reproducible_seed(seed)
    endpoints = context["endpoint_result"]
    learner = QLearningPathFinder(
        adj_matrix=endpoints["knn_adj"],
        coords=context["model_coords"],
        start_idx=endpoints["start_index"],
        end_indices=endpoints["endpoint_indices"],
        epsilon=0.9,
        alpha=0.1,
        gamma=0.9,
        n_episodes=int(episodes),
    )
    learner.train(verbose=verbose)
    paths = learner.find_shortest_paths(learner.build_sparse_graph())
    return learner, paths


def compute_pseudotime_outputs(context, paths, seed=42):
    n_cells = len(context["adata"])
    branches = compute_branch_pseudotimes(paths, n_cells)
    global_pt, global_mask, start_cell = compute_graph_distance_global_pseudotime(
        paths,
        context["model_coords"],
        n_cells,
        context["endpoint_result"]["knn_graph"],
    )
    rf_pt, rf_mask, rf_mse, rf_r2, rf_validation = (
        compute_rf_pseudotime_with_validation(
            global_pt,
            context["model_coords"],
            seed=seed,
            return_details=True,
        )
    )
    return {
        "branch_results": branches,
        "global_pseudotime": global_pt,
        "global_mask": global_mask,
        "rf_pseudotime": rf_pt,
        "rf_mask": rf_mask,
        "rf_mse": rf_mse,
        "rf_r2": rf_r2,
        "rf_validation": rf_validation,
        "start_cell": start_cell,
    }


def run_single_experiment(
    input_path,
    output_dir,
    dataset_config,
    episodes=10000,
    seed=42,
    save_processed=True,
):
    """Run the final notebook single-dataset workflow once.

    This follows the final execution helper in source-notebook cell index 21:
    preprocessing and inference are resource-monitored, RF validation uses the
    held-out path-cell implementation, trajectory metric v2.0 is evaluated, and
    the same reference figures are exported. Metric time is excluded from the
    reported pipeline runtime.
    """

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_started = time.perf_counter()
    with PeakRSSMonitor() as pipeline_memory:
        preprocessing_started = time.perf_counter()
        context = prepare_model_context(input_path, dataset_config, seed)
        preprocessing_seconds = time.perf_counter() - preprocessing_started

        inference_started = time.perf_counter()
        learner, paths = train_one_q_learning(
            context, episodes=episodes, seed=seed, verbose=True
        )
        branch_results = compute_branch_pseudotimes(paths, len(context["adata"]))
        global_pt, global_mask, start_cell = compute_graph_distance_global_pseudotime(
            paths,
            context["model_coords"],
            len(context["adata"]),
            context["endpoint_result"]["knn_graph"],
        )
        rf_pt, rf_mask, rf_mse, rf_r2, rf_validation = (
            compute_rf_pseudotime_with_validation(
                global_pt,
                context["model_coords"],
                seed=seed,
                return_details=True,
            )
        )
        inference_seconds = time.perf_counter() - inference_started
    pipeline_seconds = time.perf_counter() - pipeline_started

    pseudotime = {
        "branch_results": branch_results,
        "global_pseudotime": global_pt,
        "global_mask": global_mask,
        "rf_pseudotime": rf_pt,
        "rf_mask": rf_mask,
        "start_cell": start_cell,
    }
    adata = context["adata"]
    labels = context["labels"]
    adata.obs["QLearning_pseudotime"] = rf_pt

    reference_values = reference_values_from_config(adata, labels, dataset_config)
    metric_row = build_repeated_result_row(
        cycle=1,
        dijkstra_paths=paths,
        branch_results=branch_results,
        global_pt=global_pt,
        global_mask=global_mask,
        rf_pt=rf_pt,
        rf_mask=rf_mask,
        rf_mse=rf_mse,
        rf_r2=rf_r2,
        start_cell=start_cell,
        cell_types=labels,
        reference_values=reference_values,
        rf_validation=rf_validation,
        seed=seed,
    )
    metric_started = time.perf_counter()
    extra_metrics = compute_trajectory_benchmark_metrics(
        context, paths, pseudotime, dataset_config, seed=seed
    )
    metric_seconds = time.perf_counter() - metric_started
    add_resource_metrics(
        extra_metrics,
        preprocessing_seconds,
        inference_seconds,
        pipeline_memory.peak_rss_mb,
        pipeline_memory.increase_mb,
        metric_seconds,
        preprocessing_shared=False,
    )
    extra_metrics["pipeline_runtime_seconds"] = float(pipeline_seconds)
    metric_row.update(extra_metrics)
    metric_row["K"] = int(context["endpoint_result"]["k_value"])
    mode = evaluation_mode_from_config(dataset_config)
    metric_frame = filter_metric_frame_for_mode(pd.DataFrame([metric_row]), mode)
    metrics_path = output_dir / f"{dataset_config['key']}_single_run_metrics.csv"
    metric_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    table_data = {
        "cell_id": adata.obs_names,
        "random_seed": int(seed),
        "K": int(context["endpoint_result"]["k_value"]),
        "cell_type": labels,
        "global_path_pseudotime": global_pt,
        "QLearning_pseudotime": rf_pt,
    }
    rf_role = np.full(len(adata), "mapped_non_path", dtype=object)
    rf_role[np.asarray(rf_validation["train_indices"], dtype=int)] = "train_path"
    rf_role[np.asarray(rf_validation["test_indices"], dtype=int)] = "test_path"
    table_data["rf_mapping_role"] = rf_role
    reference_column = dataset_config.get("reference_pseudotime_column")
    if reference_column is not None:
        table_data[reference_column] = np.asarray(adata.obs[reference_column])
    reference_branch_column = dataset_config.get("reference_branch_column")
    if reference_branch_column and reference_branch_column in adata.obs:
        table_data[reference_branch_column] = np.asarray(
            adata.obs[reference_branch_column]
        )
    table_path = output_dir / f"{dataset_config['key']}_pseudotime.csv"
    pd.DataFrame(table_data).to_csv(table_path, index=False, encoding="utf-8-sig")

    heldout_prediction = {
        int(index): float(prediction)
        for index, prediction in zip(
            rf_validation["test_indices"],
            rf_validation["test_prediction"],
        )
    }
    split_path = output_dir / f"{dataset_config['key']}_rf_path_cell_split.csv"
    split_rows = []
    train_set = set(map(int, rf_validation["train_indices"]))
    test_set = set(map(int, rf_validation["test_indices"]))
    for index in map(int, rf_validation["path_indices"]):
        split_rows.append({
            "cell_index": index,
            "cell_id": str(adata.obs_names[index]),
            "split": "train" if index in train_set else "test",
            "global_path_pseudotime": float(global_pt[index]),
            "heldout_test_prediction": heldout_prediction.get(index, np.nan),
            "heldout_squared_error": (
                (heldout_prediction[index] - float(global_pt[index])) ** 2
                if index in test_set else np.nan
            ),
            "final_all_cell_mapping": float(rf_pt[index]),
            "random_seed": int(seed),
        })
    pd.DataFrame(split_rows).to_csv(
        split_path, index=False, encoding="utf-8-sig"
    )

    figures = save_reference_figures(
        adata,
        learner,
        context["endpoint_result"],
        paths,
        pseudotime,
        context["plot_coords"],
        dataset_config,
        output_dir,
    )
    reproducibility = {
        "code_version": CODE_REVISION,
        "entry_script": "sccgrl_main_v3_0.py",
        "input_path": str(context["input_path"]),
        "dataset_config": dataset_config,
        "episodes": int(episodes),
        "seed": int(seed),
        "rf_validation": {
            "design": "80% path-cell training / 20% held-out path-cell testing",
            "validation_scope": "held_out_path_cells",
            "mapping_scope": "all_cells",
            "model_refit_after_validation": False,
            "n_path_cells": int(rf_validation["n_path_cells"]),
            "n_train_cells": int(rf_validation["n_train_cells"]),
            "n_test_cells": int(rf_validation["n_test_cells"]),
            "realized_train_fraction": float(
                rf_validation["realized_train_fraction"]
            ),
            "realized_test_fraction": float(
                rf_validation["realized_test_fraction"]
            ),
            "split_seed": int(rf_validation["split_seed"]),
            "model_parameters": rf_validation["model_parameters"],
        },
        "software_versions": software_versions(),
    }
    with figures["report"].open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + "=" * 50 + "\nReproducibility manifest\n" + "=" * 50 + "\n")
        json.dump(reproducibility, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    state_path = output_dir / "scCGRL_model_state.pkl"
    state = {
        "code_revision": CODE_REVISION,
        "config": dataset_config,
        "start_index": int(context["endpoint_result"]["start_index"]),
        "endpoint_indices": [int(value) for value in context["endpoint_result"]["endpoint_indices"]],
        "dijkstra_paths": {int(key): [int(value) for value in path] for key, path in paths.items()},
        "Q": {int(state): {int(action): float(value) for action, value in actions.items()}
              for state, actions in learner.Q.items()},
        "rf_validation": {
            "split_seed": int(rf_validation["split_seed"]),
            "path_indices": list(map(int, rf_validation["path_indices"])),
            "train_indices": list(map(int, rf_validation["train_indices"])),
            "test_indices": list(map(int, rf_validation["test_indices"])),
            "test_truth": list(map(float, rf_validation["test_truth"])),
            "test_prediction": list(map(float, rf_validation["test_prediction"])),
            "model_parameters": rf_validation["model_parameters"],
            "model_refit_after_validation": False,
        },
        "reproducibility": reproducibility,
    }
    with state_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    processed_path = None
    if save_processed:
        processed_path = output_dir / "scCGRL_processed.h5ad"
        preprocessing_audit = dict(
            adata.uns.get("sccgrl_preprocessing_audit", {})
        )
        original_audit = adata.uns.get("sccgrl_preprocessing_audit")
        if isinstance(preprocessing_audit.get("steps"), list):
            preprocessing_audit["steps_json"] = json.dumps(
                preprocessing_audit.pop("steps"),
                ensure_ascii=False,
                sort_keys=True,
            )
            adata.uns["sccgrl_preprocessing_audit"] = preprocessing_audit
        try:
            adata.write_h5ad(processed_path, compression="gzip")
        finally:
            if original_audit is not None:
                adata.uns["sccgrl_preprocessing_audit"] = original_audit
    return {
        "dataset_key": dataset_config["key"],
        "config": dataset_config,
        "context": context,
        "adata": adata,
        "q_learner": learner,
        "dijkstra_paths": paths,
        "result": context["endpoint_result"],
        "results": pseudotime,
        "figures": figures["figures"],
        "report": figures["report"],
        "pseudotime_table": table_path,
        "rf_path_cell_split": split_path,
        "metrics_table": metrics_path,
        "metric_row": metric_row,
        "model_state": state_path,
        "processed_h5ad": processed_path,
    }


def run_repeated_experiments(input_path, output_dir, dataset_config, runs=100, episodes=10000, seed=42):
    """Repeated evaluation without plotting; one row per run plus resource fields."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_started = time.perf_counter()
    with PeakRSSMonitor() as preprocessing_memory:
        context = prepare_model_context(input_path, dataset_config, seed)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    initial_rss_bytes = preprocessing_memory.baseline_bytes

    adata, labels = context["adata"], context["labels"]
    reference = reference_values_from_config(adata, labels, dataset_config)
    rows = []
    maximum_paths = max(2, len(dataset_config.get("target_lineages", [])))

    for cycle in range(1, int(runs) + 1):
        cycle_seed = int(seed) + cycle - 1
        inference_started = time.perf_counter()
        with PeakRSSMonitor() as inference_memory:
            learner, paths = train_one_q_learning(
                context, episodes=episodes, seed=cycle_seed, verbose=False
            )
            if paths:
                branches = compute_branch_pseudotimes(paths, len(adata))
                global_pt, global_mask, start_cell = compute_graph_distance_global_pseudotime(
                    paths,
                    context["model_coords"],
                    len(adata),
                    context["endpoint_result"]["knn_graph"],
                )
            else:
                branches, global_pt, global_mask, start_cell = {}, None, None, -1
            if global_pt is not None:
                rf_pt, rf_mask, rf_mse, rf_r2, rf_validation = (
                    compute_rf_pseudotime_with_validation(
                        global_pt,
                        context["model_coords"],
                        seed=cycle_seed,
                        return_details=True,
                    )
                )
            else:
                rf_pt, rf_mask, rf_mse, rf_r2, rf_validation = (
                    None, None, np.nan, np.nan, None
                )
        inference_seconds = time.perf_counter() - inference_started

        peak_bytes = max(preprocessing_memory.peak_bytes, inference_memory.peak_bytes)
        peak_rss_mb = peak_bytes / (1024.0 ** 2)
        memory_increase_mb = max(0.0, peak_bytes - initial_rss_bytes) / (1024.0 ** 2)

        if not paths or global_pt is None:
            row = {
                "run": cycle,
                "K": int(context["endpoint_result"]["k_value"]),
                "n_paths": 0,
                "trajectory_metric_scope": "not_available_failed_run",
                "trajectory_metric_note": "No usable path/global pseudotime was produced in this run.",
                "official_dyneval_status": "not_available_failed_run",
                "official_dyneval_error": "No usable path/global pseudotime was produced in this run.",
                **metric_provenance(cycle_seed),
            }
            add_resource_metrics(
                row, preprocessing_seconds, inference_seconds,
                peak_rss_mb, memory_increase_mb, 0.0,
                preprocessing_shared=True,
            )
            rows.append(row)
            continue

        pseudotime = {
            "branch_results": branches,
            "global_pseudotime": global_pt,
            "global_mask": global_mask,
            "rf_pseudotime": rf_pt,
            "rf_mask": rf_mask,
            "start_cell": start_cell,
        }
        metric_started = time.perf_counter()
        extra_metrics = compute_trajectory_benchmark_metrics(
            context, paths, pseudotime, dataset_config, seed=cycle_seed
        )
        metric_seconds = time.perf_counter() - metric_started
        add_resource_metrics(
            extra_metrics, preprocessing_seconds, inference_seconds,
            peak_rss_mb, memory_increase_mb, metric_seconds,
            preprocessing_shared=True,
        )
        row = build_repeated_result_row(
            cycle=cycle,
            dijkstra_paths=paths,
            branch_results=branches,
            global_pt=global_pt,
            global_mask=global_mask,
            rf_pt=rf_pt,
            rf_mask=rf_mask,
            rf_mse=rf_mse,
            rf_r2=rf_r2,
            start_cell=start_cell,
            cell_types=labels,
            reference_values=reference,
            rf_validation=rf_validation,
            seed=cycle_seed,
        )
        row.update(extra_metrics)
        row["K"] = int(context["endpoint_result"]["k_value"])
        rows.append(row)
        maximum_paths = max(maximum_paths, len(paths))
        if cycle % 10 == 0 or cycle == int(runs):
            print(f"[{cycle}/{runs}] repeated evaluation completed")

    columns = ordered_result_columns(maximum_paths)
    frame = pd.DataFrame(rows).reindex(columns=columns)
    frame = filter_metric_frame_for_mode(frame, evaluation_mode_from_config(dataset_config))
    csv_path = output_dir / f"{dataset_config['key']}_repeated_{int(runs)}_runs.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def set_reproducible_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return {"numpy": seed, "python_random": seed, "torch": "not_installed"}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"numpy": seed, "python_random": seed, "torch": seed}
