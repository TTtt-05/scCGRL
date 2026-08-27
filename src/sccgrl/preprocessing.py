# Canonical source notebook: 2026-08-17_scCGRL_five_datasets_v1.ipynb
# Notebook date/version: 2026-08-17 / CODE_REVISION 1.6
# Source cell: index 5 / order 6
"""Configuration-driven data loading and preprocessing for scCGRL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse


def match_label(labels, requested):
    unique = list(dict.fromkeys(np.asarray(labels).tolist()))
    if requested is None:
        raise ValueError(
            f"early_label must be configured; available={list(map(str, unique))}"
        )
    for value in unique:
        if value == requested or str(value) == str(requested):
            return value
    raise ValueError(
        f"Early label {requested!r} not found; available={list(map(str, unique))}"
    )


def select_dimensions(matrix, dimensions, name):
    dimensions = tuple(int(index) for index in dimensions)
    if (
        len(dimensions) < 2
        or any(index < 0 or index >= matrix.shape[1] for index in dimensions)
    ):
        raise ValueError(f"{name} dimensions {dimensions} invalid for shape {matrix.shape}")
    return np.asarray(matrix)[:, dimensions]


def get_matrix_min_max(adata):
    """Return finite min/max values without densifying a sparse expression matrix."""
    if adata.n_obs == 0 or adata.n_vars == 0:
        return float("nan"), float("nan")
    if sparse.issparse(adata.X):
        if adata.X.nnz == 0:
            return 0.0, 0.0
        data = np.asarray(adata.X.data)
        return float(np.nanmin(data)), float(np.nanmax(data))
    matrix = np.asarray(adata.X)
    return float(np.nanmin(matrix)), float(np.nanmax(matrix))


def apply_preprocessing_steps(adata, dataset_config, seed=42):
    """Apply the complete dataset config before downstream trajectory inference."""
    steps = [dict(step) for step in dataset_config.get("preprocessing_steps", [])]
    input_shape = [int(adata.n_obs), int(adata.n_vars)]
    input_min, input_max = get_matrix_min_max(adata)
    records = []

    for step_index, step in enumerate(steps, start=1):
        operation = str(step.get("operation", ""))
        if not operation:
            raise ValueError(f"Preprocessing step {step_index} lacks operation: {step}")

        parameters = {key: value for key, value in step.items() if key != "operation"}
        status = "applied"
        print(f"[Preprocessing {step_index}/{len(steps)}] {operation}")

        if operation == "reuse_existing_representations":
            required = list(step.get("required_obsm", ["X_pca", "X_umap"]))
            missing = [key for key in required if key not in adata.obsm]
            if missing:
                raise KeyError(f"Missing stored representations: {missing}")
            expected_umap_dimensions = step.get("expected_umap_dimensions")
            if expected_umap_dimensions is not None:
                actual_umap_dimensions = int(np.asarray(adata.obsm["X_umap"]).shape[1])
                if actual_umap_dimensions != int(expected_umap_dimensions):
                    raise ValueError(
                        "Stored X_umap dimension mismatch: "
                        f"expected={expected_umap_dimensions}, actual={actual_umap_dimensions}"
                    )
            status = "reused_reference_preprocessing_output"

        elif operation == "exclude_labels":
            column = step.get("label_column", dataset_config.get("label_column"))
            if column is None or column not in adata.obs:
                raise KeyError(f"exclude_labels cannot find label column {column!r}")
            excluded = {str(value) for value in step.get("values", [])}
            keep = ~adata.obs[column].astype(str).isin(excluded)
            adata = adata[keep].copy()
            if getattr(adata.obs[column].dtype, "name", "") == "category":
                adata.obs[column] = adata.obs[column].cat.remove_unused_categories()

        elif operation == "normalize_total":
            sc.pp.normalize_total(
                adata,
                target_sum=float(step.get("target_sum", 10000.0)),
            )

        elif operation == "log1p":
            sc.pp.log1p(adata)

        elif operation == "highly_variable_genes":
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=int(step.get("n_top_genes", 2000)),
                flavor=step.get("flavor", "seurat"),
            )

        elif operation == "log1p_if_max_gt":
            threshold = float(step.get("threshold", 100))
            _, current_max = get_matrix_min_max(adata)
            if np.isfinite(current_max) and current_max > threshold:
                sc.pp.log1p(adata)
            else:
                status = "skipped_already_below_threshold"

        elif operation == "highly_variable_genes_if_missing":
            if "highly_variable" not in adata.var:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=int(step.get("n_top_genes", 2000)),
                    flavor=step.get("flavor", "seurat"),
                )
            else:
                status = "skipped_already_available"

        elif operation == "set_raw":
            adata.raw = adata.copy()

        elif operation == "subset_highly_variable":
            if "highly_variable" not in adata.var:
                raise KeyError("subset_highly_variable requires adata.var['highly_variable']")
            mask = np.asarray(adata.var["highly_variable"], dtype=bool)
            if not mask.any():
                raise ValueError("No highly variable genes are available for subsetting")
            adata = adata[:, mask].copy()

        elif operation == "scale":
            sc.pp.scale(adata, max_value=float(step.get("max_value", 10)))

        elif operation == "pca":
            maximum_n_comps = min(int(adata.n_obs), int(adata.n_vars)) - 1
            if maximum_n_comps < 1:
                raise ValueError("PCA requires at least two cells and two variables")
            pca_kwargs = {
                "n_comps": min(int(step.get("n_comps", 50)), maximum_n_comps),
                "svd_solver": step.get("svd_solver", "arpack"),
                "zero_center": bool(step.get("zero_center", True)),
                "use_highly_variable": bool(step.get("use_highly_variable", False)),
                "random_state": int(step.get("random_state", seed)),
            }
            print(f"  PCA parameters: {pca_kwargs}")
            sc.tl.pca(adata, **pca_kwargs)

        elif operation == "neighbors":
            if "X_pca" not in adata.obsm:
                raise KeyError("neighbors requires adata.obsm['X_pca']")
            neighbors_kwargs = {
                "n_neighbors": min(
                    int(step.get("n_neighbors", 15)), max(1, int(adata.n_obs) - 1)
                ),
                "n_pcs": min(
                    int(step.get("n_pcs", 30)), int(adata.obsm["X_pca"].shape[1])
                ),
                "method": step.get("method", "umap"),
                "metric": step.get("metric", "euclidean"),
                "random_state": int(step.get("random_state", seed)),
            }
            if step.get("use_rep") is not None:
                neighbors_kwargs["use_rep"] = step["use_rep"]
            print(f"  Neighbors parameters: {neighbors_kwargs}")
            sc.pp.neighbors(adata, **neighbors_kwargs)

        elif operation == "umap":
            if "neighbors" not in adata.uns:
                raise KeyError("UMAP requires adata.uns['neighbors']")
            preserve_existing = bool(step.get("preserve_existing", False))
            if preserve_existing and "X_umap" in adata.obsm:
                existing = np.asarray(adata.obsm["X_umap"])
                expected_dimensions = int(step.get("n_components", 2))
                if existing.ndim != 2 or existing.shape != (adata.n_obs, expected_dimensions):
                    raise ValueError(
                        "Stored X_umap is incompatible with the configured cell count/dimensions: "
                        f"{existing.shape}"
                    )
                status = "reused_existing_coordinates"
                print(
                    "  Preserved stored X_umap to reproduce the reference distribution; "
                    f"shape={existing.shape}"
                )
            else:
                umap_kwargs = {
                    "n_components": int(step.get("n_components", 2)),
                    "random_state": int(step.get("random_state", seed)),
                }
                for key in ("a", "b", "min_dist", "spread"):
                    if step.get(key) is not None:
                        umap_kwargs[key] = float(step[key])
                if step.get("init_pos") is not None:
                    umap_kwargs["init_pos"] = step["init_pos"]
                if step.get("maxiter") is not None:
                    umap_kwargs["maxiter"] = int(step["maxiter"])
                print(f"  UMAP parameters: {umap_kwargs}")
                sc.tl.umap(adata, **umap_kwargs)

        else:
            raise ValueError(f"Unsupported preprocessing operation: {operation!r}")

        records.append(
            {"operation": operation, "status": status, "parameters": parameters}
        )

    audit = {
        "profile": dataset_config.get("preprocessing_profile", "dataset_configured"),
        "completed_before_inference": True,
        "random_seed": int(seed),
        "input_shape": input_shape,
        "output_shape": [int(adata.n_obs), int(adata.n_vars)],
        "input_x_min": input_min,
        "input_x_max": input_max,
        "steps": records,
        "model_coordinate_key_ready": dataset_config["model_coordinate_key"] in adata.obsm,
        "plot_coordinate_key_ready": dataset_config["plot_coordinate_key"] in adata.obsm,
    }
    adata.uns["sccgrl_preprocessing_audit"] = audit
    return adata


def apply_configured_gene_symbols(adata, dataset_config, input_path=None):
    """Use an audited gene-symbol column before any expression preprocessing."""
    symbol_column = dataset_config.get("gene_symbol_column")
    if symbol_column is None:
        return adata
    if symbol_column not in adata.var:
        raise KeyError(
            f"Configured gene symbol column {symbol_column!r} is missing from "
            f"{input_path}; available var columns={list(adata.var.columns)}"
        )
    original_ids = np.asarray(adata.var_names.astype(str))
    symbols = np.asarray(adata.var[symbol_column].astype(str).str.strip())
    invalid = np.isin(np.char.lower(symbols.astype(str)), ["", "nan", "none"])
    if invalid.any():
        raise ValueError(
            f"Gene symbol column {symbol_column!r} contains "
            f"{int(invalid.sum())} empty/invalid values"
        )
    if len(np.unique(symbols)) != len(symbols):
        raise ValueError(
            f"Gene symbol column {symbol_column!r} is not unique; refusing to "
            "silently rename or merge expression columns"
        )
    adata.var["raw_gene_id"] = original_ids
    adata.var_names = symbols
    if dataset_config.get("require_gene_symbols", False):
        numeric_fraction = float(np.mean([value.isdigit() for value in symbols]))
        if numeric_fraction > 0.95:
            raise ValueError(
                "Gene-symbol mapping failed: more than 95% of var_names remain numeric"
            )
    adata.uns["sccgrl_gene_identifier_audit"] = {
        "input_path": str(input_path) if input_path is not None else "",
        "symbol_column": str(symbol_column),
        "n_genes": int(adata.n_vars),
        "original_id_preserved_in": "var['raw_gene_id']",
        "mapping_applied_before_preprocessing": True,
    }
    return adata


def load_prepared_dataset(input_path, dataset_config, seed=42):
    """Load one H5AD, finish preprocessing, then expose arrays used downstream."""
    input_path = Path(input_path).resolve()
    adata = sc.read_h5ad(input_path)
    adata = apply_configured_gene_symbols(adata, dataset_config, input_path)
    label_column = dataset_config["label_column"]
    if label_column not in adata.obs:
        raise KeyError(f"{label_column!r} not in adata.obs: {list(adata.obs.columns)}")

    adata = apply_preprocessing_steps(adata, dataset_config, seed)
    audit = dict(adata.uns.get("sccgrl_preprocessing_audit", {}))
    if not audit.get("completed_before_inference", False):
        raise RuntimeError("Preprocessing did not complete before model inference")

    model_key = dataset_config["model_coordinate_key"]
    plot_key = dataset_config["plot_coordinate_key"]
    for key in (model_key, plot_key):
        if key not in adata.obsm:
            raise KeyError(f"{key!r} not in adata.obsm: {list(adata.obsm.keys())}")

    labels = np.asarray(adata.obs[label_column].astype(object))
    early_label = match_label(labels, dataset_config["early_label"])
    model_coords = select_dimensions(
        adata.obsm[model_key], dataset_config["model_dimensions"], model_key
    )
    plot_coords = select_dimensions(
        adata.obsm[plot_key], dataset_config["plot_dimensions"], plot_key
    )
    return {
        "input_path": input_path,
        "adata": adata,
        "labels": labels,
        "early_label": early_label,
        "model_coords": model_coords,
        "plot_coords": plot_coords,
        "preprocessing_audit": audit,
    }
