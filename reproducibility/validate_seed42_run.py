"""Validate one repository seed-42 run against an established historical run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


RESOURCE_COLUMNS = {
    "preprocessing_runtime_seconds",
    "inference_runtime_seconds",
    "pipeline_runtime_seconds",
    "trajectory_metrics_runtime_seconds",
    "pipeline_peak_rss_mb",
    "pipeline_memory_increase_mb",
}


def matrix_comparison(previous_path: Path, current_path: Path) -> dict:
    previous = sc.read_h5ad(previous_path)
    current = sc.read_h5ad(current_path)
    same_shape = previous.shape == current.shape
    same_cells = np.array_equal(previous.obs_names, current.obs_names)
    same_genes = np.array_equal(previous.var_names, current.var_names)
    if same_shape:
        difference = previous.X - current.X
        different_entries = int(
            difference.nnz if sparse.issparse(difference)
            else np.count_nonzero(difference)
        )
        if sparse.issparse(difference):
            max_absolute_difference = (
                float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
            )
        else:
            max_absolute_difference = float(np.max(np.abs(difference)))
    else:
        different_entries = -1
        max_absolute_difference = float("nan")
    return {
        "same_shape": bool(same_shape),
        "same_cell_ids_and_order": bool(same_cells),
        "same_gene_ids_and_order": bool(same_genes),
        "number_of_different_entries": different_entries,
        "max_absolute_difference": max_absolute_difference,
        "exact_matrix_match": bool(
            same_shape and same_cells and same_genes and different_entries == 0
        ),
    }


def metric_comparison(
    previous_csv: Path,
    current_csv: Path,
    expected_differences: set[str] | None = None,
) -> pd.DataFrame:
    expected_differences = expected_differences or set()
    previous = pd.read_csv(previous_csv).iloc[0]
    current = pd.read_csv(current_csv).iloc[0]
    rows = []
    for metric in sorted(set(previous.index).intersection(current.index)):
        old_value, new_value = previous[metric], current[metric]
        if metric in RESOURCE_COLUMNS:
            category = "resource"
        elif metric in expected_differences:
            category = "expected_context_difference"
        else:
            category = "model_or_metric"
        old_number = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
        new_number = pd.to_numeric(pd.Series([new_value]), errors="coerce").iloc[0]
        if pd.notna(old_number) and pd.notna(new_number):
            absolute_difference = float(abs(float(new_number) - float(old_number)))
            equal = bool(np.isclose(old_number, new_number, rtol=1e-10, atol=1e-12))
        else:
            absolute_difference = np.nan
            equal = bool(
                (pd.isna(old_value) and pd.isna(new_value))
                or str(old_value) == str(new_value)
            )
        rows.append({
            "metric": metric,
            "category": category,
            "historical_seed42": old_value,
            "repository_seed42": new_value,
            "absolute_difference": absolute_difference,
            "equal_with_tolerance": equal,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-raw", type=Path, required=True)
    parser.add_argument("--current-raw", type=Path, required=True)
    parser.add_argument("--previous-metrics", type=Path, required=True)
    parser.add_argument("--current-metrics", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--expected-difference",
        action="append",
        default=[],
        help="Metric field whose difference is expected and must be reported separately",
    )
    args = parser.parse_args()

    matrix = matrix_comparison(args.previous_raw, args.current_raw)
    metrics = metric_comparison(
        args.previous_metrics,
        args.current_metrics,
        expected_differences=set(args.expected_difference),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    model_rows = metrics[metrics["category"].eq("model_or_metric")]
    expected_rows = metrics[metrics["category"].eq("expected_context_difference")]
    summary = {
        "raw_input_comparison": matrix,
        "common_metric_count": int(len(metrics)),
        "model_or_metric_equal_count": int(model_rows["equal_with_tolerance"].sum()),
        "model_or_metric_mismatch_count": int((~model_rows["equal_with_tolerance"]).sum()),
        "expected_context_difference_fields": expected_rows["metric"].tolist(),
        "resource_values_expected_to_vary": True,
        "model_result_match": bool(model_rows["equal_with_tolerance"].all()),
    }
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if matrix["exact_matrix_match"] and summary["model_result_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
