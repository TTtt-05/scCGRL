# Audited 8.12 baseline orchestrator; import paths are adapted below only.
import argparse
import gc
import importlib.metadata
import json
import platform
import time
import traceback
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark._config import (
    ALL_METHODS, BASE_SEED, BENCHMARK_ROOT, DATASETS, EXTERNAL_METHODS, REPEAT_RUNS,
    RESULTS_ROOT, ROOT_USAGE, R_METHODS, SYSTEM_RSCRIPT, CURRENT_R_LIBRARY,
    MONOCLE2_ENV, MONOCLE2_R_LIBRARY,
    MONOCLE2_LEGACY_DPLYR_LIBRARY, MONOCLE3_R_LIBRARY, MONOCLE1_ENV,
    MONOCLE1_RSCRIPT, MONOCLE1_R_LIBRARY,
)
from benchmark.common.evaluation import evaluate, normalize01, numeric_summary
from benchmark.common.python_methods import run_python_method
from benchmark.common.r_bridge import run_r_method
from benchmark.common.preprocessing import (
    build_baseline_input, build_root_anchor, evaluation_context, import_sccgrl_rows,
    load_project_namespace, resolve_dataset,
)
from benchmark.common.resource_monitor import ProcessTreeRSSMonitor
from tables.benchmark_summary import summarize


_EVALUATION_CONTEXT_CACHE = {}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def prepare_directories():
    for method in ALL_METHODS:
        (RESULTS_ROOT / method.lower()).mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "combined").mkdir(parents=True, exist_ok=True)


def locate_root(cell_ids, root_cell_id):
    matches = np.flatnonzero(np.asarray(cell_ids).astype(str) == str(root_cell_id))
    if matches.size != 1:
        raise ValueError("Shared scCGRL root does not map uniquely to baseline cells")
    return int(matches[0])


def save_method_parameters(method, method_dir, parameter_records):
    requested_runs = 1 if method == "SLICER" else REPEAT_RUNS
    payload = {
        "method": method,
        "runs": requested_runs,
        "seeds": list(range(BASE_SEED, BASE_SEED + requested_runs)),
        "root_usage": ROOT_USAGE.get(method, "scCGRL existing result"),
        "terminal_information_supplied": False if method != "scCGRL" else None,
        "records": parameter_records,
    }
    (method_dir / "method_parameters.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def import_sccgrl(dataset, runs):
    method_dir = RESULTS_ROOT / "sccgrl" / dataset
    method_dir.mkdir(parents=True, exist_ok=True)
    frame = import_sccgrl_rows(dataset, runs)
    frame.to_csv(method_dir / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    resources = frame[[column for column in frame if any(token in column.lower() for token in ("runtime", "rss", "memory"))]].copy()
    resources.insert(0, "random_seed", frame["random_seed"].to_numpy())
    resources.to_csv(method_dir / "resource_metrics.csv", index=False, encoding="utf-8-sig")
    save_method_parameters("scCGRL", method_dir, [{"source": value} for value in frame["scCGRL_result_source"].unique()])


def run_method_dataset(namespace, dataset, method, runs, base_seed, resume=False):
    config, input_path = resolve_dataset(namespace, dataset)
    dataset_dir = RESULTS_ROOT / method.lower() / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows, resources, parameter_records = [], [], []
    metrics_path = dataset_dir / "evaluation_metrics.csv"
    resources_path = dataset_dir / "resource_metrics.csv"
    parameters_path = dataset_dir / "method_parameters.json"
    if resume and metrics_path.exists():
        rows = pd.read_csv(metrics_path).to_dict("records")
    if resume and resources_path.exists():
        resources = pd.read_csv(resources_path).to_dict("records")
    if resume and parameters_path.exists():
        try:
            parameter_records = json.loads(parameters_path.read_text(encoding="utf-8"))\
                .get("records", [])
        except (OSError, ValueError, TypeError):
            parameter_records = []
    seeds = list(range(base_seed, base_seed + runs))
    successful_seeds = {
        int(record["seed"])
        for record in rows
        if str(record.get("status", "")).lower() == "success"
        and pd.notna(record.get("seed"))
    }
    for run_number, seed in enumerate(seeds, 1):
        if resume and seed in successful_seeds:
            print(f"[{dataset}] [{method}] seed={seed} SKIP existing success", flush=True)
            continue
        if resume:
            rows = [record for record in rows if int(record.get("seed", -1)) != seed]
            resources = [record for record in resources if int(record.get("seed", -1)) != seed]
            parameter_records = [
                record for record in parameter_records if int(record.get("seed", -1)) != seed
            ]
        print(f"[{dataset}] [{method}] [{run_number}/{len(seeds)}] seed={seed} START", flush=True)
        row = {
            "dataset": dataset, "method": method, "run": run_number,
            "seed": seed, "random_seed": seed, "status": "failed",
            "error_message": "", "traceback": "",
        }
        resource = {"dataset": dataset, "method": method, "seed": seed, "status": "failed"}
        run_started = time.perf_counter()
        monitor = None
        try:
            # This context is exclusively the common evaluation/reference layer.
            # Its scCGRL preprocessing is not charged to baseline runtime.
            # The formal reference context uses fixed preprocessing random
            # states in DATASET_CONFIGS and is identical across seeds. Cache
            # one read-only context per dataset; per-seed roots still come
            # from the existing scCGRL CSV and baseline preprocessing reruns.
            cache_key = dataset
            if cache_key not in _EVALUATION_CONTEXT_CACHE:
                cached_context = evaluation_context(namespace, input_path, config, seed)
                _EVALUATION_CONTEXT_CACHE[cache_key] = cached_context
            context = _EVALUATION_CONTEXT_CACHE[cache_key]
            anchor = build_root_anchor(
                namespace, dataset, config, input_path, seed=seed, context=context
            )
            with ProcessTreeRSSMonitor() as monitor:
                preprocessing_started = time.perf_counter()
                # Reading the original AnnData and subsetting the identical
                # cells is part of every method run and is charged to runtime.
                baseline_input = build_baseline_input(
                    namespace, input_path, config, context
                )
                root_index = locate_root(baseline_input["cell_ids"], anchor["start_cell_id"])
                shared_input_seconds = time.perf_counter() - preprocessing_started
                if method in R_METHODS or method == "Monocle1":
                    result = run_r_method(method, baseline_input, seed, anchor["start_cell_id"], dataset_dir)
                else:
                    result = run_python_method(method, baseline_input, seed, root_index)
                pseudotime = normalize01(result["pseudotime"])
                pipeline_seconds = shared_input_seconds + float(np.nan_to_num(result["preprocessing_seconds"])) + float(result["inference_seconds"])
            metric_started = time.perf_counter()
            metrics, endpoints, _, _ = evaluate(
                namespace, context, config, seed, root_index, pseudotime, result["branches"],
                has_native_branches=result.get("has_native_branches", True),
                has_native_topology=result.get("has_native_topology", True),
            )
            metric_seconds = time.perf_counter() - metric_started
            row.update(metrics)
            row.update({
                "status": "success",
                "input_file": str(input_path),
                "input_cell_count": int(context["adata"].n_obs),
                "input_feature_count": int(len(baseline_input["gene_ids"])),
                "input_cell_ids_sha256": baseline_input["cell_ids_sha256"],
                "input_expression_sha256": baseline_input["expression_sha256"],
                "expression_source": baseline_input["expression_source"],
                "expression_is_counts": baseline_input["expression_is_counts"],
                "root_cell_id": anchor["start_cell_id"],
                "root_index": root_index,
                "scCGRL_same_seed_root_source": anchor["source"],
                "scCGRL_same_seed_K_audit": anchor["K"],
                "root_usage": ROOT_USAGE[method],
                "terminal_information_supplied": False,
                "true_labels_used_during_inference": False,
                "reference_pseudotime_used_for_orientation": False,
                "method_parameters_json": json.dumps(result["parameters"], ensure_ascii=False, default=str),
                "preprocessing_runtime_seconds": shared_input_seconds + float(np.nan_to_num(result["preprocessing_seconds"])),
                "inference_runtime_seconds": float(result["inference_seconds"]),
                "total_runtime_seconds": pipeline_seconds,
                "trajectory_metrics_runtime_seconds": metric_seconds,
                "peak_rss_memory_mb": monitor.peak_mb,
                "memory_increase_mb": monitor.increase_mb,
            })
            resource.update({
                "status": "success",
                "preprocessing_runtime_seconds": row["preprocessing_runtime_seconds"],
                "inference_runtime_seconds": row["inference_runtime_seconds"],
                "total_runtime_seconds": row["total_runtime_seconds"],
                "trajectory_metrics_runtime_seconds": metric_seconds,
                "peak_rss_memory_mb": monitor.peak_mb,
                "memory_increase_mb": monitor.increase_mb,
            })
            cell_frame = pd.DataFrame({
                "cell_id": baseline_input["cell_ids"],
                "pseudotime": pseudotime,
                "branch": np.asarray(result["branches"]).astype(str),
            })
            for lineage, values in result.get("lineage_pseudotime", {}).items():
                cell_frame[f"lineage_{lineage}_pseudotime"] = values
            cell_frame.to_csv(
                dataset_dir / f"cell_pseudotime_run{run_number:02d}.csv",
                index=False, encoding="utf-8-sig",
            )
            parameter_records.append({"seed": seed, **result["parameters"]})
        except Exception as error:
            row["error_message"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()
            resource["error_message"] = row["error_message"]
            failed_elapsed = time.perf_counter() - run_started
            failed_peak = float(getattr(monitor, "peak_mb", np.nan))
            failed_increase = float(getattr(monitor, "increase_mb", np.nan))
            row.update({
                "preprocessing_runtime_seconds": np.nan,
                "inference_runtime_seconds": np.nan,
                "total_runtime_seconds": failed_elapsed,
                "trajectory_metrics_runtime_seconds": np.nan,
                "peak_rss_memory_mb": failed_peak,
                "memory_increase_mb": failed_increase,
            })
            resource.update({
                "preprocessing_runtime_seconds": np.nan,
                "inference_runtime_seconds": np.nan,
                "total_runtime_seconds": failed_elapsed,
                "trajectory_metrics_runtime_seconds": np.nan,
                "peak_rss_memory_mb": failed_peak,
                "memory_increase_mb": failed_increase,
            })
            with (dataset_dir / "error.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{dataset}/{method}/seed={seed}]\n{row['traceback']}\n")
        rows.append(row)
        resources.append(resource)
        rows = sorted(rows, key=lambda record: int(record.get("seed", -1)))
        resources = sorted(resources, key=lambda record: int(record.get("seed", -1)))
        pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(resources).to_csv(resources_path, index=False, encoding="utf-8-sig")
        save_method_parameters(method, dataset_dir, parameter_records)
        print(f"[{dataset}] [{method}] seed={seed} {row['status']}", flush=True)
        gc.collect()
    summary = numeric_summary(pd.DataFrame(rows))
    summary.to_csv(dataset_dir / "statistical_summary.csv", index=False, encoding="utf-8-sig")


def write_environment_manifest():
    r_packages = {}
    r_version = "unavailable"
    try:
        expression = (
            f".libPaths(c('{CURRENT_R_LIBRARY.as_posix()}',.libPaths()));"
            "cat(R.version.string,'\\n');"
            "for(p in c('monocle','monocle3','slingshot','TSCAN','SLICER','SCORPIUS','igraph','SingleCellExperiment'))"
            "cat(p,'=',if(requireNamespace(p,quietly=TRUE)) as.character(packageVersion(p)) else 'not installed','\\n')"
        )
        completed = subprocess.run(
            [str(SYSTEM_RSCRIPT), "-e", expression],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if lines:
            r_version = lines[0]
        for line in lines[1:]:
            if "=" in line:
                name, version = line.split("=", 1)
                r_packages[name.strip()] = version.strip()
    except Exception as error:
        r_packages["manifest_error"] = f"{type(error).__name__}: {error}"
    method_specific_r = {
        "Monocle1": {
            "R": "3.1.3",
            "environment": str(MONOCLE1_ENV),
            "Rscript": str(MONOCLE1_RSCRIPT),
            "library": str(MONOCLE1_R_LIBRARY),
            "packages": {
                "monocle": "1.0.0",
                "HSMMSingleCell": "1.0.0",
                "Biobase": "2.26.0",
                "VGAM": "1.0-1",
                "igraph": "1.0.1",
                "irlba": "1.0.3",
            },
        },
        "Monocle2": {
            "R": "4.4.3",
            "environment": str(MONOCLE2_ENV),
            "library": str(MONOCLE2_R_LIBRARY),
            "legacy_dplyr_library": str(MONOCLE2_LEGACY_DPLYR_LIBRARY),
            "packages": {
                "monocle": "2.40.0",
                "igraph": "2.0.3",
                "dplyr": "0.8.5",
                "DDRTree": "0.1.6",
                "VGAM": "1.1-14",
            },
        },
        "Monocle3": {
            "R": "4.6.1",
            "library": str(MONOCLE3_R_LIBRARY),
            "source": str(BENCHMARK_ROOT / "environments" / "monocle3_official_source"),
            "packages": {"monocle3": "1.4.27"},
        },
    }
    payload = {
        "created": "2026-08-12",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in (
            "scanpy", "anndata", "numpy", "scipy", "pandas", "scikit-learn", "palantir", "psutil"
        )},
        "R": r_version,
        "R_packages": r_packages,
        "R_library": str(CURRENT_R_LIBRARY),
        "method_specific_R_environments": method_specific_r,
        "scCGRL_rerun": False,
        "baseline_runs": REPEAT_RUNS,
        "seeds": list(range(BASE_SEED, BASE_SEED + REPEAT_RUNS)),
    }
    (RESULTS_ROOT / "combined" / "environment_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=EXTERNAL_METHODS, default=list(EXTERNAL_METHODS))
    parser.add_argument("--runs", type=int, default=REPEAT_RUNS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--skip-sccgrl-import", action="store_true")
    parser.add_argument("--defer-combined-summary", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Preserve successful seeds and rerun only missing/failed seeds.",
    )
    args = parser.parse_args(argv)
    prepare_directories()
    namespace = load_project_namespace()
    write_environment_manifest()
    for dataset in args.datasets:
        if not args.skip_sccgrl_import:
            import_sccgrl(dataset, args.runs)
        for method in args.methods:
            run_method_dataset(
                namespace, dataset, method, args.runs, args.seed,
                resume=args.resume,
            )
        if not args.defer_combined_summary:
            summarize()
    if not args.defer_combined_summary:
        summarize()


if __name__ == "__main__":
    main()
