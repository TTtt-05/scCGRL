# Exact audited benchmark summary implementation.
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from benchmark._config import ALL_METHODS, DATASETS, REPEAT_RUNS, RESULTS_ROOT
from benchmark.common.evaluation import numeric_summary


def summarize(results_root=RESULTS_ROOT):
    (results_root / "combined").mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in DATASETS:
        for method in ALL_METHODS:
            path = results_root / method.lower() / dataset / "evaluation_metrics.csv"
            if path.exists():
                frame = pd.read_csv(path)
                if "native_branch_output_available" in frame:
                    unavailable = frame["native_branch_output_available"].astype(str).str.lower().eq("false")
                    frame.loc[unavailable, [column for column in (
                        "F1_branches", "endpoint_count", "branch_count"
                    ) if column in frame]] = float("nan")
                if "native_topology_output_available" in frame:
                    unavailable = frame["native_topology_output_available"].astype(str).str.lower().eq("false")
                    frame.loc[unavailable, [column for column in (
                        "HIM_similarity", "trajectory_overall_geometric_mean"
                    ) if column in frame]] = float("nan")
                # Persist the final scientific-missingness policy into the
                # method-level file as well as the combined output.
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                summary_frame = frame.copy()
                if method == "SLICER":
                    # SLICER official defaults exceeded three hours per run;
                    # the approved comparison therefore uses seed 42 only.
                    seed_column = "random_seed" if "random_seed" in summary_frame else "seed"
                    summary_frame = summary_frame[
                        pd.to_numeric(summary_frame[seed_column], errors="coerce").eq(42)
                    ].copy()
                summary_frame["dataset"] = dataset
                summary_frame["method"] = method
                frames.append(summary_frame)
    all_runs = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    all_runs.to_csv(results_root / "combined" / "benchmark_all_runs.csv", index=False, encoding="utf-8-sig")
    summaries, completion = [], []
    for (dataset, method), group in all_runs.groupby(["dataset", "method"], sort=False):
        success = int((group["status"] == "success").sum())
        failed = int((group["status"] == "failed").sum())
        expected_runs = 1 if method == "SLICER" else REPEAT_RUNS
        completion.append({
            "dataset": dataset, "method": method, "expected_runs": expected_runs,
            "successful_runs": success, "failed_runs": failed,
        })
        summary = numeric_summary(group)
        if not summary.empty:
            summary.insert(0, "dataset", dataset)
            summary.insert(1, "method", method)
            summary.insert(2, "successful_runs", success)
            summary.insert(3, "failed_runs", failed)
            summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined.to_csv(results_root / "combined" / "benchmark_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(completion).to_csv(
        results_root / "combined" / "benchmark_completion.csv", index=False, encoding="utf-8-sig"
    )
    return all_runs, combined


if __name__ == "__main__":
    summarize()
