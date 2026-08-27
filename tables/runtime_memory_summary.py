"""Summarize runtime and memory fields from any combined all-runs CSV."""
import argparse
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tables.common_summary import summarize_numeric

RESOURCE_COLUMNS = [
    "pipeline_runtime_seconds", "preprocessing_runtime_seconds", "inference_runtime_seconds",
    "trajectory_metrics_runtime_seconds", "pipeline_peak_rss_mb", "pipeline_memory_increase_mb",
    "total_runtime_seconds", "peak_rss_mb", "memory_increase_mb",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    groups = [name for name in ("dataset", "method", "variant", "condition") if name in frame]
    keep = list(dict.fromkeys(groups + (["status"] if "status" in frame else []) +
                              [name for name in RESOURCE_COLUMNS if name in frame]))
    summary = summarize_numeric(frame[keep], groups)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
