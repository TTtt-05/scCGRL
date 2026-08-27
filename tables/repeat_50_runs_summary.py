"""Summarize five-dataset repeated-run CSV files without rerunning scCGRL."""
import argparse
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tables.common_summary import read_csvs, summarize_numeric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input_root.rglob("*_50_runs.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "dataset" not in frame:
            frame.insert(0, "dataset", path.name.removesuffix("_50_runs.csv"))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    summary = summarize_numeric(data, ["dataset"]) if not data.empty else pd.DataFrame()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
