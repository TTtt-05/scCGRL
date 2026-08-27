#!/usr/bin/env python
"""Run the audited 50-seed scCGRL repetition protocol.

Source: 2026-08-17_scCGRL_five_datasets_v1.ipynb,
cell index 35/order 36.  Seeds are 42..91 and episodes are 10,000.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from run_sccgrl import DATASET_KEYS, load_dataset_config, resolve_input
from sccgrl.trajectory import run_repeated_experiments


REPEAT_RUNS = 50
REPEAT_EPISODES = 10000
REPEAT_BASE_SEED = 42
REPEAT_SEEDS = tuple(range(REPEAT_BASE_SEED, REPEAT_BASE_SEED + REPEAT_RUNS))


def run_repeat_50(dataset, output_root, *, input_path=None, project_root=None):
    config = load_dataset_config(dataset, project_root=project_root)
    resolved_input = resolve_input(config, explicit_input=input_path)
    config = copy.deepcopy(config)
    config.pop("input_candidates", None)
    config["key"] = dataset
    output_dir = Path(output_root).resolve() / f"{dataset}_repeat_{REPEAT_RUNS}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_index, seed in enumerate(REPEAT_SEEDS, start=1):
        run_dir = output_dir / f"run_{run_index:03d}_seed_{seed}"
        temporary_csv = run_repeated_experiments(
            input_path=resolved_input,
            output_dir=run_dir,
            dataset_config=copy.deepcopy(config),
            runs=1,
            episodes=REPEAT_EPISODES,
            seed=seed,
        )
        frame = pd.read_csv(temporary_csv)
        frame["run"] = run_index
        frame["seed"] = seed
        rows.append(frame)
        temporary_csv = Path(temporary_csv)
        if temporary_csv.exists():
            temporary_csv.unlink()
    combined = pd.concat(rows, ignore_index=True)
    front = ["run", "seed"]
    combined = combined[front + [column for column in combined if column not in front]]
    final_csv = output_dir / f"{dataset}_{REPEAT_RUNS}_runs.csv"
    combined.to_csv(final_csv, index=False, encoding="utf-8-sig")
    return final_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--output-root", default=str(REPOSITORY_ROOT / "results" / "repeat_50_runs"))
    parser.add_argument("--input", default=None)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    print(run_repeat_50(
        args.dataset,
        args.output_root,
        input_path=args.input,
        project_root=args.project_root,
    ))


if __name__ == "__main__":
    main()
