"""Generate the canonical simulation_3 single-run figures (seed 42 by default)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from run_sccgrl import run_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=REPO_ROOT / "results" / "figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    run_dataset("simulation_3", args.output_root, seed=args.seed,
                episodes=args.episodes, runs=1)


if __name__ == "__main__":
    main()
