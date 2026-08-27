#!/usr/bin/env python
"""Run the audited scCGRL pipeline from repository YAML configuration.

Algorithm source: 2026-08-17_scCGRL_five_datasets_v1.ipynb, code revision 1.6.
This adapter changes path/config loading only; it does not change preprocessing,
endpoint discovery, Q-learning, pseudotime mapping, RF propagation, or metrics.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sccgrl.trajectory import (  # noqa: E402
    CODE_REVISION,
    run_repeated_experiments,
    run_single_experiment,
)


DATASET_KEYS = (
    "human_myeloid",
    "mouse_pancreas",
    "human_bone_marrow",
    "simulation_2",
    "simulation_3",
)
DEFAULT_EPISODES = 10000
DEFAULT_SEED = 42


def _process_is_active(pid: int) -> bool:
    """Return whether a recorded run-lock owner is still alive."""
    if pid <= 0 or not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@contextmanager
def dataset_run_lock(
    output_dir: Path,
    *,
    dataset: str,
    seed: int,
    episodes: int,
    runs: int,
):
    """Prevent concurrent processes from writing the same dataset output.

    The lock is created atomically. A stale lock left by an interrupted process
    is removed only after confirming that its recorded PID is no longer active.
    """
    lock_path = output_dir / ".sccgrl_run.lock"
    metadata = {
        "pid": int(os.getpid()),
        "dataset": str(dataset),
        "seed": int(seed),
        "episodes": int(episodes),
        "runs": int(runs),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    while True:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
            break
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", -1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                existing = {}
                existing_pid = -1

            if _process_is_active(existing_pid):
                raise RuntimeError(
                    "Another scCGRL process is already writing this dataset "
                    f"output: {output_dir} (PID {existing_pid})."
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    try:
        yield lock_path
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if int(current.get("pid", -1)) == os.getpid():
                lock_path.unlink()
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            pass


def _portable_path(value) -> str | None:
    """Store repository paths relative to the repository when possible."""
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def default_project_root() -> Path:
    """Resolve the repository root, with an optional explicit override."""
    override = os.environ.get("SCCGRL_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return REPOSITORY_ROOT


def _expand_config_value(value, project_root: Path):
    if isinstance(value, str):
        return (
            value.replace("${PROJECT_ROOT}", project_root.as_posix())
            .replace("${REPOSITORY_ROOT}", REPOSITORY_ROOT.as_posix())
        )
    if isinstance(value, list):
        return [_expand_config_value(item, project_root) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_value(item, project_root) for key, item in value.items()}
    return value


def load_dataset_config(dataset: str, project_root: Path | None = None) -> dict:
    if dataset not in DATASET_KEYS:
        raise KeyError(f"Unknown dataset {dataset!r}; choose from {DATASET_KEYS}")
    path = REPOSITORY_ROOT / "configs" / f"{dataset}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = default_project_root() if project_root is None else Path(project_root).resolve()
    config = _expand_config_value(config, root)
    config["key"] = dataset
    config["repository_config"] = str(path)
    return config


def resolve_input(config: dict, explicit_input: str | Path | None = None) -> Path:
    if explicit_input is not None:
        candidate = Path(explicit_input).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = [Path(value).expanduser() for value in config.get("input_candidates", [])]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No configured input exists. Checked: " + ", ".join(map(str, candidates))
    )


def resolve_processed_input(
    config: dict,
    explicit_input: str | Path | None = None,
) -> Path:
    """Resolve a repository-packaged, model-ready processed H5AD."""
    if explicit_input is not None:
        candidate = Path(explicit_input).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = [
        Path(value).expanduser()
        for value in config.get("processed_input_candidates", [])
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No configured processed input exists. Checked: "
        + ", ".join(map(str, candidates))
    )


def configure_model_ready_input(config: dict) -> dict:
    """Reuse audited PCA/neighbors/UMAP without repeating preprocessing."""
    runtime = copy.deepcopy(config)
    runtime["preprocessing_profile"] = runtime.get(
        "processed_preprocessing_profile",
        f"{runtime.get('preprocessing_profile', 'configured')}_model_ready_cache",
    )
    runtime["preprocessing_steps"] = copy.deepcopy(
        runtime.get(
            "processed_preprocessing_steps",
            [
                {
                    "operation": "reuse_existing_representations",
                    "required_obsm": ["X_pca", "X_umap"],
                    "expected_umap_dimensions": 3,
                }
            ],
        )
    )
    # Gene-symbol conversion was already applied before the cache was written.
    runtime.pop("gene_symbol_column", None)
    runtime.pop("require_gene_symbols", None)
    runtime["input_stage"] = "processed_model_ready"
    return runtime


def repository_commit_hash() -> str:
    """Return the exact Git commit used for a run, when available."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "not_available"


def write_run_manifest(
    output_dir: Path,
    dataset: str,
    config: dict,
    input_path: Path,
    seed: int,
    episodes: int,
    runs: int,
    input_stage: str,
) -> Path:
    manifest = {
        "repository_version": CODE_REVISION,
        "repository_commit": repository_commit_hash(),
        "canonical_source_notebook": "2026-08-17_scCGRL_five_datasets_v1.ipynb",
        "source_notebook_date": "2026-08-17",
        "dataset": dataset,
        "input_file": _portable_path(input_path),
        "input_stage": str(input_stage),
        "config_file": _portable_path(config.get("repository_config")),
        "preprocessing_profile": config.get("preprocessing_profile"),
        "seed": int(seed),
        "episodes": int(episodes),
        "runs": int(runs),
        "evaluation_metric_version": "v2.0",
        "provenance_conflicts": {
            "human_bone_marrow_notebook_display_seed": 81 if dataset == "human_bone_marrow" else None,
            "selected_cli_seed": int(seed),
        },
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def resolve_run_output_dir(
    output_root: str | Path,
    dataset: str,
    seed: int,
    runs: int,
) -> Path:
    """Return the canonical dataset/seed output directory."""
    root = Path(output_root).expanduser().resolve()
    if int(runs) == 1:
        run_label = f"seed{int(seed)}"
    else:
        last_seed = int(seed) + int(runs) - 1
        run_label = f"seeds{int(seed)}-{last_seed}"
    return root / dataset / run_label


def run_dataset(
    dataset: str,
    output_root: str | Path,
    *,
    input_path: str | Path | None = None,
    project_root: str | Path | None = None,
    episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    runs: int = 1,
    save_processed: bool = True,
    input_stage: str = "auto",
):
    config = load_dataset_config(dataset, project_root=project_root)
    requested_stage = str(input_stage).lower()
    if requested_stage not in {"auto", "raw", "processed"}:
        raise ValueError("input_stage must be one of: auto, raw, processed")

    selected_stage = requested_stage
    if requested_stage == "processed":
        resolved_input = resolve_processed_input(config, input_path)
        runtime_config = configure_model_ready_input(config)
    elif requested_stage == "raw":
        resolved_input = resolve_input(config, input_path)
        runtime_config = copy.deepcopy(config)
    elif input_path is not None:
        # Explicit inputs retain the historical raw/preprocessing behavior.
        selected_stage = "raw"
        resolved_input = resolve_input(config, input_path)
        runtime_config = copy.deepcopy(config)
    else:
        try:
            resolved_input = resolve_input(config)
            runtime_config = copy.deepcopy(config)
            selected_stage = "raw"
        except FileNotFoundError:
            resolved_input = resolve_processed_input(config)
            runtime_config = configure_model_ready_input(config)
            selected_stage = "processed"

    runtime_config.pop("input_candidates", None)
    runtime_config.pop("processed_input_candidates", None)
    runtime_config.pop("processed_preprocessing_steps", None)
    output_dir = resolve_run_output_dir(output_root, dataset, seed, runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    with dataset_run_lock(
        output_dir,
        dataset=dataset,
        seed=seed,
        episodes=episodes,
        runs=runs,
    ):
        write_run_manifest(
            output_dir,
            dataset,
            config,
            resolved_input,
            seed,
            episodes,
            runs,
            selected_stage,
        )
        if int(runs) == 1:
            return run_single_experiment(
                resolved_input,
                output_dir,
                runtime_config,
                episodes=int(episodes),
                seed=int(seed),
                save_processed=bool(save_processed),
            )
        return run_repeated_experiments(
            resolved_input,
            output_dir,
            runtime_config,
            runs=int(runs),
            episodes=int(episodes),
            seed=int(seed),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--input", default=None, help="Explicit H5AD input override")
    parser.add_argument("--project-root", default=None, help="Root containing data/ and results/")
    parser.add_argument("--output-root", default=str(REPOSITORY_ROOT / "results"))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--input-stage",
        choices=("auto", "raw", "processed"),
        default="auto",
        help=(
            "raw: apply configured preprocessing; processed: reuse the packaged "
            "audited PCA/neighbors/UMAP; auto: prefer raw and otherwise use processed"
        ),
    )
    parser.add_argument("--no-save-processed", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_config:
        config = load_dataset_config(args.dataset, project_root=args.project_root)
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return 0
    result = run_dataset(
        args.dataset,
        args.output_root,
        input_path=args.input,
        project_root=args.project_root,
        episodes=args.episodes,
        seed=args.seed,
        runs=args.runs,
        save_processed=not args.no_save_processed,
        input_stage=args.input_stage,
    )
    if isinstance(result, dict):
        dataset_dir = resolve_run_output_dir(
            args.output_root,
            args.dataset,
            args.seed,
            args.runs,
        )
        print(f"Completed {args.dataset}: {dataset_dir}")
        print(f"Metrics: {result.get('metrics_table', dataset_dir)}")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
