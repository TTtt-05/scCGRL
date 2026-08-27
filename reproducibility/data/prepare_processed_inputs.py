"""Create the five configured preprocessed H5AD files without trajectory inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from run_sccgrl import DATASET_KEYS, load_dataset_config, resolve_input
from sccgrl.preprocessing import load_prepared_dataset


def sequence_hash(values) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the file is packaged locally."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def prepare_one(dataset: str, output_root: Path, seed: int, force: bool = False) -> dict:
    config = load_dataset_config(dataset)
    source = resolve_input(config)
    target = output_root / f"{dataset}.h5ad"
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --force explicitly")
    runtime_config = dict(config)
    runtime_config.pop("input_candidates", None)
    prepared = load_prepared_dataset(source, runtime_config, seed=int(seed))
    adata = prepared["adata"]
    adata.uns["sccgrl_processed_input_provenance"] = {
        "dataset": dataset,
        "source_file": portable_path(source),
        "config_file": portable_path(config["repository_config"]),
        "preprocessing_profile": str(config.get("preprocessing_profile", "")),
        "preprocessing_seed": int(seed),
        "trajectory_inference_performed": False,
    }
    # AnnData/HDF5 cannot serialize a heterogeneous list of dictionaries.
    # Preserve the complete preprocessing audit as JSON without changing any
    # expression matrix, representation, or model parameter.
    preprocessing_audit = dict(adata.uns.get("sccgrl_preprocessing_audit", {}))
    if isinstance(preprocessing_audit.get("steps"), list):
        preprocessing_audit["steps_json"] = json.dumps(
            preprocessing_audit.pop("steps"), ensure_ascii=False, sort_keys=True
        )
    adata.uns["sccgrl_preprocessing_audit"] = preprocessing_audit
    target.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(target, compression="gzip")
    return {
        "dataset": dataset,
        "source_file": portable_path(source),
        "processed_file": portable_path(target),
        "preprocessing_seed": int(seed),
        "n_cells": int(adata.n_obs),
        "n_model_genes": int(adata.n_vars),
        "raw_gene_count": int(adata.raw.n_vars) if adata.raw is not None else np.nan,
        "cell_id_sha256": sequence_hash(adata.obs_names),
        "gene_id_sha256": sequence_hash(adata.var_names),
        "pca_dimensions": int(adata.obsm["X_pca"].shape[1]),
        "umap_dimensions": int(adata.obsm["X_umap"].shape[1]),
        "preprocessing_profile": config.get("preprocessing_profile"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS,
                        default=list(DATASET_KEYS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path,
                        default=REPOSITORY_ROOT / "data" / "processed")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = [prepare_one(dataset, args.output_root, args.seed, args.force)
            for dataset in args.datasets]
    audit = pd.DataFrame(rows)
    audit_path = args.output_root / "processed_inputs_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    (args.output_root / "processed_inputs_manifest.json").write_text(
        json.dumps({"seed": args.seed, "datasets": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
