#!/usr/bin/env python
"""Verify the three packaged raw-count H5AD inputs and optional references.

The verifier never modifies H5AD inputs.  It requires exact cell/gene order and
integer sparse-matrix equality; no mismatch-driven reordering or subsetting is
performed.  On success it writes ``raw_input_checksums.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC = ROOT / "data" / "raw"
DEFAULT_CURRENT = ROOT / "data" / "raw"
DEFAULT_CHECKSUMS = ROOT / "reproducibility" / "data" / "raw_input_checksums.sha256"
CONFIG = {
    "human_myeloid": {
        "file": "human_myeloid.h5ad",
        "shape": (3264, 19089),
        "total_counts": 33138858,
        "nnz": 9061521,
    },
    "mouse_pancreas": {
        "file": "mouse_pancreas.h5ad",
        "shape": (2780, 27998),
        "total_counts": 21246800,
        "nnz": 7663432,
    },
    "human_bone_marrow": {
        "file": "human_bone_marrow.h5ad",
        "shape": (7439, 17226),
        "total_counts": 42001306,
        "nnz": 17667278,
    },
}
REQUIRED_PROVENANCE = {
    "source", "accession", "source_file", "selection", "annotation_source", "transformation"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_csr(matrix) -> sp.csr_matrix:
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))


def verify_dataset(name: str, spec: dict, public_dir: Path, current_dir: Path) -> dict:
    public_path = public_dir / spec["file"]
    current_path = current_dir / spec["file"]
    public = ad.read_h5ad(public_path)
    current = ad.read_h5ad(current_path)
    x = as_csr(public.X)
    old_x = as_csr(current.X)

    integer_counts = bool(
        np.issubdtype(x.dtype, np.integer)
        and (x.data.size == 0 or (np.all(x.data >= 0) and np.all(x.data == np.rint(x.data))))
    )
    delta = (x.astype(np.int64) - old_x.astype(np.int64)).tocsr()
    delta.eliminate_zeros()
    different_entries = int(delta.nnz)
    max_abs_difference = int(np.max(np.abs(delta.data))) if delta.nnz else 0
    provenance = public.uns.get("raw_counts_provenance", {})
    missing_provenance = sorted(REQUIRED_PROVENANCE - set(provenance))

    result = {
        "dataset": name,
        "shape": list(public.shape),
        "shape_ok": tuple(public.shape) == tuple(spec["shape"]),
        "integer_counts": integer_counts,
        "cell_count": public.n_obs,
        "cell_set_equal": set(public.obs_names) == set(current.obs_names),
        "cell_order_equal": public.obs_names.equals(current.obs_names),
        "gene_count": public.n_vars,
        "gene_set_equal": set(public.var_names) == set(current.var_names),
        "gene_order_equal": public.var_names.equals(current.var_names),
        "total_counts": int(x.sum()),
        "total_counts_ok": int(x.sum()) == int(spec["total_counts"]),
        "nnz": int(x.nnz),
        "nnz_ok": int(x.nnz) == int(spec["nnz"]),
        "number_of_different_entries": different_entries,
        "max_absolute_difference": max_abs_difference,
        "sha256": sha256_file(public_path),
        "required_provenance_complete": not missing_provenance,
        "missing_provenance_fields": missing_provenance,
        "raw_slot_absent": public.raw is None,
        "layers_absent": len(public.layers) == 0,
        "obsm_absent": len(public.obsm) == 0,
        "obsp_absent": len(public.obsp) == 0,
        "varm_absent": len(public.varm) == 0,
    }
    required_true = [
        "shape_ok", "integer_counts", "cell_set_equal", "cell_order_equal",
        "gene_set_equal", "gene_order_equal", "total_counts_ok", "nnz_ok",
        "required_provenance_complete", "raw_slot_absent", "layers_absent",
        "obsm_absent", "obsp_absent", "varm_absent",
    ]
    result["pass"] = bool(
        all(result[key] for key in required_true)
        and different_entries == 0
        and max_abs_difference == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    args = parser.parse_args()

    results = [verify_dataset(name, spec, args.public_dir, args.current_dir) for name, spec in CONFIG.items()]
    for result in results:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    failures = [result["dataset"] for result in results if not result["pass"]]
    if failures:
        raise SystemExit(f"RAW INPUT VERIFICATION FAILED: {', '.join(failures)}")

    args.checksums.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{result['sha256']}  ../../data/raw/{CONFIG[result['dataset']]['file']}"
        for result in results
    ]
    args.checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ALL PASS; checksums written to {args.checksums}")


if __name__ == "__main__":
    main()
