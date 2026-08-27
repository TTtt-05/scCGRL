#!/usr/bin/env python
"""Prepare the public human_bone_marrow raw-count H5AD from GSE200046.

The matrix is copied directly from ``GSE200046_bm_multiome_rna.h5ad.raw.X``.
ETP and BcellPre cells remain in this raw input and are excluded only by the
shared downstream preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "downloads" / "GSE200046_bm_multiome_rna.h5ad"
DEFAULT_CURRENT = ROOT / "data" / "raw" / "human_bone_marrow.h5ad"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "human_bone_marrow.h5ad"
SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200046/suppl/"
    "GSE200046_bm_multiome_rna.h5ad.gz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sparse_difference(left: sp.spmatrix, right: sp.spmatrix) -> tuple[int, int]:
    delta = (left.tocsr().astype(np.int64) - right.tocsr().astype(np.int64)).tocsr()
    delta.eliminate_zeros()
    return int(delta.nnz), int(np.max(np.abs(delta.data))) if delta.nnz else 0


def prepare(source: Path, current_path: Path, output_path: Path) -> None:
    official = ad.read_h5ad(source)
    if official.raw is None:
        raise ValueError("official GSE200046 object has no raw slot")
    if official.raw.shape != (7439, 17226):
        raise ValueError(f"unexpected official raw shape: {official.raw.shape}")
    current = ad.read_h5ad(current_path)

    raw_counts = official.raw.X.tocsr()
    if np.any(raw_counts.data < 0) or not np.all(raw_counts.data == np.rint(raw_counts.data)):
        raise ValueError("official raw.X is not a nonnegative integer-count matrix")
    counts = raw_counts.astype(np.int32)
    counts.sum_duplicates()
    counts.eliminate_zeros()

    # raw.var_names in the deposited GEO object are stable numeric identifiers.
    # The processed X has the same number/order of variables and supplies readable
    # gene symbols, which are retained as metadata without changing raw gene IDs.
    raw_gene_ids = official.raw.var_names.copy()
    if len(raw_gene_ids) != len(official.var_names):
        raise ValueError("official raw/main variables cannot be mapped positionally")
    obs = official.obs.loc[:, ["sample", "batch", "celltype"]].copy()
    var = pd.DataFrame(index=raw_gene_ids)
    var["gene_symbol"] = official.var_names.astype(str).to_numpy()

    output = ad.AnnData(X=counts, obs=obs, var=var)
    output.uns["raw_counts_provenance"] = {
        "source": "NCBI GEO GSE200046",
        "accession": "GSE200046",
        "source_file": "GSE200046_bm_multiome_rna.h5ad",
        "source_url": SOURCE_URL,
        "source_sha256_uncompressed_local_copy": sha256_file(source),
        "selection": (
            "all 7439 cells and all 17226 raw variables in deposited order; "
            "ETP (60) and BcellPre (154) retained"
        ),
        "annotation_source": (
            "celltype, sample and batch copied from the same official GEO processed H5AD obs"
        ),
        "transformation": "none; direct integer copy of GSE200046_bm_multiome_rna.h5ad raw.X",
        "gene_identifier_note": (
            "raw.var_names are the deposited numeric identifiers; processed var_names are "
            "retained positionally in var['gene_symbol']"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")

    written = ad.read_h5ad(output_path)
    n_diff, max_diff = exact_sparse_difference(current.X, written.X)
    if not written.obs_names.equals(current.obs_names):
        raise AssertionError("new/current cell IDs or order differ")
    if not written.var_names.equals(current.var_names):
        raise AssertionError("new/current gene IDs or order differ")
    if n_diff != 0 or max_diff != 0:
        raise AssertionError(f"new/current count mismatch: n_diff={n_diff}, max_diff={max_diff}")
    excluded = written.obs["celltype"].astype(str).isin(["ETP", "BcellPre"])
    if int(excluded.sum()) != 214 or int((~excluded).sum()) != 7225:
        raise AssertionError("ETP/BcellPre counts do not match the fixed preprocessing rule")
    print(
        f"WROTE {output_path}\nshape={written.shape}; total_counts={int(written.X.sum())}; "
        f"nnz={written.X.nnz}; ETP_BcellPre={int(excluded.sum())}; "
        f"different_entries={n_diff}; max_abs_difference={max_diff}; "
        f"sha256={sha256_file(output_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.source, args.current, args.output)


if __name__ == "__main__":
    main()
