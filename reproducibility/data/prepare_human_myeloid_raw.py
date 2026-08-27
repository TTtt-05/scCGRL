#!/usr/bin/env python
"""Prepare the public human_myeloid integer-count H5AD.

Expression counts are selected directly from the official 10x Genomics PBMC
10k v3 filtered feature-barcode matrix.  The selection is defined solely by
the already fixed current cell barcodes and gene IDs.  Expression values are
never inspected to choose cells or genes.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "downloads" / "pbmc_10k_v3_filtered_feature_bc_matrix.h5"
DEFAULT_CURRENT = ROOT / "data" / "raw" / "human_myeloid.h5ad"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "human_myeloid.h5ad"
SOURCE_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/"
    "pbmc_10k_v3_filtered_feature_bc_matrix.h5"
)
EXPECTED_SOURCE_SHA256 = "ebc5dedc938830e20f8e1aafb893b0a9e0bf88584f3ef2d00b232dd277a188af"


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
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("official 10x source SHA-256 does not match the verified file")

    current = ad.read_h5ad(current_path)
    official = sc.read_10x_h5(source, gex_only=True)
    if current.shape != (3264, 19089):
        raise ValueError(f"unexpected current shape: {current.shape}")

    # This is the only allowed barcode rewrite.  It is applied to identifiers,
    # not chosen conditionally from expression values.
    current_ids = pd.Index(current.obs_names.astype(str))
    if not bool(current_ids.str.startswith("rna_").all()):
        raise ValueError("not every current barcode has the verified rna_ prefix")
    source_barcodes = current_ids.str.removeprefix("rna_")
    official_cells = pd.Index(official.obs_names.astype(str))
    if not official_cells.is_unique:
        raise ValueError("official 10x cell barcodes are not unique")
    cell_positions = official_cells.get_indexer(source_barcodes)
    if np.any(cell_positions < 0):
        missing = source_barcodes[cell_positions < 0]
        raise KeyError(f"{len(missing)} fixed current barcodes are absent from official 10x data")

    if "source_gene_ids" not in current.var:
        raise KeyError("current raw file lacks the previously verified source_gene_ids field")
    current_gene_ids = pd.Index(current.var["source_gene_ids"].astype(str))
    official_gene_ids = pd.Index(official.var["gene_ids"].astype(str))
    if not official_gene_ids.is_unique or not current_gene_ids.is_unique:
        raise ValueError("gene IDs must be unique for exact identifier matching")
    gene_positions = official_gene_ids.get_indexer(current_gene_ids)
    if np.any(gene_positions < 0):
        missing = current_gene_ids[gene_positions < 0]
        raise KeyError(f"{len(missing)} fixed current genes are absent from official 10x data")

    counts = official.X[cell_positions][:, gene_positions].tocsr().astype(np.int32)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if np.any(counts.data < 0):
        raise ValueError("negative values detected in official UMI matrix")

    obs = current.obs.loc[:, ["cluster"]].copy()
    obs["cluster"] = obs["cluster"].astype(str).astype("category")
    var = pd.DataFrame(index=current.var_names.copy())
    var["gene_id"] = current_gene_ids.to_numpy()
    var["gene_symbol"] = official.var_names[gene_positions].astype(str).to_numpy()
    for old, new in (("feature_types", "feature_type"), ("genome", "genome")):
        if old in official.var:
            var[new] = official.var.iloc[gene_positions][old].astype(str).to_numpy()

    output = ad.AnnData(X=counts, obs=obs, var=var)
    output.uns["raw_counts_provenance"] = {
        "source": "10x Genomics PBMC 10k v3 filtered feature-barcode matrix",
        "accession": "pbmc_10k_v3",
        "source_file": "pbmc_10k_v3_filtered_feature_bc_matrix.h5",
        "source_url": SOURCE_URL,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "selection": (
            "Fixed 3264 current cell IDs matched after removing only the verified rna_ prefix; "
            "fixed 19089 genes matched by official 10x gene_id in current order"
        ),
        "annotation_source": (
            "cluster copied from the existing scCGRL study/analysis AnnData; "
            "not supplied by the 10x expression-count file"
        ),
        "transformation": "none; direct subset of original nonnegative integer UMI counts",
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
    print(
        f"WROTE {output_path}\nshape={written.shape}; total_counts={int(written.X.sum())}; "
        f"nnz={written.X.nnz}; different_entries={n_diff}; max_abs_difference={max_diff}; "
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
