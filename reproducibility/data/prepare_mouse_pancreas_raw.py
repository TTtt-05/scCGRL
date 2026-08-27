#!/usr/bin/env python
"""Prepare the public mouse_pancreas raw-count H5AD from official GSE132188.

Counts come only from GSM3852755 E15.5 raw Matrix Market data.  Cell selection
and annotations come from the official GSE132188 processed AnnData.  No
CellEnergy file is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import tarfile
import time
import urllib.request
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
from anndata.utils import make_index_unique
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT = ROOT / "data" / "raw" / "mouse_pancreas.h5ad"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "mouse_pancreas.h5ad"
DEFAULT_CACHE = ROOT / "data" / "downloads" / "gse132188"
ADATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132188/suppl/"
    "GSE132188_adata.h5ad.h5"
)
RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3852nnn/GSM3852755/suppl/"
    "GSM3852755_E15_5_counts.tar.gz"
)
ADATA_SIZE = 333_642_690
RAW_SIZE = 122_047_499
ADATA_SHA256 = "3b49358be6a9ba79fa4666bb61a9c3b9158705015d9feb5709ecfba441e80a96"
RAW_SHA256 = "a1a50ca052287415816491b7ac62dcab72420bb5a1968b7101d250c9dcfae7ad"
TARGET_LABELS = (
    "Ngn3 low EP", "Ngn3 high EP", "Fev+", "Alpha", "Beta", "Delta", "Epsilon"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_range(url: str, path: Path, start: int, end: int) -> None:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return
    temporary = path.with_suffix(".download")
    for attempt in range(12):
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if temporary.stat().st_size != expected:
                raise IOError(f"range returned {temporary.stat().st_size}, expected {expected}")
            os.replace(temporary, path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 11:
                raise
            time.sleep(min(5 * (attempt + 1), 30))


def download_ranged(url: str, path: Path, size: int, sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size and sha256_file(path) == sha256:
        return path
    parts = path.parent / f".{path.name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    workers = 8
    chunk = math.ceil(size / workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for index in range(workers):
            start = index * chunk
            end = min(size - 1, (index + 1) * chunk - 1)
            futures.append(executor.submit(_download_range, url, parts / f"part{index:02d}", start, end))
        for future in futures:
            future.result()
    with path.open("wb") as out:
        for part in sorted(parts.glob("part*")):
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=16 * 1024 * 1024)
    shutil.rmtree(parts)
    if path.stat().st_size != size or sha256_file(path) != sha256:
        raise IOError(f"download validation failed for {path.name}")
    return path


def safe_extract(archive: Path, destination: Path) -> Path:
    matrix_dir = destination / "mm10"
    required = [matrix_dir / "matrix.mtx", matrix_dir / "genes.tsv", matrix_dir / "barcodes.tsv"]
    if all(path.exists() for path in required):
        return matrix_dir
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root) + os.sep):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(root)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("official matrix.mtx/genes.tsv/barcodes.tsv were not extracted")
    return matrix_dir


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def stream_fixed_cells(
    matrix_path: Path,
    raw_index_to_output_row: dict[int, int],
    n_genes: int,
    n_raw_cells: int,
    n_output_cells: int,
) -> sp.csr_matrix:
    rows, cols, values = array("i"), array("i"), array("i")
    header_read = False
    with matrix_path.open("r", encoding="ascii", buffering=16 * 1024 * 1024) as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            if not header_read:
                matrix_genes, matrix_cells, _ = map(int, line.split())
                if (matrix_genes, matrix_cells) != (n_genes, n_raw_cells):
                    raise ValueError("official Matrix Market dimensions disagree with genes/barcodes")
                header_read = True
                continue
            gene_one, cell_one, value = map(int, line.split())
            output_row = raw_index_to_output_row.get(cell_one - 1)
            if output_row is not None:
                rows.append(output_row)
                cols.append(gene_one - 1)
                values.append(value)
    result = sp.coo_matrix(
        (np.frombuffer(values, dtype=np.int32),
         (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
        shape=(n_output_cells, n_genes),
        dtype=np.int32,
    ).tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def exact_sparse_difference(left: sp.spmatrix, right: sp.spmatrix) -> tuple[int, int]:
    delta = (left.tocsr().astype(np.int64) - right.tocsr().astype(np.int64)).tocsr()
    delta.eliminate_zeros()
    return int(delta.nnz), int(np.max(np.abs(delta.data))) if delta.nnz else 0


def prepare(current_path: Path, output_path: Path, cache: Path) -> None:
    processed_path = download_ranged(ADATA_URL, cache / "GSE132188_adata.h5ad.h5", ADATA_SIZE, ADATA_SHA256)
    raw_archive = download_ranged(RAW_URL, cache / "GSM3852755_E15_5_counts.tar.gz", RAW_SIZE, RAW_SHA256)
    matrix_dir = safe_extract(raw_archive, cache / "GSM3852755_E15_5_counts_extracted")

    metadata = ad.read_h5ad(processed_path, backed="r")
    official_obs = metadata.obs.copy()
    fixed = official_obs[
        (official_obs["day"].astype(str) == "15.5")
        & official_obs["clusters_fig6_broad_final"].astype(str).isin(TARGET_LABELS)
    ].copy()
    fixed["raw_barcode"] = fixed.index.astype(str).str.replace(r"-3$", "", regex=True)
    if len(fixed) != 2780 or not fixed["raw_barcode"].is_unique:
        raise AssertionError(f"fixed official selection yielded {len(fixed)} cells, expected 2780")

    genes_table = pd.read_csv(matrix_dir / "genes.tsv", sep="\t", header=None, names=["gene_id", "gene_symbol"])
    barcodes = pd.Index(read_lines(matrix_dir / "barcodes.tsv"))
    if len(genes_table) != 27998:
        raise AssertionError(f"official gene count is {len(genes_table)}, expected 27998")
    barcode_positions = barcodes.get_indexer(pd.Index(fixed["raw_barcode"]))
    if np.any(barcode_positions < 0):
        raise KeyError("fixed official E15.5 cells are absent from GSM3852755 raw barcodes")
    raw_to_output = {int(raw_position): output_row for output_row, raw_position in enumerate(barcode_positions)}
    counts = stream_fixed_cells(
        matrix_dir / "matrix.mtx", raw_to_output, len(genes_table), len(barcodes), len(fixed)
    )

    obs = pd.DataFrame(index=pd.Index(fixed["raw_barcode"], name=None))
    obs["day"] = fixed["day"].astype(str).to_numpy()
    obs["clusters_fig6_broad_final"] = fixed["clusters_fig6_broad_final"].astype(str).to_numpy()
    obs["day"] = obs["day"].astype("category")
    obs["clusters_fig6_broad_final"] = obs["clusters_fig6_broad_final"].astype("category")
    gene_names = make_index_unique(pd.Index(genes_table["gene_symbol"].astype(str)), join=".")
    gene_names.name = None
    var = pd.DataFrame(index=gene_names)
    var["gene_id"] = genes_table["gene_id"].astype(str).to_numpy()
    var["gene_symbol"] = genes_table["gene_symbol"].astype(str).to_numpy()

    output = ad.AnnData(X=counts, obs=obs, var=var)
    output.uns["raw_counts_provenance"] = {
        "source": "NCBI GEO GSE132188",
        "accession": "GSE132188",
        "source_file": "GSM3852755_E15_5_counts.tar.gz",
        "raw_count_sample": "GSM3852755",
        "developmental_stage": "E15.5",
        "source_url": RAW_URL,
        "source_sha256": RAW_SHA256,
        "selection": (
            "official day == '15.5' and clusters_fig6_broad_final in "
            "{Ngn3 low EP, Ngn3 high EP, Fev+, Alpha, Beta, Delta, Epsilon}; "
            "remove only combined-object suffix '-3'; retain all 27998 genes in raw order"
        ),
        "annotation_source": "official GSE132188_adata.h5ad.h5 obs",
        "annotation_source_url": ADATA_URL,
        "annotation_source_sha256": ADATA_SHA256,
        "transformation": "none; integer raw counts; Matrix Market genes-by-cells transposed to cells-by-genes",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")
    metadata.file.close()

    current = ad.read_h5ad(current_path)
    written = ad.read_h5ad(output_path)
    n_diff, max_diff = exact_sparse_difference(current.X, written.X)
    if not written.obs_names.equals(current.obs_names):
        raise AssertionError("new/current cell IDs or order differ")
    if not written.var_names.equals(current.var_names):
        raise AssertionError("new/current gene IDs or order differ")
    if n_diff != 0 or max_diff != 0:
        raise AssertionError(f"new/current count mismatch: n_diff={n_diff}, max_diff={max_diff}")
    if set(written.obs["day"].astype(str)) != {"15.5"}:
        raise AssertionError("output day is not the official E15.5 annotation")
    print(
        f"WROTE {output_path}\nshape={written.shape}; total_counts={int(written.X.sum())}; "
        f"nnz={written.X.nnz}; different_entries={n_diff}; max_abs_difference={max_diff}; "
        f"sha256={sha256_file(output_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    prepare(args.current, args.output, args.cache)


if __name__ == "__main__":
    main()
