# Public raw-count inputs for the three empirical scCGRL datasets

Version: 2026-08-22  
Output directory: repository-local `data/raw/`

## 1. Preparation and exact verification

Each public input stores unnormalized nonnegative integer counts directly in `AnnData.X`. No normalize/log/HVG/scale/PCA/neighbors/UMAP result is embedded, and `.raw`, `.layers`, `.obsm`, `.obsp`, and `.varm` are absent. Existing experimental results, the scCGRL algorithm, and the previous raw files were not modified.

| Dataset | Shape | Total counts | nnz | Different entries | Maximum absolute difference | Result |
|---|---:|---:|---:|---:|---:|---|
| human_myeloid | 3,264 × 19,089 | 33,138,858 | 9,061,521 | 0 | 0 | PASS |
| mouse_pancreas | 2,780 × 27,998 | 21,246,800 | 7,663,432 | 0 | 0 | PASS |
| human_bone_marrow | 7,439 × 17,226 | 42,001,306 | 17,667,278 | 0 | 0 | PASS |

Exact verification additionally required identical cell-ID and gene-ID sets and order. For integer counts, complete matrix equality was accepted only when `number_of_different_entries == 0`.

## 2. human_myeloid

**Expression source.** The counts come from the official 10x Genomics PBMC 10k v3 filtered feature-barcode matrix (`pbmc_10k_v3_filtered_feature_bc_matrix.h5`; [dataset page](https://www.10xgenomics.com/datasets/10-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0); [matrix](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5)). Source SHA256: `ebc5dedc938830e20f8e1aafb893b0a9e0bf88584f3ef2d00b232dd277a188af`.

**Annotation source.** Only `cluster`, which scCGRL uses, is retained. It was copied from the existing processed human_myeloid study/analysis AnnData, traced in the project records to Figshare DOI `10.6084/m9.figshare.25243225`. It is not an annotation supplied by the 10x expression-count file and is explicitly described as an existing study/analysis annotation.

**Selection.** The fixed 3,264 cell IDs were matched after removing only the validated `rna_` prefix. No expression-dependent cell selection was permitted. The 19,089 genes were matched one-to-one by official 10x `gene_id` and retained in current order. X is a direct count subset with no numerical transformation.

Output SHA256: `a9677a817edff06fdc20cca53177a1181a88920569d38685810efd4bd5fb6dc6`.

## 3. mouse_pancreas

**Expression source.** Counts come from [NCBI GEO GSE132188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188), raw-count sample GSM3852755 (E15.5), file `GSM3852755_E15_5_counts.tar.gz`. Source SHA256: `a1a50ca052287415816491b7ac62dcab72420bb5a1968b7101d250c9dcfae7ad`.

**Annotation source.** `day` and `clusters_fig6_broad_final` come from the official processed supplementary file `GSE132188_adata.h5ad.h5`. `day` is the official developmental annotation `15.5`, not the old merged-sample index `3`. No CellEnergy file is used to build counts or annotations.

**Fixed selection.** Before count comparison, the rule was fixed as: official `day == "15.5"`; broad label in Ngn3 low EP, Ngn3 high EP, Fev+, Alpha, Beta, Delta, or Epsilon; remove only the combined-object `-3` suffix to recover GSM3852755 barcodes; retain all 27,998 genes from `genes.tsv` in original order. The only matrix operation is genes-by-cells to cells-by-genes transposition.

Output SHA256: `a987b5ac96193ffe6638af826cbe1913dc9e665663113969c6844603045f5285`.

## 4. human_bone_marrow

**Expression source.** Counts come from [NCBI GEO GSE200046](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200046), supplementary file `GSE200046_bm_multiome_rna.h5ad`, and are copied directly from its `raw.X`, not from its normalized main `.X`. SHA256 of the local uncompressed official file: `550b84caf5c5acca3b9b22b5916c35f3a567f0d521a4438b052e1f1b8be252dd`.

**Annotation source.** `celltype`, `sample`, and `batch` come from the same official GEO processed H5AD. Historical PCA, UMAP, FDL, MAGIC, neighbors, and SEACell results are not retained.

**Selection.** All 7,439 cells and all 17,226 raw variables remain in deposited order. The deposited `raw.var_names` are numeric identifiers and are preserved to maintain exact old/new gene identity and order; processed variable names are retained positionally in `var["gene_symbol"]`. ETP (60) and BcellPre (154) remain in the raw file and are removed only during dataset-specific preprocessing.

Output SHA256: `e70ceac703239546f6b667d5a2c344c22f125d8c8aac0de7dd93c0b106319ef8`.

## 5. Common preprocessing

After loading the raw inputs, all three datasets use the same pipeline: (1) dataset-specific exclusion—none for human_myeloid and mouse_pancreas; ETP and BcellPre for human_bone_marrow (7,439 to 7,225); (2) `normalize_total(target_sum=10000)`; (3) natural `log1p`; (4) `highly_variable_genes(n_top_genes=2000, flavor="seurat")`; (5) `set_raw`; (6) subset to HVGs; (7) `scale(max_value=10)`; (8) PCA with `n_comps=50, svd_solver="arpack", random_state=0`; (9) neighbors with `n_neighbors=15, n_pcs=30, use_rep="X_pca", metric="euclidean", random_state=0`; and (10) UMAP with `n_components=3, min_dist=0.3, random_state=0`.

These are runtime preprocessing steps and are not stored in the raw_public H5AD files.

## 6. Redistribution note

This document records source files, selection, annotation provenance, and
transformations. The 10x expression source and the Figshare annotation record
both specify CC BY 4.0. NCBI states that it places no restrictions on GEO data
use or distribution, while warning that submitters may retain rights. The
integrated H5AD files do not broaden those source terms. See the repository
root `DATA_LICENSES.md` for authoritative links and the release boundary.
