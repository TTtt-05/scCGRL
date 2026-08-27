# Dataset provenance and preprocessing

The complete bilingual source/annotation/selection descriptions are maintained
in:

- `reproducibility/data/datasets_en.md`
- `reproducibility/data/datasets_cn.md`

## Expression and annotation sources

| Dataset | Expression-count source | Annotation source | Raw shape | Runtime exclusion |
|---|---|---|---:|---|
| human_myeloid | 10x Genomics PBMC 10k v3 filtered feature-barcode matrix | existing study/analysis annotation; not attributed to 10x | 3,264 × 19,089 | none |
| mouse_pancreas | GEO GSE132188, GSM3852755 E15.5 raw counts | GSE132188 processed `GSE132188_adata.h5ad.h5` | 2,780 × 27,998 | none |
| human_bone_marrow | GEO GSE200046 `GSE200046_bm_multiome_rna.h5ad`, `raw.X` | official file `celltype` metadata | 7,439 × 17,226 | ETP (60), BcellPre (154), leaving 7,225 |
| simulation_2 | project PROSSTT-derived H5AD | simulation truth stored with the project data | configured file | none |
| simulation_3 | project PROSSTT-derived H5AD | simulation truth stored with the project data | configured file | none |

The public-ready H5AD `X` matrices are raw integer counts. Licenses and
redistribution permissions are not inferred here; official accessions and the
actual transformation are documented without claiming free redistribution.

## Real-data common preprocessing

1. Dataset-specific exclusion.
2. `normalize_total(target_sum=10000)`.
3. Natural `log1p`.
4. `highly_variable_genes(n_top_genes=2000, flavor="seurat")`.
5. `set_raw`.
6. Subset to HVGs.
7. `scale(max_value=10)`.
8. PCA: 50 components, `arpack`, random state 0.
9. Neighbors: 15 neighbors, 30 PCs, Euclidean, random state 0.
10. UMAP: 3 components, `min_dist=0.3`, random state 0.

Inference uses UMAP dimensions `(0,1,2)`. Myeloid/pancreas figures use `(0,1)`;
bone-marrow figures use `(0,2)`. The simulation recipe remains separately
configured and is not silently replaced by the real-data pipeline.
