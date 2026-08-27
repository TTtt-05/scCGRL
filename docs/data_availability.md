# Data availability and redistribution scope

This release contains the three verified real-data raw-count H5AD files under
`data/raw` and all five configured processed H5AD inputs under
`data/processed`. Their repository-relative checksums are recorded in
`data/checksums.sha256` and
`data/processed/processed_inputs_checksums.sha256`.

| Dataset | Expression source | Annotation source | Raw artifact | Processed artifact |
|---|---|---|---|---|
| human_myeloid | 10x Genomics PBMC 10k v3 filtered feature-barcode matrix | Existing study/analysis annotation; not attributed to 10x | `data/raw/human_myeloid.h5ad` | `data/processed/human_myeloid.h5ad` |
| mouse_pancreas | GEO GSE132188, GSM3852755 E15.5 raw counts | GSE132188 processed `GSE132188_adata.h5ad.h5` | `data/raw/mouse_pancreas.h5ad` | `data/processed/mouse_pancreas.h5ad` |
| human_bone_marrow | GEO GSE200046 `GSE200046_bm_multiome_rna.h5ad`, `raw.X` | Official `celltype` metadata | `data/raw/human_bone_marrow.h5ad` | `data/processed/human_bone_marrow.h5ad` |
| simulation_2 | PROSSTT-derived project simulation | Project simulation truth | not packaged as a raw-count H5AD | `data/processed/simulation_2.h5ad` |
| simulation_3 | PROSSTT-derived project simulation | Project simulation truth | not packaged as a raw-count H5AD | `data/processed/simulation_3.h5ad` |

The exact official downloads, fixed selection rules, annotation sources,
transformations, and matrix-verification procedures are documented under
`reproducibility/data`. Expression sources and annotation sources are reported
separately because annotations are not always supplied by the count-matrix
provider.

The 10x expression source and the Figshare annotation record are CC BY 4.0.
NCBI states that it places no restrictions on use or distribution of GEO data,
while warning that submitters may assert rights in submitted material. Local
packaging does not broaden those source terms. The source-specific boundary and
authoritative links are recorded in `DATA_LICENSES.md`.
