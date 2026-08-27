# Dataset licenses, provenance, and redistribution boundary

The repository code is licensed under the MIT License. Data are not relicensed
by that file. Each packaged input retains the terms of its expression and
annotation sources.

| Dataset | Expression source | Annotation source | Public-use basis | Packaged artifact |
|---|---|---|---|---|
| `human_myeloid` | 10x Genomics PBMC 10k v3 filtered feature-barcode matrix | Figshare DOI `10.6084/m9.figshare.25243225.v1` study/analysis annotation | Both source pages specify CC BY 4.0; attribution remains required | verified raw and processed H5AD |
| `mouse_pancreas` | NCBI GEO GSE132188, GSM3852755 E15.5 | GSE132188 processed supplementary H5AD | GEO states that NCBI places no restrictions on use or distribution, while warning that submitters may retain rights | verified raw and processed H5AD |
| `human_bone_marrow` | NCBI GEO GSE200046 processed supplementary H5AD, `raw.X` | metadata in the same official H5AD | GEO states that NCBI places no restrictions on use or distribution, while warning that submitters may retain rights | verified raw and processed H5AD |
| `simulation_2` | project-generated with PROSSTT | project simulation truth | project-generated derived data, released with this repository for reproducibility | processed H5AD |
| `simulation_3` | project-generated with PROSSTT | project simulation truth | project-generated derived data, released with this repository for reproducibility | processed H5AD |

Authoritative pages:

- 10x PBMC 10k v3 dataset and license: <https://www.10xgenomics.com/datasets/10-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0>
- Figshare annotation record: <https://doi.org/10.6084/m9.figshare.25243225.v1>
- GSE132188: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188>
- GSE200046: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200046>
- GEO disclaimer: <https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html>

The H5AD files are reproducibility subsets/derivatives rather than replacements
for the source records. Users must cite the original studies and comply with
any terms asserted by source submitters or publishers. The exact selection and
matrix-equality checks are documented under `reproducibility/data/`.
