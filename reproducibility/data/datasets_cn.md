# scCGRL 三个真实数据集的公开 raw-count 输入说明

版本：2026-08-22  
输出目录：仓库内 `data/raw/`

## 1. 整理原则与验证结果

三份公开输入均把未经归一化的非负整数 counts 直接存储在 `AnnData.X`。文件中没有 `normalize_total`、`log1p`、HVG 子集、scale、PCA、邻居图或 UMAP 结果，也没有 `.raw`、`.layers`、`.obsm`、`.obsp` 或 `.varm` 中间对象。现有实验结果、scCGRL 算法及旧 raw 文件均未修改。

| 数据集 | shape | total counts | nnz | 新旧不同元素数 | 最大绝对差 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| human_myeloid | 3,264 × 19,089 | 33,138,858 | 9,061,521 | 0 | 0 | PASS |
| mouse_pancreas | 2,780 × 27,998 | 21,246,800 | 7,663,432 | 0 | 0 | PASS |
| human_bone_marrow | 7,439 × 17,226 | 42,001,306 | 17,667,278 | 0 | 0 | PASS |

上述比较同时要求 cell ID 集合与顺序、gene ID 集合与顺序完全相同。`number_of_different_entries == 0` 是整数 counts 完全一致的必要条件，三个数据集均满足。

## 2. human_myeloid

### Expression counts 来源

- 官方数据：10x Genomics，10k PBMCs from a Healthy Donor（v3 chemistry，Cell Ranger 3.0.0）。
- 数据标识：`pbmc_10k_v3`。
- 官方文件：`pbmc_10k_v3_filtered_feature_bc_matrix.h5`。
- 官方链接：[10x PBMC 10k v3 数据页面](https://www.10xgenomics.com/datasets/10-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0)；[矩阵文件](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5)。
- 官方矩阵 SHA256：`ebc5dedc938830e20f8e1aafb893b0a9e0bf88584f3ef2d00b232dd277a188af`。

### Annotation 来源

最终文件仅保留 scCGRL 使用的 `cluster`。该字段复制自项目现有的 processed human_myeloid study/analysis AnnData；现有项目资料将该 processed 对象追溯至 Figshare DOI `10.6084/m9.figshare.25243225`。该 `cluster` 不是 10x filtered feature-barcode matrix 提供的官方 annotation，因此 provenance 明确将其记录为 `existing study/analysis annotation`，不归因于 10x Genomics。

### 固定选择与变换

- 细胞规则在读取表达值之前固定：仅从当前 3,264 个 cell ID 去除已验证的 `rna_` 前缀，再与 10x barcode 精确匹配。
- 不进行表达值驱动的细胞选择。
- 19,089 个基因按当前文件已经验证的 10x `gene_id` 一对一匹配，并保持当前顺序。
- X 为官方 UMI counts 的直接子集；无数值变换。

最终文件：`human_myeloid_raw_counts.h5ad`；SHA256：`a9677a817edff06fdc20cca53177a1181a88920569d38685810efd4bd5fb6dc6`。

## 3. mouse_pancreas

### Expression counts 来源

- 官方数据集：[NCBI GEO GSE132188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188)。
- raw-count sample：`GSM3852755`，E15.5。
- 官方文件：`GSM3852755_E15_5_counts.tar.gz`。
- 官方下载链接：[GSM3852755 raw counts](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3852nnn/GSM3852755/suppl/GSM3852755_E15_5_counts.tar.gz)。
- 官方文件 SHA256：`a1a50ca052287415816491b7ac62dcab72420bb5a1968b7101d250c9dcfae7ad`。

### Annotation 来源

- annotation 文件：GEO 官方 processed supplementary file `GSE132188_adata.h5ad.h5`。
- 使用字段：官方 `day` 和 `clusters_fig6_broad_final`。
- `day` 保存为官方 developmental annotation `15.5`，不再沿用旧文件中代表合并样本编号的 `3`。
- CellEnergy 未作为 counts 或 annotation 的构建来源。

### 固定选择与变换

在任何 count 比较之前固定以下规则：

1. `day == "15.5"`；
2. `clusters_fig6_broad_final` 属于 `Ngn3 low EP`、`Ngn3 high EP`、`Fev+`、`Alpha`、`Beta`、`Delta`、`Epsilon`；
3. 仅删除官方合并 AnnData cell ID 的 `-3` suffix，以恢复 GSM3852755 原始 10x barcode；
4. 保留 `genes.tsv` 的全部 27,998 个基因及原始顺序。

Matrix Market 文件仅从 genes × cells 转置为 AnnData 的 cells × genes；无其他数值变换。该规则固定得到 2,780 个细胞，不允许为获得匹配而另行删除细胞或基因。

最终文件：`mouse_pancreas_raw_counts.h5ad`；SHA256：`a987b5ac96193ffe6638af826cbe1913dc9e665663113969c6844603045f5285`。

## 4. human_bone_marrow

### Expression counts 来源

- 官方数据集：[NCBI GEO GSE200046](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200046)。
- 官方 supplementary file：`GSE200046_bm_multiome_rna.h5ad`（GEO 下载文件为 `.h5ad.gz`）。
- X 直接复制自官方对象的 `raw.X`，而不是已标准化的主 `.X`。
- 本地官方解压文件 SHA256：`550b84caf5c5acca3b9b22b5916c35f3a567f0d521a4438b052e1f1b8be252dd`。

### Annotation 来源

最终文件保留同一 GEO processed H5AD 的 `celltype`、`sample` 和 `batch`。没有保留 `X_pca`、`X_umap`、`X_FDL`、`MAGIC_imputed_data`、neighbors、SEACell 等历史分析结果。

### 固定选择与变换

- 保留官方对象中的全部 7,439 个细胞和 `raw.X` 的全部 17,226 个变量，顺序不变。
- 官方 deposited `raw.var_names` 为数字标识，最终文件为满足新旧 gene ID/order 精确一致而原样保留；同位置的 processed `var_names` 作为 `var["gene_symbol"]` 保存，不改变 raw matrix 的列身份或顺序。
- `ETP=60`、`BcellPre=154` 均保留在 raw 文件中。
- X 为 `raw.X` 的直接整数复制；不使用官方主 `.X`、PCA、UMAP 或 MAGIC。

最终文件：`human_bone_marrow_raw_counts.h5ad`；SHA256：`e70ceac703239546f6b667d5a2c344c22f125d8c8aac0de7dd93c0b106319ef8`。

## 5. 三个数据集共同使用的 preprocessing

raw 文件生成后，三者使用同一个 scCGRL preprocessing pipeline，参数不得按数据集重新调整：

1. dataset-specific exclusion：human_myeloid 无；mouse_pancreas 无；human_bone_marrow 排除 `ETP` 和 `BcellPre`（7,439 → 7,225）；
2. `normalize_total(target_sum=10000)`；
3. natural `log1p`；
4. `highly_variable_genes(n_top_genes=2000, flavor="seurat")`；
5. `set_raw`；
6. subset to HVGs；
7. `scale(max_value=10)`；
8. PCA：`n_comps=50, svd_solver="arpack", random_state=0`；
9. neighbors：`n_neighbors=15, n_pcs=30, use_rep="X_pca", metric="euclidean", random_state=0`；
10. UMAP：`n_components=3, min_dist=0.3, random_state=0`。

这些步骤属于运行时 preprocessing，不存在于三份 raw_public H5AD 中。

## 6. Redistribution note

本说明记录官方来源、文件、选择规则、annotation 来源和实际变换。10x 表达数据页面与 Figshare annotation 记录均标注为 CC BY 4.0。NCBI 声明 GEO 数据的使用与分发不受 NCBI 限制，同时提示原始提交者可能保留相应权利。仓库内的整合 H5AD 不扩大这些来源条款。权威链接及本次发布边界见仓库根目录 `DATA_LICENSES.md`。
