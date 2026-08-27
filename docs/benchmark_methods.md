# Final benchmark implementations

All baselines receive the same dataset/cell basis and audited root policy while
retaining their native trajectory algorithms. No method receives terminal
identity, terminal count, true branch count, or true pseudotime. Metrics that
require a native branch or topology output remain `NaN` when unavailable.

## PAGA

Python/Scanpy 1.11.5. Scanpy normalization, HVG, PCA and neighbors are followed
by unsupervised Leiden and PAGA. The shared root is used for DPT ordering, not
to alter the PAGA graph. No terminal prior is provided.

## DPT

Python/Scanpy 1.11.5. Scanpy preprocessing, diffusion map and DPT with
`n_branchings=0`; the shared exact root is stored as `iroot`.

## Palantir

Python/Palantir 1.4.3. PCA, diffusion maps and multiscale space are followed by
`run_palantir`; the shared early cell is supplied and `terminal_states=None`.

## Slingshot

R/Slingshot 2.20.0. Library-size/log normalization, variable genes, PCA,
unsupervised mclust, then Slingshot. The root-containing cluster is
`start.clus`; no end cluster is supplied.

## Monocle 1

R 3.1.3, Bioconductor 3.0 and monocle 1.0.0 in an isolated environment. The
ICA/MST/orderCells workflow is retained. Compatibility helpers only address the
old irlba rank request and igraph P-Q graph assembly.

## Monocle 2

R 4.4.3, monocle 2.40.0, igraph 2.0.3, DDRTree 0.1.6, VGAM 1.1-14 and isolated
dplyr 0.8.5. The root-containing state becomes `root_state`.

## Monocle 3

R/monocle3 1.4.27. The official sequence is `preprocess_cds ->
reduce_dimension -> cluster_cells -> learn_graph -> order_cells`. The exact
root cell is supplied; no terminal prior is used.

## SCORPIUS

R/SCORPIUS 1.0.10. Native dimensionality reduction and trajectory inference
are root-free; the shared root only orients pseudotime after inference.

## TSCAN

R/TSCAN 1.50.0. Sparse-equivalent normalization/filtering, model clustering,
cluster MST, cell mapping and ordering are used. The runner records its CV=0.5
fallback, `clusternum=2-9` search and non-finite-fit recovery explicitly.

## SLICER

R/SLICER 0.2.0. Native gene selection, `select_k`, LLE, KNN graph, cell order
and branch assignment are used. The shared start cell is supplied and no
terminal prior is provided. Timeouts/errors are recorded rather than replaced
with fabricated pseudotime or structure metrics.

Environment definitions are under `benchmark/environments`. Every new run must
record the actual input, software versions, parameters, root-use mode, runtime,
memory, status and error message.
