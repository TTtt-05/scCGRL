# Exact audited Python baseline implementations (PAGA, DPT, Palantir).
import time

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc


def normalize01(values):
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Method returned no finite pseudotime")
    values[~finite] = np.nanmedian(values[finite])
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high <= low else (values - low) / (high - low)


def orient_from_shared_root(values, root_index):
    values = normalize01(values)
    if values[int(root_index)] > 0.5:
        values = 1.0 - values
    return normalize01(values)


def _new_adata(baseline_input):
    matrix = baseline_input["expression"]
    matrix = matrix.copy() if sp.issparse(matrix) else np.asarray(matrix).copy()
    result = ad.AnnData(X=matrix)
    result.obs_names = baseline_input["cell_ids"]
    result.var_names = pd.Index(baseline_input["gene_ids"]).astype(str).to_numpy()
    result.var_names_make_unique()
    return result


def _scanpy_recommended_preprocess(baseline_input, seed):
    """Scanpy tutorial-style expression preprocessing on the fixed cells."""
    adata = _new_adata(baseline_input)
    audit = []
    if baseline_input["expression_is_counts"]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        audit.extend(["normalize_total(target_sum=1e4)", "log1p"])
    else:
        if np.nanmin(adata.X.data if sp.issparse(adata.X) else adata.X) < 0:
            raise ValueError("Official baseline preprocessing requires nonnegative expression")
        audit.append("input_already_nonnegative_normalized;no_second_log")
    if adata.n_vars > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=True)
        audit.append("highly_variable_genes(flavor=seurat,n_top_genes=2000)")
    sc.pp.scale(adata, max_value=10)
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", random_state=int(seed))
    audit.extend(["scale(max_value=10)", f"pca(n_comps={n_comps},svd_solver=arpack)"])
    return adata, audit


def _leiden(adata, seed):
    sc.pp.neighbors(adata, random_state=int(seed))
    sc.tl.leiden(adata, resolution=1.0, random_state=int(seed), key_added="baseline_leiden")
    return adata.obs["baseline_leiden"].astype(str).to_numpy()


def _paga_leaf_branches(adata, clusters, root_index):
    matrix = adata.uns["paga"]["connectivities"]
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[1]):
            weight = float(matrix[left, right])
            if weight > 0:
                graph.add_edge(left, right, weight=weight)
    root_cluster = int(clusters[int(root_index)])
    if graph.number_of_edges() == 0:
        return np.repeat("paga_single", len(clusters)), []
    tree = nx.maximum_spanning_tree(graph, weight="weight")
    component = nx.node_connected_component(tree, root_cluster)
    tree = tree.subgraph(component).copy()
    leaves = [node for node, degree in tree.degree() if degree <= 1 and node != root_cluster]
    if not leaves:
        leaves = [max(tree, key=lambda node: nx.shortest_path_length(tree, root_cluster, node))]
    assignments = {}
    for node in tree:
        assignments[node] = min(leaves, key=lambda leaf: nx.shortest_path_length(tree, node, leaf))
    branches = np.asarray([
        f"paga_leaf_{assignments.get(int(cluster), int(cluster))}" for cluster in clusters
    ])
    return branches, leaves


def run_paga(baseline_input, seed, root_index):
    started = time.perf_counter()
    adata, audit = _scanpy_recommended_preprocess(baseline_input, seed)
    clusters = _leiden(adata, seed)
    preprocessing_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    sc.tl.paga(adata, groups="baseline_leiden")
    sc.tl.diffmap(adata)
    adata.uns["iroot"] = int(root_index)
    sc.tl.dpt(adata)
    pseudotime = orient_from_shared_root(adata.obs["dpt_pseudotime"], root_index)
    branches, leaves = _paga_leaf_branches(adata, clusters, root_index)
    return {
        "pseudotime": pseudotime,
        "branches": branches,
        "lineage_pseudotime": {},
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": time.perf_counter() - inference_started,
        "parameters": {
            "workflow": audit + ["neighbors(defaults)", "leiden(resolution=1.0)", "paga(defaults)", "diffmap+dpt"],
            "paga_leaf_clusters": leaves,
            "root_usage": "DPT iroot; PAGA topology itself is root-free",
        },
        "has_native_branches": True,
        "has_native_topology": True,
    }


def run_dpt(baseline_input, seed, root_index):
    started = time.perf_counter()
    adata, audit = _scanpy_recommended_preprocess(baseline_input, seed)
    sc.pp.neighbors(adata, random_state=int(seed))
    preprocessing_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    sc.tl.diffmap(adata)
    adata.uns["iroot"] = int(root_index)
    sc.tl.dpt(adata, n_branchings=0)
    pseudotime = orient_from_shared_root(adata.obs["dpt_pseudotime"], root_index)
    return {
        "pseudotime": pseudotime,
        "branches": np.repeat("dpt_trajectory", adata.n_obs),
        "lineage_pseudotime": {},
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": time.perf_counter() - inference_started,
        "parameters": {
            "workflow": audit + ["neighbors(defaults)", "diffmap(defaults)", "dpt(n_branchings=0)"],
            "root_usage": "exact shared root as iroot",
            "branch_note": "DPT returned global ordering only; no branch labels were invented",
        },
        "has_native_branches": False,
        "has_native_topology": False,
    }


def run_palantir(baseline_input, seed, root_index):
    import palantir

    started = time.perf_counter()
    adata, audit = _scanpy_recommended_preprocess(baseline_input, seed)
    pca = pd.DataFrame(np.asarray(adata.obsm["X_pca"]), index=adata.obs_names)
    diffusion = palantir.utils.run_diffusion_maps(pca, seed=int(seed))
    multiscale = palantir.utils.determine_multiscale_space(diffusion)
    preprocessing_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    result = palantir.core.run_palantir(
        multiscale,
        early_cell=str(adata.obs_names[int(root_index)]),
        terminal_states=None,
        seed=int(seed),
    )
    pseudotime = pd.Series(result.pseudotime).reindex(adata.obs_names).to_numpy(float)
    pseudotime = orient_from_shared_root(pseudotime, root_index)
    fate = getattr(result, "branch_probs", None)
    if fate is None:
        fate = getattr(result, "fate_probabilities", None)
    lineage = {}
    if fate is not None and np.asarray(fate).ndim == 2 and np.asarray(fate).shape[1]:
        fate = pd.DataFrame(fate, index=getattr(fate, "index", adata.obs_names)).reindex(adata.obs_names)
        branches = fate.columns[np.nanargmax(fate.to_numpy(float), axis=1)].astype(str).to_numpy()
        for name in fate.columns:
            lineage[str(name)] = np.where(fate[str(name)].to_numpy(float) > 0, pseudotime, np.nan)
    else:
        branches = np.repeat("palantir_trajectory", adata.n_obs)
    return {
        "pseudotime": pseudotime,
        "branches": branches,
        "lineage_pseudotime": lineage,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": time.perf_counter() - inference_started,
        "parameters": {
            "workflow": audit + ["run_diffusion_maps(defaults)", "determine_multiscale_space", "run_palantir(defaults_except_seed)"],
            "root_usage": "exact shared cell as early_cell",
            "terminal_states_supplied": False,
            "random_seed": int(seed),
        },
        "has_native_branches": bool(fate is not None and np.asarray(fate).ndim == 2 and np.asarray(fate).shape[1]),
        "has_native_topology": bool(fate is not None and np.asarray(fate).ndim == 2 and np.asarray(fate).shape[1]),
    }


RUNNERS = {"PAGA": run_paga, "DPT": run_dpt, "Palantir": run_palantir}


def run_python_method(method, baseline_input, seed, root_index):
    return RUNNERS[method](baseline_input, int(seed), int(root_index))
