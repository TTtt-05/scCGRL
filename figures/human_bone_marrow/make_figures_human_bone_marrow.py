"""Generate the canonical human bone marrow publication figure."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("SCCGRL_PROJECT_ROOT", REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_sccgrl import run_dataset

DATASET_KEY = "human_bone_marrow"
SOURCE_NOTEBOOK = "2026-08-17_scCGRL_five_datasets_v1.ipynb"
SOURCE_CELL_INDEX = 64


def activate_big_figure_dataset(run_output_root, seed=42, episodes=10000):
    """Run the canonical pipeline once and expose the figure inputs."""
    bundle = run_dataset(
        DATASET_KEY,
        output_root=run_output_root,
        project_root=PROJECT_ROOT,
        seed=int(seed),
        episodes=int(episodes),
        runs=1,
        save_processed=True,
    )
    return (
        bundle["adata"], bundle["q_learner"], bundle["dijkstra_paths"],
        bundle["result"], bundle["results"],
    )


def make_figures(output_dir, run_output_root, seed=42, episodes=10000):
    """Render the original bone-marrow mixed-modality panel layout."""
    adata, q_learner, dijkstra_paths, result, results = (
        activate_big_figure_dataset(run_output_root, seed=seed, episodes=episodes)
    )
    big_output_dir = Path(output_dir)
    big_output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl
    import seaborn as sns
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import f as f_dist
    from patsy import dmatrix, build_design_matrices

    # ============================================================
    # Biological-panel settings: STRICTLY matched to human_myeloid
    # ============================================================
    BIO_PT_SIGMA = 0.08
    BIO_GRID_SIZE = 100
    DYNAMIC_TOP_N = 80
    DYNAMIC_SPLINE_DF = 5
    DYNAMIC_FDR_REPORT = 0.05
    DYNAMIC_MIN_DETECTION = 0.10
    DYNAMIC_MIN_SD = 1e-4

    # ==========================================
    # Global plotting style
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

    START_MARKER = '*'
    START_SIZE = 250
    START_COLOR = 'white'
    START_EDGE = 'black'
    END_MARKER = 'X'
    END_SIZE = 120

    cluster_key = "celltype" 

    # Extract model variables.
    umap_coords = adata.obsm['X_umap'][:,[0,2]]
    cell_types_arr = np.array(adata.obs[cluster_key].astype(str))
    start_idx = result['start_index']
    ends_idx = result['endpoint_indices']
    global_pt = results['rf_pseudotime']
    trajectory_global_pt = np.asarray(results['global_pseudotime'], dtype=float)

    # ==========================================
    # Fixed terminal lineages and colors.
    # ==========================================
    TARGET_LINEAGES = ['Ery', 'Mono', 'Mono2', 'DC']
    EXPLICIT_COLORS = {
        'Ery': '#f9736e',
        'Mono': '#95CAE9',
        'Mono2': '#53A85F',
        'DC': '#8E7CC3',
    }

    unique_cell_types = np.unique(cell_types_arr)
    global_palette = sns.color_palette("Set2", len(unique_cell_types))
    GLOBAL_COLOR_MAP = {ctype: mpl.colors.to_hex(color) for ctype, color in zip(unique_cell_types, global_palette)}
    GLOBAL_COLOR_MAP.update(EXPLICIT_COLORS)

    # ==========================================
    # Path smoothing and pseudotime normalization.
    # ==========================================
    def extract_and_smooth_tree(paths_dict, coords, num_points=100, sigma=3, smooth=False):
        if paths_dict is None or len(paths_dict) == 0: return [], [], []
        clean_paths = {k: (v['path'] if isinstance(v, dict) else v) for k, v in paths_dict.items()}
        edges, leaves = set(), set()
        for path in clean_paths.values():
            for i in range(len(path) - 1): edges.add((path[i], path[i+1]))
            if len(path) > 0: leaves.add(path[-1])
        out_degree = {u: 0 for u, v in edges}
        for u, v in edges: out_degree[u] += 1

        segments, terminal_flags, covered_edges = [], [], set()
        for path in clean_paths.values():
            if not path: continue
            curr_seg = [path[0]]
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if (u, v) not in covered_edges:
                    covered_edges.add((u, v))
                    curr_seg.append(v)
                    if out_degree.get(v, 0) != 1 or v in leaves:
                        segments.append(curr_seg); terminal_flags.append(v in leaves); curr_seg = [v]
                else: curr_seg = [v]

        res_segs = []
        for nodes in segments:
            raw = coords[nodes]
            valid_idx = [0]
            for i in range(1, len(raw)):
                if np.linalg.norm(raw[i] - raw[valid_idx[-1]]) > 1e-4: valid_idx.append(i)
            uni = raw[valid_idx]
            if len(uni) < 3 or not smooth:
                res_segs.append(uni); continue
            diffs = np.diff(uni, axis=0)
            arc = np.concatenate(([0], np.cumsum(np.linalg.norm(diffs, axis=1))))
            if arc[-1] > 0: arc /= arc[-1]
            u_new = np.linspace(0, 1, num_points)
            smoothed = []
            for d in range(coords.shape[1]):
                f = interp1d(arc, uni[:, d], kind='linear')
                ds = gaussian_filter1d(f(u_new), sigma=sigma)
                ds[0], ds[-1] = uni[0, d], uni[-1, d]
                smoothed.append(ds)
            res_segs.append(np.column_stack(smoothed))

        return res_segs, terminal_flags, segments

    def normalize_pt_array(pt_array):
        pt_norm = np.copy(pt_array).astype(float)
        mask = (pt_norm >= 0) & np.isfinite(pt_norm)
        if np.any(mask):
            pmin, pmax = np.min(pt_norm[mask]), np.max(pt_norm[mask])
            if pmax > pmin: pt_norm[mask] = (pt_norm[mask] - pmin) / (pmax - pmin)
        return pt_norm


    def nw_gaussian_curve(pt, expr, sigma_pt=BIO_PT_SIGMA, grid_size=BIO_GRID_SIZE):
        """Same trajectory-wise Gaussian-kernel NW smoother as human_myeloid."""
        pt = np.asarray(pt, dtype=float)
        expr = np.asarray(expr, dtype=float)

        valid = np.isfinite(pt) & np.isfinite(expr) & (pt >= 0)
        pt, expr = np.clip(pt[valid], 0, 1), expr[valid]

        if len(pt) < 3 or np.ptp(pt) < 1e-8:
            return None, None

        grid = np.linspace(pt.min(), pt.max(), grid_size)
        weights = np.exp(-0.5 * ((grid[:, None] - pt[None, :]) / sigma_pt) ** 2)
        denom = weights.sum(axis=1)
        smooth = np.divide(
            weights @ expr,
            denom,
            out=np.full(grid.shape, np.nan),
            where=denom > 1e-12,
        )
        return grid, smooth


    def get_normalized_expression_source(a):
        """Same expression-source priority as human_myeloid."""
        if a.raw is not None:
            return a.raw.X, np.asarray(a.raw.var_names.astype(str)), "adata.raw.X"
        return a.X, np.asarray(a.var_names.astype(str)), "adata.X"


    BIO_EXPR_X, BIO_EXPR_GENES, BIO_EXPR_SOURCE = get_normalized_expression_source(adata)
    BIO_GENE_IDX = {g: i for i, g in enumerate(BIO_EXPR_GENES)}


    def get_bio_gene_expression(gene):
        i = BIO_GENE_IDX.get(gene)
        if i is None:
            return None
        x = BIO_EXPR_X[:, i]
        return x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()


    def bh_fdr(p):
        """Benjamini-Hochberg FDR correction; identical logic to human_myeloid."""
        p = np.asarray(p, float)
        q = np.full(p.shape, np.nan)
        ok = np.isfinite(p)
        if not ok.any():
            return q

        x = p[ok]
        order = np.argsort(x)
        ranked = x[order]
        m = len(x)
        adj = np.minimum.accumulate(
            (ranked * m / np.arange(1, m + 1))[::-1]
        )[::-1]

        back = np.empty_like(adj)
        back[order] = np.clip(adj, 0, 1)
        q[ok] = back
        return q


    def prepare_spline_heatmap(path_cells, marker_genes=(), top_n=DYNAMIC_TOP_N):
        """
        STRICT human_myeloid heatmap method:
          1) use only cells assigned to one inferred scCGRL trajectory;
          2) expression source prefers adata.raw.X;
          3) retain genes with detection >=10% and SD >=1e-4;
          4) fit centered natural cubic spline expression ~ pseudotime, df=5;
          5) compare full spline with intercept-only model by F test;
          6) apply Benjamini-Hochberg FDR and retain q<=0.05;
          7) rank significant genes by spline R^2 and keep at most Top80;
          8) evaluate selected spline fits on a 100-point pseudotime grid;
          9) min-max scale each fitted gene independently to [0,1];
         10) order genes by fitted peak pseudotime.

        Marker lists are annotation-only and NEVER alter statistical selection.
        """
        cells = np.asarray(path_cells, dtype=int)
        pt = np.asarray(ef_pt_norm[cells], float)

        valid = np.isfinite(pt) & (pt >= 0)
        cells, pt = cells[valid], np.clip(pt[valid], 0, 1)

        if len(cells) < 20 or np.ptp(pt) < 1e-8:
            return None

        X = BIO_EXPR_X[cells, :]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))

        det = np.asarray((X > 0).sum(axis=0)).ravel() / len(cells)
        mean = np.asarray(X.mean(axis=0)).ravel()
        sd = np.sqrt(
            np.maximum(
                np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean**2,
                0,
            )
        )

        eligible = np.where(
            np.isfinite(sd)
            & (sd >= DYNAMIC_MIN_SD)
            & (det >= DYNAMIC_MIN_DETECTION)
        )[0]

        if len(eligible) < 1:
            return None

        Y = X[:, eligible].toarray()

        design_df = dmatrix(
            f"cr(pt, df={DYNAMIC_SPLINE_DF}, constraints='center')",
            {"pt": pt},
            return_type="dataframe",
        )

        D = np.asarray(design_df, float)
        rank_full = np.linalg.matrix_rank(D)
        df_num = rank_full - 1
        df_den = len(pt) - rank_full

        if df_num <= 0 or df_den <= 0:
            return None

        beta = np.linalg.pinv(D) @ Y
        fitted = D @ beta

        sse1 = np.sum((Y - fitted) ** 2, axis=0)
        sse0 = np.sum(
            (Y - Y.mean(axis=0, keepdims=True)) ** 2,
            axis=0,
        )

        gain = np.maximum(sse0 - sse1, 0.0)
        r2 = np.divide(
            gain,
            sse0,
            out=np.zeros_like(gain),
            where=sse0 > 1e-12,
        )

        mse1 = sse1 / df_den
        F = np.divide(
            gain / df_num,
            mse1,
            out=np.zeros_like(gain),
            where=mse1 > 1e-12,
        )

        p = f_dist.sf(F, df_num, df_den)
        q = bh_fdr(p)

        # Same selection order as human_myeloid: FDR first, then R^2 ranking.
        sig_local = np.where(
            np.isfinite(q) & (q <= DYNAMIC_FDR_REPORT)
        )[0]

        if len(sig_local) == 0:
            return None

        sig_rank = sig_local[
            np.argsort(-r2[sig_local], kind="mergesort")
        ]

        chosen_local = sig_rank[
            :min(int(top_n), len(sig_rank))
        ]
        chosen = eligible[chosen_local]
        genes = BIO_EXPR_GENES[chosen]

        grid = np.linspace(pt.min(), pt.max(), BIO_GRID_SIZE)

        Dg = np.asarray(
            build_design_matrices(
                [design_df.design_info],
                {"pt": grid},
            )[0],
            float,
        )

        curves = Dg @ beta[:, chosen_local]

        lo = np.nanmin(curves, axis=0)
        hi = np.nanmax(curves, axis=0)
        scaled = (curves - lo) / (hi - lo + 1e-9)

        order = np.argsort(
            grid[np.nanargmax(scaled, axis=0)],
            kind="mergesort",
        )

        genes = genes[order]
        marker_set = set(map(str, marker_genes))

        return {
            "grid": grid,
            "scaled": scaled[:, order],
            "gene_names": genes,
            "markers_in_topn": [g for g in genes if g in marker_set],
            "n_cells": len(cells),
            "n_eligible": len(eligible),
            "n_significant": int(
                np.sum(np.isfinite(q) & (q <= DYNAMIC_FDR_REPORT))
            ),
            "r2": r2[chosen_local][order],
            "f_stat": F[chosen_local][order],
            "p_value": p[chosen_local][order],
            "q_value": q[chosen_local][order],
        }

    # ==========================================
    # Multi-panel layout.
    # ==========================================
    fig = plt.figure(figsize=(26, 24))

    gs_top = gridspec.GridSpec(1, 3, left=0.05, right=0.95, top=0.96, bottom=0.74, wspace=0.25)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])

    gs_bottom = gridspec.GridSpec(1, 4, left=0.05, right=0.95, top=0.68, bottom=0.05, wspace=0.35)

    gs_left = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_bottom[0, 0:3], hspace=0.4, wspace=0.25)
    ax_d = fig.add_subplot(gs_left[0, :])
    axes_e = [fig.add_subplot(gs_left[1, i]) for i in range(3)]
    axes_f = [fig.add_subplot(gs_left[2, i]) for i in range(3)]

    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0, 3:5], hspace=0.2)
    axes_hm = [fig.add_subplot(gs_right[0, 0]), fig.add_subplot(gs_right[1, 0])]

    all_target_paths = {}
    for t_name in TARGET_LINEAGES:
        matches = [
            k for k in dijkstra_paths
            if str(cell_types_arr[k]) == t_name
        ]
        if matches:
            all_target_paths[matches[0]] = dijkstra_paths[matches[0]]

    GLOBAL_SMOOTH_SEGS, _, GLOBAL_NODE_LISTS = extract_and_smooth_tree(
        all_target_paths,
        umap_coords,
        smooth=True,
    )

    ef_pt_norm = normalize_pt_array(trajectory_global_pt)

    type_to_node = {
        str(cell_types_arr[node]): node
        for node in dijkstra_paths
    }
    selected_ends_ef = [
        type_to_node[t]
        for t in TARGET_LINEAGES
        if t in type_to_node
    ]
    selected_ends_f = selected_ends_ef.copy()

    # ==========================================
    # Panels a-c: cell types, inferred paths, and pseudotime.
    # ==========================================
    # (a)
    for label in unique_cell_types:
        mask = cell_types_arr == label
        c_color = GLOBAL_COLOR_MAP.get(str(label), 'gray') 
        ax_a.scatter(umap_coords[mask, 0], umap_coords[mask, 1], c=c_color, label=f'{label}', alpha=0.6, s=15, edgecolors='none')
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for spine in ax_a.spines.values(): spine.set_visible(False)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), borderaxespad=0., frameon=False, ncol=1, fontsize=9)
    ax_a.annotate("", xy=(-0.05 + 0.12, -0.05), xytext=(-0.05 - 0.02, -0.05), xycoords='axes fraction', arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax_a.annotate("", xy=(-0.05, -0.05 + 0.2), xytext=(-0.05, -0.05 - 0.02), xycoords='axes fraction', arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax_a.text(0.05, -0.08, "UMAP1", transform=ax_a.transAxes, ha='center', va='top', fontsize=12)
    ax_a.text(-0.08, 0.05, "UMAP3", transform=ax_a.transAxes, ha='right', va='center', rotation=90, fontsize=12)

    # (b)
    ax_b.scatter(umap_coords[:, 0], umap_coords[:, 1], c='#e0e0e0', alpha=0.4, s=15, edgecolors='none')
    ax_b.scatter(umap_coords[start_idx, 0], umap_coords[start_idx, 1], fc=START_COLOR, ec=START_EDGE, marker=START_MARKER, s=START_SIZE, zorder=15)
    for seg, nodes in zip(GLOBAL_SMOOTH_SEGS, GLOBAL_NODE_LISTS):
        belonging_indices = [e for e, p in all_target_paths.items() if all(n in p for n in nodes)]
        active_labels = [str(cell_types_arr[e]) for e in belonging_indices if str(cell_types_arr[e]) in TARGET_LINEAGES]
        if not active_labels: continue 
        ax_b.plot(seg[:, 0], seg[:, 1], c=EXPLICIT_COLORS[active_labels[0]], linewidth=3.5, alpha=0.85, zorder=10)
    for end_idx in ends_idx:
        label = str(cell_types_arr[end_idx])
        if label in TARGET_LINEAGES:
            ax_b.scatter(umap_coords[end_idx, 0], umap_coords[end_idx, 1], c=EXPLICIT_COLORS[label], s=END_SIZE, marker=END_MARKER, edgecolors='white', zorder=12)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.axis('off')

    # (c)
    global_pt_norm = normalize_pt_array(global_pt) 
    sc_c = ax_c.scatter(umap_coords[:, 0], umap_coords[:, 1], c=global_pt_norm, cmap='viridis', s=15, alpha=0.9, edgecolors='none')
    for seg in GLOBAL_SMOOTH_SEGS:
        ax_c.plot(seg[:, 0], seg[:, 1], color='black', linewidth=3, zorder=10)
    for end_idx in ends_idx:
        label = str(cell_types_arr[end_idx])
        if label in TARGET_LINEAGES:
            ax_c.scatter(umap_coords[end_idx, 0], umap_coords[end_idx, 1], color=EXPLICIT_COLORS[label], marker=END_MARKER, s=END_SIZE, zorder=12, edgecolors='white')
    ax_c.scatter(umap_coords[start_idx, 0], umap_coords[start_idx, 1], fc=START_COLOR, ec=START_EDGE, marker=START_MARKER, s=START_SIZE, zorder=15)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.axis('off')
    cax = ax_c.inset_axes([0.1, -0.08, 0.8, 0.05])
    cbar = fig.colorbar(sc_c, cax=cax, orientation='horizontal')
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(['Low', 'High']); cbar.set_label('Pseudotime', fontsize=10)

    # ==========================================
    # Panel d: reward profiles.
    # ==========================================
    _, path_steps_dict = q_learner.calculate_path_rewards(dijkstra_paths)
    for end_node, raw_rewards in path_steps_dict.items():
        real_label = str(cell_types_arr[end_node])
        if real_label not in TARGET_LINEAGES: continue
        ax_d.plot(range(len(raw_rewards)), raw_rewards, marker='o', ms=4, lw=2.5, color=EXPLICIT_COLORS[real_label], alpha=0.85, label=real_label)
    ax_d.axhline(0, color='red', ls='--', lw=1.5, alpha=0.6, zorder=1) 
    ax_d.set_xlabel("Steps"); ax_d.set_ylabel("Reward Value")
    ax_d.grid(True, ls=':', alpha=0.5)
    ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
    ax_d.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=11, title="Endpoints")

    # ==========================================
    # Panel e: marker dynamics.
    # STRICTLY matched to the human_myeloid method:
    # independent trajectory-wise Gaussian-kernel NW, sigma=0.08, grid=100
    # ==========================================
    target_genes_e = ['HBD', 'MPO', 'FLT3']

    for idx, (gene, ax) in enumerate(zip(target_genes_e, axes_e)):
        expr_all = get_bio_gene_expression(gene)

        if expr_all is None:
            ax.set_title(f"{gene} Not Found")
            ax.set_axis_off()
            continue

        for end_node in selected_ends_ef:
            end_type = str(cell_types_arr[end_node])
            path_cells = np.asarray(dijkstra_paths[end_node], dtype=int)

            if len(path_cells) < 10:
                continue

            grid, curve = nw_gaussian_curve(
                ef_pt_norm[path_cells],
                expr_all[path_cells],
                sigma_pt=BIO_PT_SIGMA,
            )

            if grid is None:
                continue

            ax.plot(
                grid,
                curve,
                color=EXPLICIT_COLORS[end_type],
                lw=3.0,
                alpha=0.95,
                label=f"Trajectory {end_type}",
            )

        ax.set_title(gene, fontsize=16, pad=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.annotate(
            '',
            xy=(1.04, 0),
            xycoords='axes fraction',
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            annotation_clip=False,
        )
        ax.annotate(
            '',
            xy=(0, 1.04),
            xycoords='axes fraction',
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            annotation_clip=False,
        )

        ax.set_xlabel(
            "Global trajectory pseudotime",
            fontsize=12,
            fontweight='bold',
            labelpad=12,
        )

        if idx == 0:
            ax.set_ylabel(
                "Smoothed normalized expression",
                fontsize=12,
                fontweight='bold',
                labelpad=16,
            )

    # ==========================================
    # Panel f: feature plots with lineage paths.
    # ==========================================
    target_pairs_f = [('HBD', 'Ery'), ('MPO', 'Mono'), ('FLT3', 'DC')]

    for idx, ((gene, target_type), ax) in enumerate(zip(target_pairs_f, axes_f)):
        if target_type not in type_to_node:
            ax.axis('off')
            continue

        expr_all = get_bio_gene_expression(gene)
        if expr_all is None:
            ax.axis('off')
            continue

        order = np.argsort(expr_all)
        expr_sorted = expr_all[order]
        coords_sorted = umap_coords[order]

        sc_plot = ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1], c=expr_sorted, cmap='plasma', s=10, alpha=0.8, edgecolors='none', zorder=1)

        end_node = type_to_node[target_type]
        if end_node in dijkstra_paths:
            path_nodes = dijkstra_paths[end_node]
            segs, _, _ = extract_and_smooth_tree({end_node: path_nodes}, umap_coords, smooth=True)
            path_coords = segs[0] if segs else umap_coords[path_nodes]

            line_color = EXPLICIT_COLORS.get(target_type, 'black')
            ax.plot(path_coords[:, 0], path_coords[:, 1], color=line_color, linewidth=4, zorder=10)
            ax.scatter(path_coords[0, 0], path_coords[0, 1], c='white', edgecolors='black', marker='*', s=150, zorder=15)
            ax.scatter(path_coords[-1, 0], path_coords[-1, 1], c=line_color, edgecolors='white', marker='X', s=80, zorder=15)

        ax.set_title(f"{gene} Expression\n\u2192 Trajectory {target_type}", fontsize=14, pad=10)
        ax.axis('off')

        cax = ax.inset_axes([0.1, -0.05, 0.8, 0.05])
        cbar = fig.colorbar(sc_plot, cax=cax, orientation='horizontal')
        cbar.set_ticks([expr_sorted.min(), expr_sorted.max()]); cbar.set_ticklabels(['Low', 'High'])

    # ==========================================
    # Panel g: dynamic-gene heatmaps.
    # STRICTLY matched to the human_myeloid method:
    # path-assigned cells -> natural cubic spline(df=5) -> F test ->
    # BH-FDR q<=0.05 -> R^2 ranking -> Top80 ->
    # fitted 100-point grid -> per-gene 0-1 scaling -> peak ordering
    # Known markers are annotation-only and NEVER forced into Top80.
    # ==========================================
    heatmap_targets = ['Ery', 'Mono']
    selected_ends_hm = [
        type_to_node[t]
        for t in heatmap_targets
        if t in type_to_node
    ]

    # Bone-marrow marker names are used ONLY for annotation.
    COMMON_EARLY_MARKERS = [
        'CD34', 'KIT', 'PROM1', 'MSI2',
        'GATA2', 'HLF', 'MECOM', 'SPINK2', 'FLT3',
    ]

    HEATMAP_MARKERS = {
        'Ery': COMMON_EARLY_MARKERS + [
            'TAL1', 'GATA1', 'KLF1', 'GFI1B', 'ZFPM1',
            'EPOR', 'TFRC', 'CD36',
            'ALAS2', 'FECH', 'SLC25A37', 'AHSP',
            'HBA1', 'HBA2', 'HBB', 'HBD',
            'GYPA', 'GYPB', 'SLC4A1',
            'RHAG', 'ANK1', 'EPB42', 'BLVRB',
        ],
        'Mono': COMMON_EARLY_MARKERS + [
            'MPO', 'SPI1', 'IRF8', 'CSF1R',
            'LYZ', 'LST1', 'FCN1', 'VCAN',
            'CD14', 'CCR2', 'S100A8', 'S100A9',
            'TYROBP', 'FCER1G', 'AIF1',
            'MS4A7', 'FCGR3A',
            'CTSS', 'CST3', 'SERPINA1', 'LGALS3',
            'LILRB1', 'LILRB3',
        ],
    }

    print("Biological-expression source:", BIO_EXPR_SOURCE)

    for idx, (end_node, ax) in enumerate(zip(selected_ends_hm, axes_hm)):
        end_type = str(cell_types_arr[end_node])
        path_cells = np.asarray(dijkstra_paths[end_node], dtype=int)

        heat = prepare_spline_heatmap(
            path_cells,
            HEATMAP_MARKERS.get(end_type, []),
            DYNAMIC_TOP_N,
        )

        if heat is None:
            ax.set_axis_off()
            print(
                f"Trajectory {end_type}: path_cells={len(path_cells)}, "
                "spline analysis unavailable"
            )
            continue

        grid = heat["grid"]
        gene_names = np.asarray(heat["gene_names"], dtype=str)

        im = ax.imshow(
            heat["scaled"].T,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            vmin=0,
            vmax=1,
            extent=[
                grid[0],
                grid[-1],
                len(gene_names),
                0,
            ],
        )

        ax.set_title(
            f"→ Trajectory {end_type}",
            fontsize=16,
            pad=15,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        if idx == 0:
            ax.set_ylabel(
                f"Dynamic genes (n={len(gene_names)})",
                fontsize=14,
                labelpad=10,
            )

        ax.set_xlabel(
            "Trajectory pseudotime →",
            fontsize=13,
            labelpad=5,
        )

        # Annotate only known markers that naturally entered the statistically selected Top80.
        for gene in heat["markers_in_topn"]:
            y_pos = np.where(gene_names == gene)[0][0] + 0.5
            ax.annotate(
                gene,
                xy=(grid[-1], y_pos),
                xycoords='data',
                xytext=(1.04, y_pos),
                textcoords=('axes fraction', 'data'),
                arrowprops=dict(
                    facecolor='red',
                    edgecolor='red',
                    width=2.0,
                    headwidth=7,
                    headlength=6,
                ),
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                fontweight='bold',
                annotation_clip=False,
            )

        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation='horizontal',
            fraction=0.04,
            pad=0.08,
        )
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        print(
            f"Trajectory {end_type}: "
            f"path_cells={len(path_cells)}, "
            f"eligible_genes={heat['n_eligible']}, "
            f"q<{DYNAMIC_FDR_REPORT:g}={heat['n_significant']}, "
            f"plotted={len(gene_names)}, "
            f"markers_in_top80={heat['markers_in_topn']}"
        )

    # ==========================================
    # Add panel labels.
    # ==========================================
    axes_label_map = zip(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 
                         [ax_a, ax_b, ax_c, ax_d, axes_e[0], axes_f[0], axes_hm[0]])

    for label, ax in axes_label_map:
        ax.text(-0.05, 1.05, label, transform=ax.transAxes, fontsize=32, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout()
    # Save four publication formats.
    base_name = "human_bone_marrow_total"
    output_paths = {
        "png": big_output_dir / f"{base_name}.png",
        "tiff": big_output_dir / f"{base_name}.tiff",
        "eps": big_output_dir / f"{base_name}.eps",
        "svg": big_output_dir / f"{base_name}.svg",
    }
    fig.savefig(output_paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(output_paths["tiff"], dpi=600, bbox_inches="tight", format="tiff")
    fig.savefig(output_paths["eps"], bbox_inches="tight", format="eps")
    fig.savefig(output_paths["svg"], bbox_inches="tight", format="svg")
    plt.close(fig)

    print("Big figure saved to:")
    for fmt, path in output_paths.items():
        print(f"{fmt.upper():4s}: {path}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results" / "figures",
    )
    parser.add_argument(
        "--run-output-root", type=Path,
        default=REPO_ROOT / "results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    make_figures(args.output_dir, args.run_output_root, args.seed, args.episodes)


if __name__ == "__main__":
    main()
