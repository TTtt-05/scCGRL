"""Dataset-specific publication figure migrated without merging plotting logic."""
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

DATASET_KEY = "mouse_pancreas"
SOURCE_NOTEBOOK = "2026-08-17_scCGRL_five_datasets_v1.ipynb"
SOURCE_CELL_INDEX = 42


def activate_big_figure_dataset(run_output_root, seed=42, episodes=10000):
    """Run the canonical final pipeline once and expose the notebook variables."""
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
    adata, q_learner, dijkstra_paths, result, results = activate_big_figure_dataset(
        run_output_root, seed=seed, episodes=episodes
    )
    big_output_dir = Path(output_dir)
    big_output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d
    from scipy.stats import f as f_dist
    from patsy import dmatrix, build_design_matrices
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import scanpy as sc
    import seaborn as sns
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # ============================================================
    # Configuration -- same statistical/heatmap settings as the
    # supplied human_myeloid code
    # ============================================================
    BIO_PT_SIGMA = 0.08
    BIO_GRID_SIZE = 100
    DYNAMIC_TOP_N = 80

    DYNAMIC_SPLINE_DF = 5
    DYNAMIC_FDR_REPORT = 0.05
    DYNAMIC_MIN_DETECTION = 0.10
    DYNAMIC_MIN_SD = 1e-4

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    START_MARKER, START_SIZE, START_COLOR, START_EDGE = '*', 250, 'white', 'black'
    END_MARKER, END_SIZE = 'X', 120
    END_COLOR = 'red'

    # Keep the mouse_pancreas annotation column from the original code.
    cluster_key = "clusters_fig6_broad_final"

    umap_coords = adata.obsm['X_umap'][:, :2]
    cell_types_arr = np.asarray(adata.obs[cluster_key].astype(str))
    start_idx = result['start_index']
    ends_idx = result['endpoint_indices']

    # Same division of pseudotime roles as the reference implementation:
    # C uses cell-level RF pseudotime; trajectory-specific biological panels
    # and spline heatmaps use the inferred global trajectory pseudotime.
    rf_pt_all = np.asarray(results['rf_pseudotime'], dtype=float)
    trajectory_global_pt = np.asarray(results['global_pseudotime'], dtype=float)
    global_pt = rf_pt_all

    TARGET_LINEAGES = ['Beta', 'Alpha', 'Delta', 'Epsilon']

    # Fixed lineage colors so the same endpoint keeps the same color across panels.
    # These colors affect only visualization, not any calculation.
    EXPLICIT_COLORS = {
        'Beta': '#66C2A5',
        'Alpha': '#FC8D62',
        'Delta': '#8DA0CB',
        'Epsilon': '#E78AC3',
    }

    unique_cell_types = np.unique(cell_types_arr)
    global_palette = sns.color_palette("Set2", len(unique_cell_types))
    GLOBAL_COLOR_MAP = {
        ctype: mpl.colors.to_hex(c)
        for ctype, c in zip(unique_cell_types, global_palette)
    }
    GLOBAL_COLOR_MAP.update(EXPLICIT_COLORS)

    # ============================================================
    # Helper functions
    # The following trajectory and spline/heatmap helpers are kept
    # methodologically identical to the supplied human_myeloid code.
    # ============================================================
    def normalize_pt_array(pt):
        pt = np.asarray(pt, dtype=float).copy()
        mask = np.isfinite(pt) & (pt >= 0)
        if np.any(mask):
            pmin, pmax = pt[mask].min(), pt[mask].max()
            if pmax > pmin:
                pt[mask] = (pt[mask] - pmin) / (pmax - pmin)
        return pt


    def extract_and_smooth_tree(paths_dict, coords, num_points=100, sigma=3, smooth=False):
        if not paths_dict:
            return [], [], []

        clean_paths = {
            k: (v['path'] if isinstance(v, dict) else v)
            for k, v in paths_dict.items()
        }
        edges, leaves = set(), set()

        for path in clean_paths.values():
            edges.update((path[i], path[i + 1]) for i in range(len(path) - 1))
            if path:
                leaves.add(path[-1])

        out_degree = {}
        for u, v in edges:
            out_degree[u] = out_degree.get(u, 0) + 1

        segments, terminal_flags, segment_node_lists, covered_edges = [], [], [], set()

        for path in clean_paths.values():
            if not path:
                continue

            curr_seg = [path[0]]
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in covered_edges:
                    curr_seg = [v]
                    continue

                covered_edges.add((u, v))
                curr_seg.append(v)

                if out_degree.get(v, 0) != 1 or v in leaves:
                    segments.append(curr_seg)
                    terminal_flags.append(v in leaves)
                    segment_node_lists.append(curr_seg)
                    curr_seg = [v]

        res_segs = []
        for nodes in segments:
            raw = coords[nodes]
            valid_idx = [0]

            for i in range(1, len(raw)):
                if np.linalg.norm(raw[i] - raw[valid_idx[-1]]) > 1e-4:
                    valid_idx.append(i)

            uni = raw[valid_idx]
            if len(uni) < 3 or not smooth:
                res_segs.append(uni)
                continue

            arc = np.concatenate((
                [0],
                np.cumsum(np.linalg.norm(np.diff(uni, axis=0), axis=1))
            ))
            if arc[-1] > 0:
                arc /= arc[-1]

            u_new = np.linspace(0, 1, num_points)
            smoothed = []

            for d in range(coords.shape[1]):
                f = interp1d(arc, uni[:, d], kind='linear')
                ds = gaussian_filter1d(f(u_new), sigma=sigma)
                ds[0], ds[-1] = uni[0, d], uni[-1, d]
                smoothed.append(ds)

            res_segs.append(np.column_stack(smoothed))

        return res_segs, terminal_flags, segment_node_lists


    def nw_gaussian_curve(pt, expr, sigma_pt=BIO_PT_SIGMA, grid_size=BIO_GRID_SIZE):
        """Nadaraya-Watson Gaussian-kernel curve used for panel E."""
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
            where=denom > 1e-12
        )

        return grid, smooth


    def get_normalized_expression_source(a):
        # Identical priority to the reference code: use normalized adata.raw.X
        # whenever it is available, otherwise use adata.X.
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
        Use only cells on one scCGRL trajectory.

        Exact method aligned with the supplied human_myeloid code:
          1) expression ~ natural spline(pseudotime, df=5)
             versus intercept-only model;
          2) BH-FDR correction, q <= 0.05;
          3) rank significant genes by R^2 and keep at most Top-N;
          4) predict spline expression on a common pseudotime grid;
          5) min-max scale each gene independently to [0, 1];
          6) order genes by pseudotime of peak fitted expression.

        Marker genes are annotation-only: they are NOT forced into selection.
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
        sd = np.sqrt(np.maximum(
            np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean ** 2,
            0
        ))

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
            return_type="dataframe"
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
        sse0 = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)

        gain = np.maximum(sse0 - sse1, 0.0)
        r2 = np.divide(
            gain,
            sse0,
            out=np.zeros_like(gain),
            where=sse0 > 1e-12
        )

        mse1 = sse1 / df_den
        F = np.divide(
            gain / df_num,
            mse1,
            out=np.zeros_like(gain),
            where=mse1 > 1e-12
        )

        p = f_dist.sf(F, df_num, df_den)
        q = bh_fdr(p)

        sig_local = np.where(
            np.isfinite(q) & (q <= DYNAMIC_FDR_REPORT)
        )[0]

        if len(sig_local) == 0:
            return None

        sig_rank = sig_local[
            np.argsort(-r2[sig_local], kind="mergesort")
        ]
        chosen_local = sig_rank[:min(int(top_n), len(sig_rank))]
        chosen = eligible[chosen_local]
        genes = BIO_EXPR_GENES[chosen]

        grid = np.linspace(pt.min(), pt.max(), BIO_GRID_SIZE)
        Dg = np.asarray(
            build_design_matrices(
                [design_df.design_info],
                {"pt": grid}
            )[0],
            float
        )

        curves = Dg @ beta[:, chosen_local]

        lo = np.nanmin(curves, axis=0)
        hi = np.nanmax(curves, axis=0)
        scaled = (curves - lo) / (hi - lo + 1e-9)

        order = np.argsort(
            grid[np.nanargmax(scaled, axis=0)],
            kind="mergesort"
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
            "n_significant": int(np.sum(np.isfinite(q) & (q <= DYNAMIC_FDR_REPORT))),
            "r2": r2[chosen_local][order],
            "f_stat": F[chosen_local][order],
            "p_value": p[chosen_local][order],
            "q_value": q[chosen_local][order],
        }


    # ============================================================
    # Target trajectories -- same target-path construction logic
    # as the supplied human_myeloid code, adapted to four pancreas fates
    # ============================================================
    all_target_paths = {}
    for t_name in TARGET_LINEAGES:
        matches = [
            k for k in dijkstra_paths
            if str(cell_types_arr[k]) == t_name
        ]
        if matches:
            all_target_paths[matches[0]] = dijkstra_paths[matches[0]]

    GLOBAL_SMOOTH_SEGS, GLOBAL_FLAGS, GLOBAL_NODE_LISTS = extract_and_smooth_tree(
        all_target_paths,
        umap_coords,
        smooth=True,
        sigma=3
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

    print("Selected mouse pancreas trajectories:",
          {str(cell_types_arr[e]): int(e) for e in selected_ends_f})
    print("Biological-expression source:", BIO_EXPR_SOURCE)

    # ============================================================
    # Layout -- retain the original mouse 3 x 4 structure
    # ============================================================
    fig = plt.figure(figsize=(30, 24))
    gs = gridspec.GridSpec(
        3, 4,
        height_ratios=[1, 1, 2.5],
        wspace=0.28,
        hspace=0.45
    )

    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_e = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axes_f = [fig.add_subplot(gs[2, i]) for i in range(4)]

    # ============================================================
    # A. UMAP
    # ============================================================
    for label in unique_cell_types:
        mask = cell_types_arr == label
        ax_a.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=GLOBAL_COLOR_MAP.get(str(label), 'gray'),
            label=str(label),
            alpha=0.65,
            s=20,
            edgecolors='none'
        )

    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    ax_a.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title='Cell types',
        borderaxespad=0.0
    )

    origin_x, origin_y = -0.08, -0.08
    ax_a.annotate(
        "", xy=(origin_x + 0.18, origin_y), xytext=(origin_x - 0.01, origin_y),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
    )
    ax_a.annotate(
        "", xy=(origin_x, origin_y + 0.18), xytext=(origin_x, origin_y - 0.01),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
    )
    ax_a.text(origin_x + 0.09, origin_y - 0.03, "UMAP1",
              transform=ax_a.transAxes, ha='center', va='top')
    ax_a.text(origin_x - 0.03, origin_y + 0.09, "UMAP2",
              transform=ax_a.transAxes, ha='right', va='center', rotation=90)

    # ============================================================
    # B. Trajectory
    # EXACT plotting logic aligned with the supplied human_myeloid code
    # ============================================================
    ax_b.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c='#e0e0e0', alpha=0.4, s=15, edgecolors='none'
    )

    ax_b.scatter(
        umap_coords[start_idx, 0], umap_coords[start_idx, 1],
        fc=START_COLOR, ec=START_EDGE,
        marker=START_MARKER, s=START_SIZE, zorder=15
    )

    # Pre-compute the geometric length of every target trajectory in the same
    # UMAP space used for panel B.  For a segment shared by multiple terminal paths,
    # its color is inherited from the FARTHEST terminal among the paths that contain
    # that segment.  Example: a segment shared by all four lineages uses the color of
    # the farthest of the four endpoints; a later segment shared only by Alpha/Beta
    # uses the color of whichever of Alpha/Beta is farther from the root along its
    # inferred path.
    def _trajectory_path_length(path, coords):
        path = np.asarray(path, dtype=int)
        if len(path) < 2:
            return 0.0
        xy = np.asarray(coords[path], dtype=float)
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    TARGET_PATH_LENGTHS = {
        int(end_node): _trajectory_path_length(path, umap_coords)
        for end_node, path in all_target_paths.items()
    }

    print(
        "Mouse pancreas target-path lengths used for shared-segment colors:",
        {
            str(cell_types_arr[e]): round(TARGET_PATH_LENGTHS[e], 4)
            for e in TARGET_PATH_LENGTHS
        },
    )

    for seg, nodes in zip(GLOBAL_SMOOTH_SEGS, GLOBAL_NODE_LISTS):
        belonging = [
            e for e, p in all_target_paths.items()
            if all(n in p for n in nodes)
            and str(cell_types_arr[e]) in TARGET_LINEAGES
        ]

        if not belonging:
            continue

        # IMPORTANT: shared segment -> color of the farthest terminal among only
        # the trajectories that actually contain this segment.
        farthest_end = max(
            belonging,
            key=lambda e: (TARGET_PATH_LENGTHS[int(e)], len(all_target_paths[e]))
        )
        farthest_label = str(cell_types_arr[farthest_end])
        line_color = EXPLICIT_COLORS[farthest_label]

        ax_b.plot(
            seg[:, 0], seg[:, 1],
            color=line_color,
            lw=3.5,
            alpha=0.85,
            zorder=10
        )

    for end_idx in ends_idx:
        label = str(cell_types_arr[end_idx])
        if label in TARGET_LINEAGES:
            ax_b.scatter(
                umap_coords[end_idx, 0], umap_coords[end_idx, 1],
                c=EXPLICIT_COLORS[label],
                marker=END_MARKER,
                s=END_SIZE,
                edgecolors='white',
                zorder=12
            )

    ax_b.axis('off')

    # ============================================================
    # C. RF cell-level pseudotime + the same inferred trajectory backbone
    # ============================================================
    pt_norm = normalize_pt_array(global_pt)
    sc_c = ax_c.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=pt_norm,
        cmap='viridis',
        s=15,
        alpha=0.9,
        edgecolors='none'
    )

    for seg in GLOBAL_SMOOTH_SEGS:
        ax_c.plot(seg[:, 0], seg[:, 1], color='black', lw=3, zorder=10)

    for end_idx in ends_idx:
        label = str(cell_types_arr[end_idx])
        if label in TARGET_LINEAGES:
            ax_c.scatter(
                umap_coords[end_idx, 0], umap_coords[end_idx, 1],
                color=EXPLICIT_COLORS[label],
                marker=END_MARKER,
                s=END_SIZE,
                edgecolors='white',
                zorder=12
            )

    ax_c.scatter(
        umap_coords[start_idx, 0], umap_coords[start_idx, 1],
        fc='white', ec='black', marker='*', s=START_SIZE, zorder=15
    )
    ax_c.axis('off')

    divider = make_axes_locatable(ax_c)
    cax = divider.append_axes("bottom", size="4%", pad=0.1)
    cbar = fig.colorbar(sc_c, cax=cax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Cell-level pseudotime (RF projection)', fontsize=10)

    # ============================================================
    # D. Reward profiles -- retain original calculation
    # ============================================================
    _, path_steps_dict = q_learner.calculate_path_rewards(dijkstra_paths)

    for end_node, raw_rewards in path_steps_dict.items():
        label = str(cell_types_arr[end_node])
        if label not in TARGET_LINEAGES:
            continue

        ax_d.plot(
            range(len(raw_rewards)),
            raw_rewards,
            marker='o', ms=4, lw=2.5,
            color=EXPLICIT_COLORS[label],
            alpha=0.85,
            label=label
        )

    ax_d.axhline(0, color='red', ls='--', lw=1.5, alpha=0.6)
    ax_d.set_title("Reward Profiles")
    ax_d.set_xlabel("Steps")
    ax_d.set_ylabel("Reward Value")
    ax_d.grid(True, ls=':', alpha=0.5)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.legend(frameon=False, title="Endpoints")

    # ============================================================
    # E. Marker dynamics
    # EXACTLY aligned with the current human_myeloid implementation:
    #   - normalized expression source: prefer adata.raw.X
    #   - normalized global trajectory pseudotime
    #   - Nadaraya-Watson Gaussian-kernel smoothing
    #   - BIO_PT_SIGMA = 0.08
    #   - BIO_GRID_SIZE = 100
    # Only the pancreas marker genes differ.
    # ============================================================
    target_genes_e = ['Ins2', 'Gcg', 'Sst', 'Ghrl']

    for idx, (gene, ax) in enumerate(zip(target_genes_e, axes_e)):
        expr_all = get_bio_gene_expression(gene)
        if expr_all is None:
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
                sigma_pt=BIO_PT_SIGMA
            )
            if grid is None:
                continue

            ax.plot(
                grid,
                curve,
                color=EXPLICIT_COLORS[end_type],
                lw=3.0,
                alpha=0.95,
                label=f"Trajectory {end_type}"
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
            arrowprops=dict(
                arrowstyle='->',
                color='black',
                lw=1.5
            ),
            annotation_clip=False
        )
        ax.annotate(
            '',
            xy=(0, 1.04),
            xycoords='axes fraction',
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->',
                color='black',
                lw=1.5
            ),
            annotation_clip=False
        )

        ax.set_xlabel(
            "Global trajectory pseudotime",
            fontsize=12,
            fontweight='bold',
            labelpad=12
        )

        if idx == 0:
            ax.set_ylabel(
                "Smoothed normalized expression",
                fontsize=12,
                fontweight='bold',
                labelpad=16
            )

    # ============================================================
    # F. Natural-spline pseudotime-associated gene heatmaps
    # EXACT statistical method aligned with the supplied human_myeloid code
    # ============================================================
    COMMON_EARLY_MARKERS = [
        # Endocrine progenitor / early endocrine differentiation
        'Neurog3','Pdx1','Nkx6-1','Fev','Neurod1','Insm1', 'Pax6','Isl1',  'Chga', 'Chgb', 'Cpe','Pcsk1n',
    ]

    CLUSTER_MARKERS = {
        'Beta': [
            # Canonical beta-cell markers
            'Ins1','Ins2', 'Iapp', 'Pcsk1', 'Mafa', 'Slc2a2', 'Pdx1', 'Nkx6-1', 'Ucn3', 'G6pc2',   'Abcc8', 'Kcnj11', 'Cpe',
        ],

        'Alpha': [
            # Canonical alpha-cell markers
            'Gcg', 'Arx','Mafb', 'Ttr', 'Irx1', 'Irx2', 'Pcsk2', 'Fap', 'Gc',
        ],

        'Delta': [
            # Canonical delta-cell markers
            'Sst','Hhex', 'Ghsr', 'Rbp4',  'Gpr149',  'Ffar4', 'Hdc',
        ],

        'Epsilon': [
            # Ghrelin-positive epsilon-cell program
            'Ghrl','Sox4', 'Sox11', 'Igfbp7',  'Vsig1', 'Gng12', 'Col3a1',  'Col1a1',
        ],
    }

    # Marker lists are used only for annotation of genes already selected by
    # the spline/FDR/R^2 pipeline. They do not alter statistical selection.
    marker_map = {
        lineage: COMMON_EARLY_MARKERS + CLUSTER_MARKERS[lineage]
        for lineage in TARGET_LINEAGES
    }

    for idx, (end_node, ax) in enumerate(zip(selected_ends_f, axes_f)):
        end_type = str(cell_types_arr[end_node])
        path_cells = np.asarray(dijkstra_paths[end_node], dtype=int)

        heat = prepare_spline_heatmap(
            path_cells,
            marker_map.get(end_type, []),
            DYNAMIC_TOP_N
        )

        if heat is None:
            ax.set_axis_off()
            print(
                f"Trajectory {end_type}: path_cells={len(path_cells)}, "
                "spline analysis unavailable"
            )
            continue

        grid = heat["grid"]
        genes = np.asarray(heat["gene_names"], str)

        im = ax.imshow(
            heat["scaled"].T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=[grid[0], grid[-1], len(genes), 0]
        )

        ax.set_title(
            f"→ Trajectory {end_type}",
            fontsize=16,
            pad=15,
            color="black",
            fontweight="bold"
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel(
            "Trajectory pseudotime →",
            fontsize=13,
            fontweight="bold"
        )

        if idx == 0:
            ax.set_ylabel(
                f"Dynamic genes (n={len(genes)})",
                fontsize=14
            )
        else:
            ax.set_ylabel("")

        for gene in heat["markers_in_topn"]:
            y = np.where(genes == gene)[0][0] + 0.5
            ax.annotate(
                gene,
                xy=(grid[-1], y),
                xycoords="data",
                xytext=(1.10, y),
                textcoords=("axes fraction", "data"),
                arrowprops=dict(
                    facecolor="red",
                    edgecolor="red",
                    width=3.5,
                    headwidth=9,
                    headlength=8
                ),
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                fontweight="bold",
                annotation_clip=False
            )

        cb = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            fraction=0.04,
            pad=0.08
        )
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])

        print(
            f"Trajectory {end_type}: "
            f"path_cells={len(path_cells)}, "
            f"eligible_genes={heat['n_eligible']}, "
            f"q<{DYNAMIC_FDR_REPORT:g}={heat['n_significant']}, "
            f"plotted={len(genes)}"
        )

    # ============================================================
    # Panel labels and output
    # ============================================================
    for label, ax in zip(
        ['a', 'b', 'c', 'd', 'e', 'f'],
        [ax_a, ax_b, ax_c, ax_d, axes_e[0], axes_f[0]]
    ):
        ax.text(
            -0.05, 1.05, label,
            transform=ax.transAxes,
            fontsize=32,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    plt.tight_layout(rect=[0, 0, 0.96, 1])

    # # Save to the same dataset-specific output folder used by the original code.
    # png_path = big_output_dir / "mouse_total_spline_heatmap_trajectory_v2.0.png"
    # plt.savefig(png_path, dpi=300, bbox_inches='tight')

    # plt.show()
    # print("Big figure saved to:", png_path)
    # ============================================================
    # Save figure
    # ============================================================
    big_output_dir = Path(output_dir)
    big_output_dir.mkdir(parents=True, exist_ok=True)

    base_name = "mouse_pancreas_total"

    # PNG
    png_path = big_output_dir / f"{base_name}.png"
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    # TIFF
    tiff_path = big_output_dir / f"{base_name}.tiff"
    plt.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        format="tiff"
    )

    # EPS
    eps_path = big_output_dir / f"{base_name}.eps"
    plt.savefig(
        eps_path,
        bbox_inches="tight",
        format="eps"
    )

    # SVG
    svg_path = big_output_dir / f"{base_name}.svg"
    plt.savefig(
        svg_path,
        bbox_inches="tight",
        format="svg"
    )

    plt.show()

    print("Big figure saved to:")
    print("PNG :", png_path)
    print("TIFF:", tiff_path)
    print("EPS :", eps_path)
    print("SVG :", svg_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    parser.add_argument("--run-output-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    make_figures(args.output_dir, args.run_output_root, seed=args.seed, episodes=args.episodes)


if __name__ == "__main__":
    main()
