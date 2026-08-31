# Canonical source notebook: 2026-08-17_scCGRL_five_datasets_v1.ipynb
# Notebook date/version: 2026-08-17 / CODE_REVISION 1.6
# Source cell: index 15 / order 16; I/O helpers from index 47 / order 48 appended
# ==========================================
# Module 4: universal, configuration-driven plotting
# ==========================================
from pathlib import Path
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns
from adjustText import adjust_text


def _capture_displayed_figures(callback, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(plt.get_fignums())
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        callback()
        numbers = [number for number in plt.get_fignums() if number not in before]
        saved = []
        for index, number in enumerate(numbers, start=1):
            figure = plt.figure(number)
            suffix = f'_{index}' if len(numbers) > 1 else ''
            path = output_dir / f'{stem}{suffix}.png'
            figure.savefig(path, dpi=300, bbox_inches='tight')
            saved.append(path)
            plt.close(figure)
        return saved
    finally:
        plt.show = original_show


def _dense_gene(adata, gene):
    values = adata[:, gene].X
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values).reshape(-1)


def _plot_dynamic_heatmap(ax, adata, path_cells, pseudotime, endpoint_name, marker_genes, top_n_genes):
    if len(path_cells) < 3:
        ax.text(0.5, 0.5, f'Path too short: {endpoint_name}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return None
    branch = adata[path_cells, :]
    expression = branch.X.toarray() if sp.issparse(branch.X) else np.asarray(branch.X)
    branch_pt = np.asarray(pseudotime)[path_cells]
    order = np.argsort(branch_pt)
    expression = expression[order]
    sigma = max(2, int(len(path_cells) * 0.05))
    smoothed = gaussian_filter1d(expression, sigma=sigma, axis=0)
    amplitudes = np.max(smoothed, axis=0) - np.min(smoothed, axis=0)
    amplitudes[np.mean(smoothed, axis=0) <= 0.01] = 0
    n_genes = min(int(top_n_genes), smoothed.shape[1])
    selected = np.argsort(amplitudes)[-n_genes:].tolist()
    valid_markers = [gene for gene in marker_genes if gene in branch.var_names]
    var_names = branch.var_names.tolist()
    for gene in valid_markers:
        gene_index = var_names.index(gene)
        if gene_index not in selected:
            selected.pop(0)
            selected.append(gene_index)
    matrix = smoothed[:, selected]
    low, high = matrix.min(axis=0), matrix.max(axis=0)
    scaled = (matrix - low) / (high - low + 1e-9)
    gene_order = np.argsort(np.argmax(scaled, axis=0))
    scaled = scaled[:, gene_order]
    names = np.asarray(branch.var_names[selected])[gene_order]
    image = ax.imshow(scaled.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(f'{endpoint_name} cascade', fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(f'Dynamic genes (n={n_genes})')
    for gene in valid_markers:
        positions = np.where(names == gene)[0]
        if len(positions):
            y = int(positions[0])
            ax.annotate(gene, xy=(1.0, y), xycoords=('axes fraction', 'data'),
                        xytext=(1.04, y), textcoords=('axes fraction', 'data'),
                        arrowprops=dict(color='red', width=1, headwidth=5),
                        va='center', fontsize=8)
    return image


def plot_comprehensive_configured(
    adata,
    q_learner,
    endpoint_result,
    dijkstra_paths,
    pseudotime_results,
    plot_coords,
    dataset_config,
    save_path,
):
    """Create the comprehensive figure using values supplied only by dataset_config."""
    labels = np.asarray(adata.obs[dataset_config['label_column']].astype(str))
    target_genes = list(dataset_config.get('target_genes', []))
    target_pairs = [tuple(pair) for pair in dataset_config.get('target_pairs', [])]
    marker_genes = list(dataset_config.get('marker_genes', []))
    top_n_genes = int(dataset_config.get('top_n_genes', 80))
    explicit_colors = dict(dataset_config.get('explicit_colors', {}))
    final_pt = pseudotime_results.get('rf_pseudotime')
    if final_pt is None:
        final_pt = pseudotime_results.get('global_pseudotime')
    final_pt = normalize_pt_array(final_pt)
    reference_column = dataset_config.get('reference_pseudotime_column')
    reference_pt = None
    if reference_column is not None:
        if reference_column not in adata.obs:
            raise KeyError(f'{reference_column!r} not in adata.obs')
        reference_pt = normalize_pt_array(np.asarray(adata.obs[reference_column], dtype=float))

    unique_labels = np.unique(labels)
    palette = sns.color_palette('Set2', len(unique_labels))
    color_map = {label: mpl.colors.to_hex(color) for label, color in zip(unique_labels, palette)}
    color_map.update({str(k): v for k, v in explicit_colors.items()})
    # Use the terminal nodes found by the model. Cell-type labels and configured
    # lineage names must not select, replace, or reorder endpoints for plotting.
    endpoint_nodes = [int(node) for node in dijkstra_paths]
    heatmap_nodes = endpoint_nodes[:max(1, int(dataset_config.get('heatmap_count', len(endpoint_nodes))))]

    n_gene_rows = max(1, int(np.ceil(max(1, len(target_genes)) / 2)))
    n_pair_rows = max(1, int(np.ceil(max(1, len(target_pairs)) / 2)))
    left_rows = 2 + n_gene_rows + n_pair_rows
    figure_height = max(14, left_rows * 4.2)
    fig = plt.figure(figsize=(24, figure_height), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    left = outer[0].subgridspec(left_rows, 2, hspace=0.35, wspace=0.25)
    right = outer[1].subgridspec(max(1, len(heatmap_nodes)), 1, hspace=0.30)

    ax_a, ax_b = fig.add_subplot(left[0, 0]), fig.add_subplot(left[0, 1])
    ax_c, ax_d = fig.add_subplot(left[1, 0]), fig.add_subplot(left[1, 1])

    for label in unique_labels:
        mask = labels == label
        ax_a.scatter(plot_coords[mask, 0], plot_coords[mask, 1], s=10, alpha=0.65,
                     color=color_map[label], edgecolors='none', label=label)
    ax_a.legend(frameon=False, fontsize=8, title=dataset_config['label_column'])
    ax_a.set_title('Cell types')

    ax_b.scatter(plot_coords[:, 0], plot_coords[:, 1], c='#dedede', s=7, alpha=0.35, edgecolors='none')
    segments, _, _ = extract_and_smooth_tree(dijkstra_paths, plot_coords, smooth=True)
    for segment in segments:
        ax_b.plot(segment[:, 0], segment[:, 1], color='black', lw=2.1, alpha=0.8)
    start = int(endpoint_result['start_index'])
    ax_b.scatter(plot_coords[start, 0], plot_coords[start, 1], marker='*', s=220,
                 facecolor='white', edgecolor='black', zorder=10)
    for node in endpoint_nodes:
        ax_b.scatter(plot_coords[node, 0], plot_coords[node, 1], marker='X', s=100,
                     color='red', edgecolor='white', zorder=10)
    ax_b.set_title('scCGRL trajectories')

    pt_scatter = ax_c.scatter(plot_coords[:, 0], plot_coords[:, 1], c=final_pt,
                              cmap='viridis', s=10, edgecolors='none')
    for segment in segments:
        ax_c.plot(segment[:, 0], segment[:, 1], color='black', lw=1.5)
    fig.colorbar(pt_scatter, ax=ax_c, orientation='horizontal', fraction=0.05, pad=0.05,
                 label='Pseudotime')
    ax_c.set_title('Inferred pseudotime')

    _, path_steps = q_learner.calculate_path_rewards(dijkstra_paths)
    for endpoint_number, (_, rewards) in enumerate(path_steps.items(), start=1):
        ax_d.plot(rewards, lw=1.7, label=f'Endpoint {endpoint_number}')
    ax_d.axhline(0, color='red', ls='--', lw=1)
    ax_d.legend(frameon=False, fontsize=8)
    ax_d.set(title='Reward profiles', xlabel='Step', ylabel='Reward')

    gene_axes = []
    for index in range(max(1, len(target_genes))):
        row, col = 2 + index // 2, index % 2
        gene_axes.append(fig.add_subplot(left[row, col]))
    if target_genes:
        for gene, ax in zip(target_genes, gene_axes):
            if gene not in adata.var_names:
                ax.text(0.5, 0.5, f'Missing gene: {gene}', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            expression = _dense_gene(adata, gene)
            order = np.argsort(final_pt)
            smooth = gaussian_filter1d(expression[order], sigma=max(2, len(expression) * 0.01))
            ax.plot(final_pt[order], smooth, lw=2)
            ax.set(title=gene, xlabel='Pseudotime', ylabel='Expression')
    elif reference_pt is not None:
        reference_scatter = gene_axes[0].scatter(
            plot_coords[:, 0], plot_coords[:, 1], c=reference_pt,
            cmap='viridis', s=10, edgecolors='none')
        gene_axes[0].set_title(f'Reference pseudotime ({reference_column})')
        gene_axes[0].set_xticks([])
        gene_axes[0].set_yticks([])
        fig.colorbar(reference_scatter, ax=gene_axes[0], orientation='horizontal',
                     fraction=0.05, pad=0.05, label='Reference pseudotime')
    else:
        gene_axes[0].text(0.5, 0.5, 'No target genes configured', ha='center', va='center', transform=gene_axes[0].transAxes)
        gene_axes[0].axis('off')

    pair_start_row = 2 + n_gene_rows
    pair_axes = []
    for index in range(max(1, len(target_pairs))):
        row, col = pair_start_row + index // 2, index % 2
        pair_axes.append(fig.add_subplot(left[row, col]))
    if target_pairs:
        for (gene, _), ax in zip(target_pairs, pair_axes):
            if gene not in adata.var_names:
                ax.text(0.5, 0.5, f'Missing gene: {gene}', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            expression = _dense_gene(adata, gene)
            order = np.argsort(expression)
            scatter = ax.scatter(plot_coords[order, 0], plot_coords[order, 1], c=expression[order],
                                 cmap='plasma', s=9, edgecolors='none')
            ax.set_title(str(gene))
            ax.axis('off')
            fig.colorbar(scatter, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02)
    elif reference_pt is not None:
        valid = np.isfinite(reference_pt) & np.isfinite(final_pt)
        pair_axes[0].scatter(reference_pt[valid], final_pt[valid], s=10, alpha=0.45,
                             color='#3b82f6', edgecolors='none')
        if valid.sum() >= 3:
            rho = spearmanr(reference_pt[valid], final_pt[valid])[0]
            pair_axes[0].text(0.04, 0.94, f'Spearman rho = {rho:.3f}',
                              transform=pair_axes[0].transAxes, va='top')
        pair_axes[0].plot([0, 1], [0, 1], ls='--', lw=1, color='black', alpha=0.5)
        pair_axes[0].set(title='Reference vs inferred pseudotime',
                         xlabel='Reference pseudotime', ylabel='Inferred pseudotime',
                         xlim=(0, 1), ylim=(0, 1))
    else:
        pair_axes[0].text(0.5, 0.5, 'No gene-lineage pairs configured', ha='center', va='center', transform=pair_axes[0].transAxes)
        pair_axes[0].axis('off')

    heatmap_axes = [fig.add_subplot(right[index, 0]) for index in range(max(1, len(heatmap_nodes)))]
    for endpoint_number, (node, ax) in enumerate(zip(heatmap_nodes, heatmap_axes), start=1):
        image = _plot_dynamic_heatmap(
            ax, adata, dijkstra_paths[node], final_pt,
            f'Endpoint {endpoint_number}', marker_genes, top_n_genes
        )
        if image is not None:
            fig.colorbar(image, ax=ax, orientation='horizontal', fraction=0.025, pad=0.04)

    for label, ax in zip('abcdefghijklmnopqrstuvwxyz', [ax_a, ax_b, ax_c, ax_d, *gene_axes, *pair_axes, *heatmap_axes]):
        ax.text(-0.06, 1.04, label, transform=ax.transAxes, fontsize=18, fontweight='bold')
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks([])
        ax.set_yticks([])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig


def save_reference_figures(
    adata,
    q_learner,
    endpoint_result,
    dijkstra_paths,
    pseudotime_results,
    plot_coords,
    dataset_config,
    output_dir,
):
    """Save separate reference figures; every dataset-specific value comes from dataset_config."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_column = dataset_config['label_column']
    labels_array = np.asarray(adata.obs[label_column])
    labels_series = adata.obs[label_column]
    report_path = output_dir / '00_model_report.txt'
    with report_path.open('w', encoding='utf-8') as handle, redirect_stdout(handle):
        q_learner.print_path_rewards_report(dijkstra_paths)

    saved = []
    calls = [
        ('01_dijkstra_paths', lambda: q_learner.plot_paths(dijkstra_paths, title='Dijkstra optimized paths')),
        ('02_endpoint_reach_frequency', q_learner.plot_reach_frequency),
        ('03_endpoint_statistics', lambda: q_learner.plot_endpoint_statistics(dijkstra_paths)),
        ('04_branch_reward_curves', lambda: q_learner.plot_branch_reward_curves(dijkstra_paths)),
        ('05_path_rewards_annotated', lambda: q_learner.plot_path_rewards_annotated(dijkstra_paths)),
        ('06_endpoint_training_log_rewards', q_learner.plot_endpoint_training_log_rewards),
    ]
    for stem, callback in calls:
        saved.extend(_capture_displayed_figures(callback, output_dir, stem))

    q_figure = plot_q_state_values(q_learner, labels_series, plot_coords)
    q_path = output_dir / '07_q_state_values.png'
    q_figure.savefig(q_path, dpi=300, bbox_inches='tight')
    plt.close(q_figure)
    saved.append(q_path)
    saved.extend(_capture_displayed_figures(
        lambda: plot_spatial_rewards_with_celltypes(q_learner, dijkstra_paths, plot_coords, labels_array),
        output_dir, '08_spatial_rewards'))
    saved.extend(_capture_displayed_figures(
        lambda: plot_branching_points_on_umap(q_learner, dijkstra_paths, plot_coords, labels_array),
        output_dir, '09_branching_points'))
    comprehensive_path = output_dir / '10_comprehensive_configured.png'
    plot_comprehensive_configured(
        adata, q_learner, endpoint_result, dijkstra_paths, pseudotime_results,
        plot_coords, dataset_config, comprehensive_path,
    )
    saved.append(comprehensive_path)
    return {'figures': saved, 'report': report_path}


def extract_and_smooth_tree(paths_dict, coords, num_points=100, sigma=3, smooth=False):
    """
    Extract graph segments and apply Gaussian smoothing.
    Returns smoothed segments, terminal flags, and original node lists per segment.
    """
    if paths_dict is None or len(paths_dict) == 0: return [], [], []
    clean_paths = {k: (v['path'] if isinstance(v, dict) else v) for k, v in paths_dict.items()}
    edges, leaves = set(), set()

    for path in clean_paths.values():
        for i in range(len(path) - 1): edges.add((path[i], path[i+1]))
        if len(path) > 0: leaves.add(path[-1])

    out_degree = {u: 0 for u, v in edges}
    for u, v in edges: out_degree[u] += 1

    segments, terminal_flags, covered_edges = [], [], set()
    segment_node_lists = [] 

    for path in clean_paths.values():
        if not path: continue
        curr_seg = [path[0]]
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if (u, v) not in covered_edges:
                covered_edges.add((u, v))
                curr_seg.append(v)
                if out_degree.get(v, 0) != 1 or v in leaves:
                    segments.append(curr_seg)
                    terminal_flags.append(v in leaves)
                    segment_node_lists.append(curr_seg)
                    curr_seg = [v]
            else: 
                curr_seg = [v]

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

    return res_segs, terminal_flags, segment_node_lists


def normalize_pt_array(pt_array):
    """Min-max normalization for valid pseudotime values."""
    pt_norm = np.copy(pt_array).astype(float)
    mask = (pt_norm >= 0) & np.isfinite(pt_norm)
    if np.any(mask):
        pmin, pmax = np.min(pt_norm[mask]), np.max(pt_norm[mask])
        if pmax > pmin: pt_norm[mask] = (pt_norm[mask] - pmin) / (pmax - pmin)
    return pt_norm


def plot_q_state_values(q_learner, cell_types, umap_coords):
    q_matrix = q_learner.Q

    best_incoming_q = {}

    for source, actions in q_matrix.items():
        for target, q_val in actions.items():
            # Skip unupdated initial actions (q_val == 0)
            if q_val != 0:
                if target not in best_incoming_q or q_val > best_incoming_q[target]:
                    best_incoming_q[target] = q_val

    all_barcodes = cell_types.index

    int_to_barcode = {i: bc for i, bc in enumerate(all_barcodes)}

    final_values = {}

    unupdated_cells = []

    for i, barcode in enumerate(all_barcodes):
        if i in best_incoming_q:
            final_values[barcode] = best_incoming_q[i]
        elif i == q_learner.start:
            # Start node lacks incoming edges; assign baseline later
            final_values[barcode] = None 
        else:
            # Unreached cells (isolated)
            final_values[barcode] = np.nan
            unupdated_cells.append(barcode)

    valid_scores = [v for v in final_values.values() if v is not None and not np.isnan(v)]

    min_score = min(valid_scores) if valid_scores else 0

    for bc in final_values:
        if final_values[bc] is None: 
            final_values[bc] = min_score

    print("="*40)

    print(f"Q-Table Update Report")

    print(f"Total cells: {len(all_barcodes)}")

    updated_count = len(all_barcodes) - len(unupdated_cells)
    print(f"Updated cells (including the start baseline): {updated_count}")

    print(f"Unupdated cells (isolated): {len(unupdated_cells)}")

    if unupdated_cells:
        print("\nUnupdated cells ratio by type:")
        missing_stats = cell_types.loc[unupdated_cells].value_counts()
        total_stats = cell_types.value_counts()
        for ctype, count in missing_stats.items():
            print(f" - {ctype}: {count}/{total_stats[ctype]} ({count/total_stats[ctype]*100:.1f}%)")

    print("="*40)

    df_plot = pd.DataFrame({'State_Value': pd.Series(final_values)})

    df_plot['Cell_Type'] = cell_types

    df_plot['UMAP_1'] = umap_coords[:, 0] if isinstance(umap_coords, np.ndarray) else umap_coords.iloc[:, 0].values

    df_plot['UMAP_2'] = umap_coords[:, 1] if isinstance(umap_coords, np.ndarray) else umap_coords.iloc[:, 1].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

    ax1.scatter(df_plot[df_plot['State_Value'].isna()]['UMAP_1'],
                df_plot[df_plot['State_Value'].isna()]['UMAP_2'],
                c='lightgrey', s=10, alpha=0.3, label='Unupdated/Isolated')

    df_valid = df_plot.dropna(subset=['State_Value']).sort_values('State_Value')

    sc = ax1.scatter(df_valid['UMAP_1'], df_valid['UMAP_2'], 
                     c=df_valid['State_Value'], cmap='viridis', s=15, alpha=0.8)

    plt.colorbar(sc, ax=ax1, label='Max Incoming Q-Value (V)')

    ax1.legend(loc='lower left')

    df_violin = df_plot.dropna(subset=['State_Value'])

    order = df_violin.groupby('Cell_Type')['State_Value'].median().sort_values().index

    sns.violinplot(data=df_violin, x='Cell_Type', y='State_Value', order=order, 
                   ax=ax2, palette='magma', inner='quartile')

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_spatial_rewards_with_celltypes(q_learner, paths, coords, cell_types):
    """
    [Standalone Function] 
    Plot trajectories on UMAP, color nodes by cell type, and annotate step rewards.
    """
    if cell_types is None:
        print("Error: cell_types array must be provided.")
        return

    # 1. Extract step rewards from q_learner
    path_totals, path_steps = q_learner.calculate_path_rewards(paths)

    n_ends = len(paths)
    cols = min(2, n_ends) # Max 2 per row for larger UMAPs
    rows = int(np.ceil(n_ends / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
    if n_ends == 1: axes = [axes]
    else: axes = axes.flatten()

    # Create global color palette for cell types
    unique_types = np.unique(cell_types)
    color_palette = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: color_palette[i] for i, t in enumerate(unique_types)}

    for idx, (end, path) in enumerate(paths.items()):
        ax = axes[idx]
        steps = path_steps[end]
        path_coords = coords[path]

        # --- Step A: UMAP global gray background ---
        ax.scatter(coords[:, 0], coords[:, 1], c='#e0e0e0', alpha=0.3, s=15, edgecolors='none')

        # --- Step B: Gray trajectory lines ---
        ax.plot(path_coords[:, 0], path_coords[:, 1], color='#555555', linewidth=2.5, alpha=0.7, zorder=2)

        # --- Step C: Trajectory nodes (colored by cell type) with reward text ---
        for step_idx, (node_idx, reward) in enumerate(zip(path, steps)):
            c_type = cell_types[node_idx]
            color = type_to_color[c_type]

            x, y = coords[node_idx, 0], coords[node_idx, 1]

            # Plot nodes
            ax.scatter(x, y, color=color, s=90, edgecolors='white', linewidth=1.2, zorder=5)

            # Dynamic text color: <=0 red, low orange, high green
            if step_idx == 0:
                text_label = "Start"
                text_color = "#333333"
            else:
                text_label = f"{reward:+.1f}"
                if reward <= 0: text_color = "#e74c3c"       # Red: Penalty/Obstacle
                elif reward < 1.5: text_color = "#e67e22"    # Orange: Slow progression
                else: text_color = "#2ca02c"                 # Green: Fast acceleration

            # Annotate values with white stroke for readability
            ax.text(x, y + 0.15, text_label, fontsize=10, color=text_color, fontweight='bold', 
                    zorder=10, ha='center', va='bottom',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        # Do not append the endpoint cell type to the target title.
        ax.set_title(f"Target node: {end}\nSpatial Reward vs Cell Types", fontsize=14, pad=10)
        ax.axis('off')

    # Generate global legend in the upper right of the first subplot
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=ctype,
                              markerfacecolor=color, markersize=10)
                       for ctype, color in type_to_color.items()]

    # Show legend only on the first subplot
    axes[0].legend(handles=legend_elements, loc='upper right', title="Cell Types", 
                   frameon=False, fontsize=11, title_fontsize=12)

    # Remove empty subplots
    for i in range(n_ends, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_branching_points_on_umap(q_learner, paths, coords, cell_types):
    """
    Visualize branching points and transition rewards on UMAP.
    - Branching point circles have a red border.
    - Displays cell type colors, names, and step rewards for branching points, predecessors, and successors.
    - 🌟 Uses adjustText to automatically prevent text label overlap.
    """
    if cell_types is None:
        print("Error: cell_types array must be provided.")
        return

    # 1. Extract step rewards and identify branching points
    path_totals, path_steps = q_learner.calculate_path_rewards(paths)

    out_edges = {}
    for end_node, path in paths.items():
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if u not in out_edges: out_edges[u] = set()
            out_edges[u].add(v)

    branching_points = [u for u, targets in out_edges.items() if len(targets) > 1]

    if not branching_points:
        print("No multi-branch branching points detected!")
        return

    # 2. Prepare plot and color dictionary
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_types = np.unique(cell_types)
    color_palette = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: color_palette[i] for i, t in enumerate(unique_types)}

    # 3. Plot UMAP gray background and dashed trajectory lines
    ax.scatter(coords[:, 0], coords[:, 1], c='#e0e0e0', alpha=0.3, s=15, edgecolors='none')
    for end_node, path in paths.items():
        path_coords = coords[path]
        ax.plot(path_coords[:, 0], path_coords[:, 1], color='#cccccc', linewidth=2, linestyle='--', zorder=1)

    # 4. Collect key nodes and edges to draw
    nodes_to_draw = {}  
    edges_to_draw = set() 

    for bp in branching_points:
        for end_node, path in paths.items():
            if bp in path:
                idx = path.index(bp)
                nodes_to_draw[bp] = path_steps[end_node][idx]
                if idx > 0:
                    prev = path[idx - 1]
                    nodes_to_draw[prev] = path_steps[end_node][idx - 1]
                    edges_to_draw.add((prev, bp))
                break

        for v in out_edges[bp]:
            for end_node, path in paths.items():
                if bp in path and v in path:
                    nodes_to_draw[v] = path_steps[end_node][path.index(v)]
                    edges_to_draw.add((bp, v))
                    break

    # 5. Render edge arrows
    for u, v in edges_to_draw:
        ax.annotate("", xy=coords[v], xytext=coords[u],
                    arrowprops=dict(arrowstyle="-|>", color='#555555', lw=2.5, mutation_scale=15), zorder=5)

    # 🌟 Prepare a list to collect all text objects
    texts = []

    # Record point coordinates to avoid text overlapping with dots
    point_xs, point_ys = [], []

    # 6. Render key nodes
    for node, reward in nodes_to_draw.items():
        ctype = cell_types[node]
        color = type_to_color[ctype]
        x, y = coords[node, 0], coords[node, 1]

        point_xs.append(x)
        point_ys.append(y)

        if node in branching_points:
            ax.scatter(x, y, color=color, s=150, edgecolors="#FF0000", linewidth=1.5, zorder=9)
        else:
            ax.scatter(x, y, color=color, s=150, edgecolors='white', linewidth=1.5, zorder=8)

        # Annotate reward values
        text_color = "#e74c3c" if reward <= 0 else ("#e67e22" if reward < 1.5 else "#2ca02c")

        # 🌟 Note: Instead of forcing offsets/alignments, store text objects in the texts list
        t = ax.text(x, y, f"{reward:+.4f}", fontsize=12, color=text_color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=15)
        texts.append(t)

    # 🌟 7. Where the magic happens: call adjustText for automatic layout
    if texts:
        adjust_text(texts, 
                    x=point_xs, y=point_ys,           # Avoid these points
                    force_text=(0.5, 0.8),            # Repulsion between texts
                    force_points=(1.0, 1.0),          # Repulsion between text and points
                    expand_points=(1.5, 1.5)          # Expand point influence
                    # arrowprops=dict(arrowstyle='-', color='#999999', lw=1.5, alpha=0.7, zorder=10) # Draw connection lines if needed
                     ) 

    # 8. Formatting and legend
    # ax.set_title("Trajectory Branching Points Visualization", fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=ctype,
                              markerfacecolor=color, markersize=10)
                       for ctype, color in type_to_color.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), 
              title="Cell Types", frameon=False, fontsize=12, title_fontsize=13)

    plt.tight_layout()
    plt.show()


# ---- configuration export helpers (cell 47/48) ----
import gc
import pandas as pd
import time


def _configuration_json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def export_five_dataset_preprocessing_configuration(
    output_dir=None,
    seed=42,
    run_preprocessing=True,
    file_stem="2026-08-17_five_dataset_preprocessing_config",
):
    """Export configured inputs and actual/expected preprocessing outputs."""
    output_dir = Path(output_dir or PROJECT_ROOT).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    structured = []

    for dataset_key, original_config in DATASET_CONFIGS.items():
        config = copy.deepcopy(original_config)
        input_path = resolve_input(config)
        input_adata = sc.read_h5ad(input_path)
        input_min, input_max = get_matrix_min_max(input_adata)
        input_obsm_shapes = {
            key: list(map(int, value.shape))
            for key, value in input_adata.obsm.items()
        }
        input_record = {
            "path": str(input_path),
            "shape": [int(input_adata.n_obs), int(input_adata.n_vars)],
            "x_min": input_min,
            "x_max": input_max,
            "obsm_shapes": input_obsm_shapes,
            "label_column": config["label_column"],
        }

        output_record = {
            "shape": None,
            "x_min": None,
            "x_max": None,
            "pca_shape": None,
            "pca_params": {},
            "neighbors_params": {},
            "umap_shape": None,
            "umap_params": {},
            "preprocessing_seconds": None,
            "completed_before_inference": False,
        }

        del input_adata
        gc.collect()

        if run_preprocessing:
            started = time.perf_counter()
            prepared = load_prepared_dataset(input_path, config, seed=seed)
            processed = prepared["adata"]
            output_min, output_max = get_matrix_min_max(processed)
            output_record.update({
                "shape": [int(processed.n_obs), int(processed.n_vars)],
                "x_min": output_min,
                "x_max": output_max,
                "pca_shape": (
                    list(map(int, processed.obsm["X_pca"].shape))
                    if "X_pca" in processed.obsm else None
                ),
                "pca_params": dict(processed.uns.get("pca", {}).get("params", {})),
                "neighbors_params": dict(
                    processed.uns.get("neighbors", {}).get("params", {})
                ),
                "umap_shape": (
                    list(map(int, processed.obsm["X_umap"].shape))
                    if "X_umap" in processed.obsm else None
                ),
                "umap_params": dict(processed.uns.get("umap", {}).get("params", {})),
                "preprocessing_seconds": time.perf_counter() - started,
                "completed_before_inference": bool(
                    prepared["preprocessing_audit"].get(
                        "completed_before_inference", False
                    )
                ),
            })
            del prepared, processed
            gc.collect()

        record = {
            "dataset": dataset_key,
            "title": config["title"],
            "evaluation_mode": config.get("trajectory_reference_kind"),
            "input": input_record,
            "source_expression_input": config.get("source_expression_input", str(input_path)),
            "preprocessing_profile": config.get(
                "preprocessing_profile", "reuse_stored_representation"
            ),
            "preprocessing_steps": config.get("preprocessing_steps", []),
            "reference_preprocessing_recipe": config.get("reference_preprocessing_recipe", []),
            "output": output_record,
            "model_coordinate_key": config["model_coordinate_key"],
            "model_dimensions": list(config["model_dimensions"]),
            "plot_coordinate_key": config["plot_coordinate_key"],
            "plot_dimensions": list(config["plot_dimensions"]),
            "endpoint_knn_k": 20,
            "early_label": str(config["early_label"]),
            "target_lineages": list(map(str, config.get("target_lineages", []))),
        }
        structured.append(record)
        rows.append({
            "dataset": dataset_key,
            "title": config["title"],
            "evaluation_mode": config.get("trajectory_reference_kind"),
            "input_path": str(input_path),
            "source_expression_input": config.get("source_expression_input", str(input_path)),
            "input_shape": json.dumps(input_record["shape"]),
            "input_x_range": json.dumps([input_min, input_max]),
            "label_column": config["label_column"],
            "early_label": str(config["early_label"]),
            "target_lineages": json.dumps(record["target_lineages"], ensure_ascii=False, default=_configuration_json_default),
            "endpoint_knn_k": 20,
            "preprocessing_profile": record["preprocessing_profile"],
            "preprocessing_steps": json.dumps(
                record["preprocessing_steps"], ensure_ascii=False, default=_configuration_json_default
            ),
            "reference_preprocessing_recipe": json.dumps(
                record["reference_preprocessing_recipe"], ensure_ascii=False, default=_configuration_json_default
            ),
            "output_shape": json.dumps(output_record["shape"]),
            "output_x_range": json.dumps(
                [output_record["x_min"], output_record["x_max"]]
            ),
            "pca_shape": json.dumps(output_record["pca_shape"]),
            "pca_params": json.dumps(output_record["pca_params"], ensure_ascii=False, default=_configuration_json_default),
            "neighbors_params": json.dumps(
                output_record["neighbors_params"], ensure_ascii=False, default=_configuration_json_default
            ),
            "umap_shape": json.dumps(output_record["umap_shape"]),
            "umap_params": json.dumps(output_record["umap_params"], ensure_ascii=False, default=_configuration_json_default),
            "model_coordinate_key": config["model_coordinate_key"],
            "model_dimensions": json.dumps(record["model_dimensions"]),
            "plot_coordinate_key": config["plot_coordinate_key"],
            "plot_dimensions": json.dumps(record["plot_dimensions"]),
            "completed_before_inference": output_record[
                "completed_before_inference"
            ],
            "preprocessing_seconds": output_record["preprocessing_seconds"],
        })

    table = pd.DataFrame(rows)
    csv_path = output_dir / f"{file_stem}.csv"
    json_path = output_dir / f"{file_stem}.json"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("Saved:", csv_path)
    print("Saved:", json_path)
    return table, structured, csv_path, json_path
