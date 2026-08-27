# Canonical source notebook: 2026-08-17_scCGRL_five_datasets_v1.ipynb
# Notebook date/version: 2026-08-17 / CODE_REVISION 1.6
# Source cell: index 11 / order 12
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compute_branch_pseudotimes(dijkstra_paths, n_cells):
    """Compute branch pseudotime (0-1 scale)."""
    branch_results = {}
    print(f"Computing pseudotime for {len(dijkstra_paths)} branches...")

    for i, (endpoint, path) in enumerate(dijkstra_paths.items()):
        if len(path) < 2:
            continue

        branch_pt = np.full(n_cells, -1.0, dtype=np.float32)
        for idx, cell in enumerate(path):
            branch_pt[cell] = idx / (len(path) - 1)

        branch_mask = branch_pt >= 0

        branch_results[i] = {
            'pseudotime': branch_pt,
            'mask': branch_mask,
            'path': path,
            'length': len(path),
            'endpoint_cell': path[-1],
            'original_endpoint': endpoint
        }

    return branch_results


def compute_graph_distance_global_pseudotime(dijkstra_paths, umap_coords, n_cells, knn_graph):
    """Compute global pseudotime using graph distance ratios."""
    if not dijkstra_paths:
        return None, None, None

    # 1. Find start node
    all_paths = list(dijkstra_paths.values())
    start_cells = set(all_paths[0])
    for path in all_paths[1:]:
        start_cells.intersection_update(set(path))
    start_cell = min(start_cells) if start_cells else (all_paths[0][0] if all_paths[0] else 0)

    # 2. Get distances for endpoints
    endpoints = list(dijkstra_paths.keys())
    endpoint_distances = {}
    for endpoint in endpoints:
        try:
            distance = nx.shortest_path_length(knn_graph, start_cell, endpoint, weight='weight')
            if distance < np.inf:
                endpoint_distances[endpoint] = distance
            else:
                for path in all_paths:
                    if path[-1] == endpoint:
                        endpoint_distances[endpoint] = len(path)
                        break
        except:
            for path in all_paths:
                if path[-1] == endpoint:
                    endpoint_distances[endpoint] = len(path)
                    break

    if not endpoint_distances:
        return None, None, None

    max_distance = max(endpoint_distances.values())

    # 3. Calculate target ratios
    endpoint_ratios = {}
    for endpoint, dist in endpoint_distances.items():
        endpoint_ratios[endpoint] = dist / max_distance if max_distance > 0 else 1.0

    # 4. Assign pseudotime
    global_pt = np.full(n_cells, -1.0, dtype=np.float32)
    for endpoint, path in dijkstra_paths.items():
        if len(path) < 2: continue
        target_ratio = endpoint_ratios.get(endpoint, 1.0)

        for cell in path:
            if cell == start_cell:
                global_pt[cell] = 0.0
            elif cell == endpoint:
                global_pt[cell] = target_ratio
            else:
                if cell in path:
                    cell_idx = path.index(cell)
                    cell_ratio = (cell_idx / (len(path) - 1)) * target_ratio
                    if global_pt[cell] >= 0:
                        global_pt[cell] = (global_pt[cell] + cell_ratio) / 2
                    else:
                        global_pt[cell] = cell_ratio

    # 5. Resolve bifurcation overlaps
    cell_path_count = {}
    for path in dijkstra_paths.values():
        for cell in path:
            cell_path_count[cell] = cell_path_count.get(cell, 0) + 1

    for cell, count in cell_path_count.items():
        if count > 1 and global_pt[cell] >= 0:
            values = []
            for endpoint, path in dijkstra_paths.items():
                if cell in path:
                    cell_idx = path.index(cell)
                    target_ratio = endpoint_ratios.get(endpoint, 1.0)
                    values.append((cell_idx / (len(path) - 1)) * target_ratio)
            if values:
                global_pt[cell] = np.mean(values)

    # Apply limits
    valid_mask = global_pt >= 0
    if valid_mask.any():
        global_pt[valid_mask] = np.clip(global_pt[valid_mask], 0.0, 1.0)

    global_pt[start_cell] = 0.0
    for endpoint, ratio in endpoint_ratios.items():
        if endpoint < n_cells:
            global_pt[endpoint] = ratio

    return global_pt, valid_mask, start_cell


def compute_enhanced_rf_pseudotime_with_global(global_pt, umap_coords):
    """Train RF model using global pseudotime."""
    n_cells = len(umap_coords)
    print("Training RF model with global pseudotime...")

    if global_pt is None:
        return None, None

    train_mask = global_pt >= 0
    if train_mask.sum() < 20:
        return global_pt, train_mask

    X_train = umap_coords[train_mask]
    y_train = global_pt[train_mask]

    try:
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        rf_pt = rf_model.predict(umap_coords)
        rf_pt = np.clip(rf_pt, 0.0, 1.0001)

        min_pt_idx = np.argmin(global_pt[train_mask])
        start_cell = np.where(train_mask)[0][min_pt_idx]
        rf_pt[start_cell] = 0.0

        print(f"RF pseudotime range: [{rf_pt.min():.3f}, {rf_pt.max():.3f}]")
        return rf_pt, np.ones(n_cells, dtype=bool)

    except Exception as e:
        print(f"RF training failed: {e}")
        return global_pt, train_mask


def evaluate_pseudotime(pseudotime, true_labels, cell_types, mask):
    """Evaluate pseudotime accuracy."""
    # Filter valid depth map cells
    valid_mask = mask & (true_labels != -1)

    pt_traj = pseudotime[valid_mask]
    y_traj = true_labels[valid_mask]
    cell_types_traj = cell_types[valid_mask]

    if len(pt_traj) < 2 or len(np.unique(cell_types_traj)) < 2:
        return [np.nan] * 4

    try:
        # Proportional binning
        props = pd.Series(cell_types_traj).value_counts(normalize=True).sort_index()
        bins = [0.0] + props.cumsum().tolist()
        bins[-1] = 1.0001

        bins = np.unique(np.sort(bins))
        pred_bins = pd.cut(pt_traj, bins=bins, right=False, duplicates='drop')
        unique_pred = np.unique(pred_bins)
        label_map = {l: i for i, l in enumerate(unique_pred)}
        pred_encoded = np.array([label_map.get(l, -1) for l in pred_bins])
        valid = pred_encoded != -1

        if valid.sum() < 2:
            return [np.nan] * 4

        ari = adjusted_rand_score(y_traj[valid], pred_encoded[valid])
        nmi = normalized_mutual_info_score(y_traj[valid], pred_encoded[valid])

        # Calculate correlations
        pcc = abs(pearsonr(pt_traj, y_traj)[0])
        spr = abs(spearmanr(pt_traj, y_traj)[0])

        return [ari, nmi, pcc, spr]
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return [np.nan] * 4


def evaluate_all_branches(branch_results, true_labels, cell_types):
    """Evaluate all branch pseudotimes."""
    branch_metrics = {}
    print("\nEvaluating branch pseudotimes:")

    for branch_id, branch_data in branch_results.items():
        metrics = evaluate_pseudotime(
            branch_data['pseudotime'],
            true_labels,
            cell_types,
            branch_data['mask']
        )

        branch_metrics[branch_id] = {
            'metrics': metrics,
            'length': branch_data['length'],
            'endpoint': branch_data['original_endpoint']
        }

        print(f"  Branch {branch_id+1} (Length {branch_data['length']}, End {branch_data['original_endpoint']}):")
        print(f"    ARI={metrics[0]:.4f}, NMI={metrics[1]:.4f}, PCC={metrics[2]:.4f}, Spearman={metrics[3]:.4f}")

    if branch_metrics:
        all_metrics = np.array([data['metrics'] for data in branch_metrics.values()])
        avg_metrics = np.nanmean(all_metrics, axis=0)
        print(f"\nAverage Branch Metrics:")
        print(f"  ARI={avg_metrics[0]:.4f}, NMI={avg_metrics[1]:.4f}, PCC={avg_metrics[2]:.4f}, Spearman={avg_metrics[3]:.4f}")
    else:
        avg_metrics = np.array([np.nan]*4)

    return branch_metrics, avg_metrics


def main_final_analysis_enhanced(q_learner, umap_coords, cell_types, knn_graph):
    """Main pipeline for enhanced pseudotime analysis."""
    print("=" * 70)
    print("Enhanced Pseudotime Analysis")
    print("=" * 70)

    # 1. Fetch trajectories
    print("\n1. Fetching RL trajectories...")
    sparse_graph = q_learner.build_sparse_graph()
    dijkstra_paths = q_learner.find_shortest_paths(sparse_graph)

    print(f"Found {len(dijkstra_paths)} paths:")
    for i, (endpoint, path) in enumerate(dijkstra_paths.items()):
        print(f"  Path {i+1}: End {endpoint}, Length={len(path)}")

    # 2. Branch pseudotime
    n_cells = len(umap_coords)
    branch_results = compute_branch_pseudotimes(dijkstra_paths, n_cells)

    # 3. Global pseudotime
    print("\n3. Computing global pseudotime...")
    global_pt, global_mask, start_cell = compute_graph_distance_global_pseudotime(
        dijkstra_paths, umap_coords, n_cells, knn_graph
    )

    # 4. RF pseudotime
    print("\n4. Computing enhanced RF pseudotime...")
    rf_pt, rf_mask = compute_enhanced_rf_pseudotime_with_global(global_pt, umap_coords)

    # 5. Evaluate
    print("\n5. Evaluating pseudotime...")

    depth_map = {"0": 0, "1": 1, "2": 2, "3": 2}
    true_labels = np.array([depth_map.get(str(t), -1) for t in cell_types])

    missing_cells = (true_labels == -1).sum()
    if missing_cells > 0:
        print(f"Warning: {missing_cells} cells not in depth_map, ignoring in evaluation.")

    branch_metrics, avg_branch_metrics = evaluate_all_branches(
        branch_results, true_labels, cell_types
    )

    global_metrics = None
    if global_pt is not None and global_mask is not None:
        global_metrics = evaluate_pseudotime(global_pt, true_labels, cell_types, global_mask)
        print(f"\nGlobal Pseudotime Metrics:")
        print(f"  ARI={global_metrics[0]:.4f}, NMI={global_metrics[1]:.4f}")
        print(f"  PCC={global_metrics[2]:.4f}, Spearman={global_metrics[3]:.4f}")

    rf_metrics = None
    if rf_pt is not None:
        rf_metrics = evaluate_pseudotime(rf_pt, true_labels, cell_types, rf_mask)
        print(f"\nRF Pseudotime Metrics:")
        print(f"  ARI={rf_metrics[0]:.4f}, NMI={rf_metrics[1]:.4f}")
        print(f"  PCC={rf_metrics[2]:.4f}, Spearman={rf_metrics[3]:.4f}")

    # 7. Summary
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)

    comparison_data = []

    for branch_id, data in branch_metrics.items():
        comparison_data.append({
            'Method': f'Branch {branch_id+1} PT',
            'Type': 'Branch PT',
            'ARI': f"{data['metrics'][0]:.4f}",
            'NMI': f"{data['metrics'][1]:.4f}",
            'Pearson': f"{data['metrics'][2]:.4f}",
            'Spearman': f"{data['metrics'][3]:.4f}",
            'Length': data['length'],
            'Endpoint': data['endpoint']
        })

    comparison_data.append({
        'Method': 'Avg Branch PT',
        'Type': 'Branch PT',
        'ARI': f"{avg_branch_metrics[0]:.4f}",
        'NMI': f"{avg_branch_metrics[1]:.4f}",
        'Pearson': f"{avg_branch_metrics[2]:.4f}",
        'Spearman': f"{avg_branch_metrics[3]:.4f}",
        'Length': '-',
        'Endpoint': '-'
    })

    if global_metrics is not None:
        comparison_data.append({
            'Method': 'Global PT',
            'Type': 'Global PT',
            'ARI': f"{global_metrics[0]:.4f}",
            'NMI': f"{global_metrics[1]:.4f}",
            'Pearson': f"{global_metrics[2]:.4f}",
            'Spearman': f"{global_metrics[3]:.4f}",
            'Length': '-',
            'Endpoint': f'Start:{start_cell}'
        })

    if rf_metrics is not None:
        comparison_data.append({
            'Method': 'Enhanced RF PT',
            'Type': 'Global PT',
            'ARI': f"{rf_metrics[0]:.4f}",
            'NMI': f"{rf_metrics[1]:.4f}",
            'Pearson': f"{rf_metrics[2]:.4f}",
            'Spearman': f"{rf_metrics[3]:.4f}",
            'Length': '-',
            'Endpoint': '-'
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\nEvaluation Comparison:")
    print(df_comparison.to_string(index=False))

    if len(comparison_data) > 0:
        print(f"\nRecommended Method:")
        df_temp = pd.DataFrame(comparison_data)
        numeric_cols = ['ARI', 'NMI', 'Pearson', 'Spearman']

        for col in numeric_cols:
            df_temp[col] = df_temp[col].astype(float)

        df_temp['Score'] = df_temp[numeric_cols].mean(axis=1)
        best_idx = df_temp['Score'].idxmax()
        best_method = df_temp.loc[best_idx]

        print(f"  {best_method['Method']} (Score: {best_method['Score']:.4f})")
        print(f"    ARI={best_method['ARI']:.4f}, NMI={best_method['NMI']:.4f}")
        print(f"    Pearson={best_method['Pearson']:.4f}, Spearman={best_method['Spearman']:.4f}")

    return {
        'branch_results': branch_results,
        'branch_metrics': branch_metrics,
        'avg_branch_metrics': avg_branch_metrics,
        'global_pseudotime': global_pt,
        'global_metrics': global_metrics,
        'rf_pseudotime': rf_pt,
        'rf_metrics': rf_metrics,
        'start_cell': start_cell,
        'comparison_df': df_comparison
    }
