# Canonical source notebook: 2026-08-17_scCGRL_five_datasets_v1.ipynb
# Notebook date/version: 2026-08-17 / CODE_REVISION 1.6
# Source cell: index 7 / order 8
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform

def find_start_and_endpoints(pca_data, labels, early_cell_label=0):
    """
    Find start and endpoints in multi-branch cell trajectory data.
    """
    if isinstance(pca_data, pd.DataFrame):
        pca_data = pca_data.values
    labels = np.array(labels)
    n_samples = len(pca_data)

    # 1. Determine minimal K and build connected KNN graph
    k_min = find_minimal_connected_k(pca_data, min_k=20, max_k=30)
    print(f"Using minimal connected K: {k_min}")
    knn_adj = build_connected_knn_graph(pca_data, k_min)
    knn_graph = build_knn_graph_from_adjacency(knn_adj, pca_data)

    # 2. Identify early cells
    early_cells_mask = labels == early_cell_label
    early_indices = np.where(early_cells_mask)[0]

    if len(early_indices) == 0:
        raise ValueError("No early cells found. Please check labels.")

    # 3. Select start point (maximize graph distance)
    start_index = find_start_point(knn_graph, early_indices, n_samples)
    print(f"Selected start index: {start_index}")

    # 4. Find all candidate endpoints
    target_endpoint_count = n_samples - 1 
    candidate_endpoints, candidate_indices = find_all_candidate_endpoints(
        knn_graph, pca_data, labels, start_index, early_cell_label, target_endpoint_count)
    print(f"Candidate endpoints found: {len(candidate_indices)}")

    # 5. Filter endpoints (remove subsets)
    filtered_endpoint_indices = filter_endpoints_by_label_progression(
        knn_graph, pca_data, labels, start_index, candidate_indices)
    filtered_endpoints = [pca_data[idx] for idx in filtered_endpoint_indices]
    print(f"Endpoints after filtering: {len(filtered_endpoint_indices)}")

    return {
        'start_point': pca_data[start_index],
        'start_index': start_index,
        'endpoints': filtered_endpoints,
        'endpoint_indices': filtered_endpoint_indices,
        'knn_adj': knn_adj,
        'knn_graph': knn_graph,
        'k_value': k_min,
        'all_candidate_endpoints': candidate_endpoints,
        'all_candidate_indices': candidate_indices
    }

def find_minimal_connected_k(pca_data, min_k=20, max_k=30):
    """Find the minimal K to ensure graph connectivity."""
    for k in range(int(min_k), int(max_k) + 1):
        try:
            knn_adj = build_knn_adjacency_matrix(pca_data, k)
            if is_graph_connected(knn_adj):
                return k
        except:
            continue
    return min(max_k, len(pca_data) - 1)

def build_connected_knn_graph(data, k):
    """Build a connected KNN graph."""
    knn_adj = build_knn_adjacency_matrix(data, k)

    if not is_graph_connected(knn_adj):
        knn_adj = connect_disconnected_components(knn_adj, data)

    return knn_adj

def build_knn_adjacency_matrix(data, k):
    """Build KNN adjacency matrix based on Euclidean distance."""
    n_samples = data.shape[0]

    # Use complete graph if K is too large
    if k >= n_samples - 1:
        adj_matrix = np.ones((n_samples, n_samples), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    dist_matrix = squareform(pdist(data, 'euclidean'))
    adj_matrix = np.zeros((n_samples, n_samples), dtype=int)

    for i in range(n_samples):
        distances = dist_matrix[i]
        indices = np.argsort(distances)[1:k+1] # Skip self
        adj_matrix[i, indices] = 1
        adj_matrix[indices, i] = 1 # Symmetric

    return adj_matrix

def is_graph_connected(adj_matrix):
    """Check if the graph is fully connected."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.is_connected(G)

def connect_disconnected_components(adj_matrix, data):
    """Connect disconnected graph components."""
    n_samples = adj_matrix.shape[0]
    G = nx.from_numpy_array(adj_matrix)
    components = list(nx.connected_components(G))

    if len(components) == 1:
        return adj_matrix

    dist_matrix = squareform(pdist(data, 'euclidean'))

    while len(components) > 1:
        min_dist = np.inf
        min_pair = None

        # Find closest point pair between components
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                for u in components[i]:
                    for v in components[j]:
                        if dist_matrix[u, v] < min_dist:
                            min_dist = dist_matrix[u, v]
                            min_pair = (u, v)

        # Connect the closest pair
        if min_pair:
            u, v = min_pair
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1

        # Recalculate components
        G = nx.from_numpy_array(adj_matrix)
        components = list(nx.connected_components(G))

    return adj_matrix

def build_knn_graph_from_adjacency(knn_adj, pca_data):
    """Build a weighted graph from adjacency matrix."""
    n_samples = knn_adj.shape[0]
    G = nx.Graph()

    for i in range(n_samples):
        G.add_node(i)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if knn_adj[i, j] > 0:
                distance = np.linalg.norm(pca_data[i] - pca_data[j])
                G.add_edge(i, j, weight=distance)

    return G

def find_start_point(knn_graph, early_indices, n_nodes):
    """Select start point by maximizing average graph distance."""
    max_avg_distance = -1
    best_start_index = early_indices[0]

    for early_idx in early_indices:
        graph_distances = compute_graph_distances(knn_graph, early_idx, n_nodes)
        valid_distances = graph_distances[np.isfinite(graph_distances)]

        if len(valid_distances) > 0:
            avg_distance = np.mean(valid_distances)
            if avg_distance > max_avg_distance:
                max_avg_distance = avg_distance
                best_start_index = early_idx

    return best_start_index

def compute_graph_distances(graph, source_idx, n_nodes):
    """Compute shortest path lengths in the graph."""
    graph_distances = np.full(n_nodes, np.inf)

    try:
        shortest_paths = nx.single_source_dijkstra_path_length(graph, source_idx)
        for node, distance in shortest_paths.items():
            graph_distances[node] = distance
    except:
        try:
            shortest_paths = nx.single_source_shortest_path_length(graph, source_idx)
            for node, distance in shortest_paths.items():
                graph_distances[node] = distance
        except:
            pass

    return graph_distances

def find_all_candidate_endpoints(knn_graph, pca_data, labels, start_index, early_cell_label, target_count):
    """Iteratively find candidate endpoints by selecting the farthest nodes."""
    n_cells = len(pca_data)
    graph_distances = compute_graph_distances(knn_graph, start_index, n_cells)

    # Exclude start point and early cells
    other_mask = (labels != early_cell_label) & (np.arange(len(labels)) != start_index)
    other_indices = np.where(other_mask)[0]
    other_labels = labels[other_mask]

    if len(other_indices) == 0:
        return [], []

    all_unique_labels = np.unique(labels)
    unique_labels = [label for label in all_unique_labels if label != early_cell_label]
    target_count = min(len(unique_labels), len(other_indices), target_count)

    endpoints = []
    endpoint_indices = []
    remaining_indices = other_indices.copy()
    remaining_labels = other_labels.copy()

    # Iteratively select the farthest point
    for i in range(target_count):
        if len(remaining_indices) == 0:
            break

        remaining_distances = graph_distances[remaining_indices]
        if len(remaining_distances) == 0 or np.all(np.isinf(remaining_distances)):
            break

        max_idx = np.nanargmax(remaining_distances)
        farthest_idx = remaining_indices[max_idx]
        farthest_label = labels[farthest_idx]

        endpoints.append(pca_data[farthest_idx])
        endpoint_indices.append(farthest_idx)

        # Exclude cells of the selected label
        keep_mask = remaining_labels != farthest_label
        remaining_indices = remaining_indices[keep_mask]
        remaining_labels = remaining_labels[keep_mask]

    print(f"Selection complete: {len(endpoint_indices)} endpoints found.")
    return endpoints, endpoint_indices

def filter_endpoints_by_label_progression(knn_graph, pca_data, labels, start_index, endpoint_indices):
    """Remove endpoints whose path labels are subsets of longer paths."""
    if len(endpoint_indices) <= 1:
        return endpoint_indices

    n_cells = len(pca_data)
    endpoint_distances = []
    for end_idx in endpoint_indices:
        distance = compute_graph_distances(knn_graph, start_index, n_cells)[end_idx]
        endpoint_distances.append(distance)

    # Sort descending by distance
    sorted_indices = np.argsort(endpoint_distances)[::-1]
    sorted_endpoints = [endpoint_indices[i] for i in sorted_indices]

    # Get label sets for each path
    path_label_sets = {}
    for end_idx in sorted_endpoints:
        path_label_sets[end_idx] = get_path_labels(knn_graph, start_index, end_idx, labels)

    kept_endpoints = []
    for i, current_end in enumerate(sorted_endpoints):
        current_labels = path_label_sets[current_end]
        should_keep = True

        # Check if subset of any kept path
        for kept_end in kept_endpoints:
            if current_labels.issubset(path_label_sets[kept_end]):
                should_keep = False
                break

        if should_keep:
            kept_endpoints.append(current_end)

    return kept_endpoints

def get_path_labels(knn_graph, start_index, end_index, labels):
    """Get the set of unique labels along the shortest path."""
    try:
        shortest_path = nx.shortest_path(knn_graph, start_index, end_index)
        return set(labels[cell_idx] for cell_idx in shortest_path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {labels[start_index], labels[end_index]}
