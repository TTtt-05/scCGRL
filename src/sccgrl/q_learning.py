# Canonical source notebook: 2026-08-17_scCGRL_five_datasets_v1.ipynb
# Notebook date/version: 2026-08-17 / CODE_REVISION 1.6
# Source cell: index 9 / order 10
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class QLearningPathFinder:
    def __init__(self, adj_matrix, coords, start_idx, end_indices,
                 epsilon=0.5, alpha=0.1, gamma=0.9,
                 n_episodes=1000, converge_threshold=1e-5,
                 k=15, weight_epsilon=1e-5):
        """
        Initialize Q-learning Path Finder.
        """
        self.adj_matrix = adj_matrix
        self.coords = coords
        self.n_cells = adj_matrix.shape[0]
        self.start = start_idx
        self.ends = set(end_indices)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.converge_threshold = converge_threshold
        self.k = k
        self.weight_epsilon = weight_epsilon

        self.neighbors = {i: np.where(adj_matrix[i] > 0)[0].tolist() for i in range(self.n_cells)}

        self.Q = defaultdict(dict)
        for i in range(self.n_cells):
            for j in self.neighbors[i]:
                self.Q[i][j] = 0.0

        self.reward_history = []
        self.q_diff_history = []
        self.reach_counts = {end: 0 for end in self.ends}

        self.max_dist = np.max(np.linalg.norm(coords[:, None] - coords, axis=2))

        self._precompute_end_distances()
        self._precompute_k_neighbor_distances()

        self.endpoint_reward_history = {end: [] for end in self.ends}

    def _get_end_name(self, idx):
        """Return an endpoint identifier without consulting cell-type labels."""
        return str(idx)

    def _precompute_end_distances(self):
        end_coords = self.coords[list(self.ends)]
        dists = np.linalg.norm(self.coords[:, None] - end_coords, axis=2)
        self.end_distances = np.min(dists, axis=1)
        self.max_end_distance = np.max(dists)

    def _precompute_k_neighbor_distances(self):
        dist_matrix = np.linalg.norm(self.coords[:, None] - self.coords, axis=2)
        sorted_dists = np.sort(dist_matrix, axis=1)[:, 1:]
        k_idx = min(self.k-1, sorted_dists.shape[1]-1)
        self.k_neighbor_distances = sorted_dists[:, k_idx]
        self.max_k_neighbor_distance = np.max(self.k_neighbor_distances)

    def _get_adaptive_weight(self, cell_idx):
        proximity_factor = (self.max_end_distance - self.end_distances[cell_idx] + self.weight_epsilon) / \
                           (self.max_end_distance + self.weight_epsilon)
        density_factor = (self.max_k_neighbor_distance + self.weight_epsilon) / \
                         (self.k_neighbor_distances[cell_idx] + self.weight_epsilon)
        return proximity_factor * density_factor

    def _get_reward(self, current, next_cell, visited):
        R_end = 100.0
        R_base = 2.0
        R_dist_penalty_coeff = -0.5

        if next_cell in self.ends:
            return R_end

        current_weight = self._get_adaptive_weight(current)
        step_reward = current_weight * R_base

        if next_cell in visited:
            return -(step_reward + 1.0)

        step_dist = np.linalg.norm(self.coords[current] - self.coords[next_cell])
        dist_penalty = R_dist_penalty_coeff * (step_dist / (self.max_dist + self.weight_epsilon))

        return step_reward + dist_penalty

    def _is_terminal(self, state):
        return state in self.ends

    def train(self, verbose=True):
        initial_epsilon = self.epsilon
        prev_q_values = self._get_q_values()

        for episode in range(self.n_episodes):
            state = self.start
            total_reward = 0
            steps = 0
            visited = set([state])

            while not self._is_terminal(state) and steps < self.n_cells:
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.neighbors[state])
                else:
                    q_values = {n: self.Q[state].get(n, -np.inf) for n in self.neighbors[state]}
                    action = max(q_values, key=q_values.get)

                reward = self._get_reward(state, action, visited)
                total_reward += reward

                next_q_values = [self.Q[action].get(n, 0) for n in self.neighbors[action]]
                max_next_q = max(next_q_values) if next_q_values else 0
                old_q = self.Q[state][action]
                self.Q[state][action] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

                state = action
                steps += 1
                visited.add(state)

            self.reward_history.append(total_reward)
            if state in self.ends:
                self.reach_counts[state] += 1
                self.endpoint_reward_history[state].append({
                    'episode': episode,
                    'reward': total_reward
                })

            current_q_values = self._get_q_values()
            q_diff = np.mean(np.abs(current_q_values - prev_q_values))
            self.q_diff_history.append(q_diff)
            prev_q_values = current_q_values

            decay_factor = 0.9
            min_epsilon = 0.01
            if episode % 100 == 0 and episode != 0:
                self.epsilon = max(min_epsilon, initial_epsilon * (decay_factor ** (episode // 100)))

            if verbose and (episode % 500 == 0):
                print(f"Episode {episode}: Total Reward={total_reward:.2f}, Q Diff={q_diff:.6f}, Epsilon={self.epsilon:.3f}")

            if q_diff < self.converge_threshold and episode > 100:
                if verbose:
                    print(f"Q-table converged at episode {episode}.")
                break

    def _get_q_values(self, sample_size=1000):
        q_values = []
        for s in self.Q:
            q_values.extend(self.Q[s].values())
        if len(q_values) > sample_size:
            return np.random.choice(q_values, sample_size, replace=False)
        return np.array(q_values)

    def build_sparse_graph(self):
        graph = np.zeros_like(self.adj_matrix, dtype=np.float32)
        for s in self.Q:
            for a in self.Q[s]:
                q_val = self.Q[s][a]
                graph[s, a] = 1.0 / (q_val + 1e-6) if q_val > 0 else 1000.0
        return csr_matrix(graph)

    def find_shortest_paths(self, sparse_graph=None):
        if sparse_graph is None:
            sparse_graph = self.build_sparse_graph()

        shortest_paths = {}
        dist_matrix = dijkstra(csgraph=sparse_graph, indices=self.start)

        for end in self.ends:
            path = []
            current = end
            while current != self.start:
                path.append(current)
                neighbors = self.neighbors[current]
                if not neighbors:
                    break
                current = min(neighbors, key=lambda x: dist_matrix[x] + sparse_graph[x, current])
            path.append(self.start)
            shortest_paths[end] = path[::-1]

        return shortest_paths

    def _backtrack_path_detect_loops(self, end, max_loop_tolerance=5):
        path = [end]
        current = end
        visited_counts = defaultdict(int)
        visited_counts[end] = 1
        total_loops = 0

        while current != self.start:
            candidates = self.neighbors[current]
            if not candidates:
                break

            best_prev = max(candidates, key=lambda x: self.Q[x].get(current, -np.inf))
            best_q = self.Q[best_prev].get(current, -np.inf)

            if best_q <= 0:
                print(f"Optimal previous Q-value for node {current} is {best_q:.2f}, stopping backtrack.")
                break

            if visited_counts[best_prev] > 0:
                total_loops += 1
                if visited_counts[best_prev] >= max_loop_tolerance:
                    print(f"Max loop tolerance {max_loop_tolerance} reached at node {best_prev}, stopping backtrack.")
                    break

            current = best_prev
            path.append(current)
            visited_counts[current] += 1

        return path[::-1], total_loops

    def find_paths_by_qtable(self):
        print("-" * 50)
        print("Starting Q-table backtrack path extraction...")
        print("-" * 50)

        rl_paths = {}
        loop_counts = {}

        for end in self.ends:
            path, loops = self._backtrack_path_detect_loops(end)
            rl_paths[end] = path
            loop_counts[end] = loops

            end_name = self._get_end_name(end)
            if path[0] == self.start:
                print(f"Target {end_name}: Successfully reached start | Length: {len(path)} | Loops: {loops}")
            else:
                print(f"Target {end_name}: Failed to reach start, stalled at node {path[0]} | Length: {len(path)} | Loops: {loops}")

        return rl_paths, loop_counts

    def calculate_path_rewards(self, paths):
        path_totals = {}
        path_steps = {}

        for end, path in paths.items():
            visited = set()
            total = 0
            steps = [0.0]

            visited.add(path[0])
            for i in range(len(path) - 1):
                current = path[i]
                next_cell = path[i+1]
                step_r = self._get_reward(current, next_cell, visited)
                total += step_r
                steps.append(step_r)
                visited.add(next_cell)

            path_totals[end] = total
            path_steps[end] = steps

        return path_totals, path_steps

    def print_path_rewards_report(self, paths):
        path_totals, path_steps = self.calculate_path_rewards(paths)
        print("-" * 50)
        print("Path Rewards Report")
        print("-" * 50)
        for end, path in paths.items():
            end_name = self._get_end_name(end)
            print(f"Target: {end_name} | Path Length: {len(path)} steps | Total Reward: {path_totals[end]:.2f}")
            steps_rewards = path_steps[end]
            for i in range(len(path)):
                current_node = path[i]
                current_reward = steps_rewards[i]
                if i == 0:
                    print(f"  Step {i:2d} | Start: {current_node:<4d} | Reward: 0.00")
                elif i == len(path) - 1:
                    print(f"  Step {i:2d} | End: {current_node:<4d} | Reward: {current_reward:>+7.2f}")
                else:
                    print(f"  Step {i:2d} | Move: {current_node:<4d} | Reward: {current_reward:>+7.2f}")
            print()

    def plot_reach_frequency(self):
        ends = [self._get_end_name(e) for e in self.reach_counts.keys()]
        counts = list(self.reach_counts.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(ends, counts, color='#95a5a6', edgecolor='#2c3e50', width=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + max(1, height*0.01),
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.title("Endpoint Reach Frequency during Training", fontsize=12)
        plt.xlabel("Endpoints")
        plt.ylabel("Arrival Counts")
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_endpoint_statistics(self, paths, loop_counts=None):
        path_totals, _ = self.calculate_path_rewards(paths)

        ends = [self._get_end_name(e) for e in paths.keys()]
        rewards = list(path_totals.values())

        if loop_counts is None:
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
            bars1 = ax1.bar(ends, rewards, color='#3498db', edgecolor='black', width=0.5)
            ax1.axhline(0, color='red', linewidth=1.5, linestyle='--')
            ax1.set_title("Total Rewards per Endpoint", fontsize=12)
            ax1.set_ylabel("Cumulative Reward")
            ax1.set_xlabel("Endpoint")
            for bar in bars1:
                yval = bar.get_height()
                va = 'bottom' if yval > 0 else 'top'
                offset = 1 if yval > 0 else -1
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{yval:.1f}',
                         ha='center', va=va, fontsize=10)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plt.show()
            return

        loops = list(loop_counts.values())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        bars1 = ax1.bar(ends, rewards, color='#3498db', edgecolor='black', width=0.5)
        ax1.axhline(0, color='red', linewidth=1.5, linestyle='--')
        ax1.set_title("Total Rewards per Endpoint", fontsize=12)
        ax1.set_ylabel("Cumulative Reward")
        ax1.set_xlabel("Endpoint")
        ax1.set_xticklabels(ends, rotation=15, ha='right')
        for bar in bars1:
            yval = bar.get_height()
            va = 'bottom' if yval > 0 else 'top'
            offset = 1 if yval > 0 else -1
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{yval:.1f}',
                     ha='center', va=va, fontsize=10)

        bars2 = ax2.bar(ends, loops, color='#e74c3c', edgecolor='black', width=0.5)
        ax2.set_title("Loop Counts during Backtracking", fontsize=12)
        ax2.set_ylabel("Loop Count")
        ax2.set_xlabel("Endpoint")
        ax2.set_xticklabels(ends, rotation=15, ha='right')
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{int(yval)}',
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_branch_reward_curves(self, paths):
        path_totals, path_steps = self.calculate_path_rewards(paths)

        n_ends = len(paths)
        cols = min(3, n_ends)
        rows = int(np.ceil(n_ends / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n_ends == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (end, path) in enumerate(paths.items()):
            ax = axes[idx]
            steps = path_steps[end]
            end_name = self._get_end_name(end)

            ax.plot(range(len(steps)), steps, color='#3498db', marker='o', markersize=4, linewidth=1.5)
            ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Zero Reward')

            ax.set_title(f"Target: {end_name}\nStep-by-Step Reward", fontsize=11)
            ax.set_xlabel("Step Sequence")
            ax.set_ylabel("Step Reward")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best', frameon=False)

        for i in range(n_ends, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def plot_path_rewards_annotated(self, paths):
        path_totals, path_steps = self.calculate_path_rewards(paths)

        n_ends = len(paths)
        cols = min(3, n_ends)
        rows = int(np.ceil(n_ends / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if n_ends == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (end, path) in enumerate(paths.items()):
            ax = axes[idx]
            steps = path_steps[end]
            path_coords = self.coords[path]
            end_name = self._get_end_name(end)

            ax.scatter(self.coords[:, 0], self.coords[:, 1], c='#e0e0e0', alpha=0.4, s=10, edgecolors='none')

            ax.scatter(self.coords[self.start, 0], self.coords[self.start, 1],
                       c='#d62728', marker='*', s=180, zorder=10, edgecolors='white', linewidth=1.5, label='Start')
            ax.scatter(self.coords[end, 0], self.coords[end, 1],
                       c='#1f77b4', marker='X', s=120, zorder=10, edgecolors='white', linewidth=1.5, label=f'End')

            ax.plot(path_coords[:, 0], path_coords[:, 1], color='#333333', linewidth=2, alpha=0.7, zorder=4)
            ax.scatter(path_coords[:, 0], path_coords[:, 1], c='#ff7f0e', s=25, zorder=5, edgecolors='white', linewidth=0.5)

            for i, (node_idx, reward) in enumerate(zip(path, steps)):
                x, y = self.coords[node_idx, :2]

                if i == 0:
                    text_label = "Start"
                    color = "#d62728"
                else:
                    text_label = f"{reward:+.1f}"
                    color = "#2ca02c" if reward > 0 else ("#d62728" if reward < 0 else "#7f7f7f")

                ax.text(x, y, text_label, fontsize=8, color=color, fontweight='bold',
                        zorder=15, ha='left', va='bottom',
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

            ax.set_title(f"Target: {end_name}\n(Total Reward: {path_totals[end]:.2f})", fontsize=11)
            ax.legend(loc='best', frameon=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        for i in range(n_ends, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def plot_endpoint_training_log_rewards(self):
        n_ends = len(self.ends)
        cols = min(3, n_ends)
        rows = int(np.ceil(n_ends / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n_ends == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, end in enumerate(self.ends):
            ax = axes[idx]
            history = self.endpoint_reward_history[end]

            if not history:
                ax.text(0.5, 0.5, "Never Reached", ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f"Target: {self._get_end_name(end)}")
                continue

            episodes = [item['episode'] for item in history]
            raw_rewards = np.array([item['reward'] for item in history])

            log_rewards = np.sign(raw_rewards) * np.log10(np.abs(raw_rewards) + 1)

            color = colors[idx % len(colors)]

            ax.plot(episodes, log_rewards, color=color, alpha=0.3, linewidth=1)
            ax.scatter(episodes, log_rewards, color=color, s=15, alpha=0.7)

            ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Zero Reward')

            end_name = self._get_end_name(end)
            ax.set_title(f"Target: {end_name}\nTraining History (SymLog10)", fontsize=11)
            ax.set_xlabel("Episode Number")
            ax.set_ylabel("Log Total Reward")
            ax.grid(True, linestyle='--', alpha=0.5)

            ax.text(0.05, 0.95, f"Reached: {len(history)} times",
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        for i in range(n_ends, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def plot_paths(self, paths, title="Identified Trajectories"):
        is_3d = self.coords.shape[1] >= 3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

        if is_3d:
            ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], c='lightgray', alpha=0.3, s=15)
            ax.scatter(self.coords[self.start, 0], self.coords[self.start, 1], self.coords[self.start, 2], c='red', s=250, marker='*', zorder=10, label='Start')
        else:
            ax.scatter(self.coords[:, 0], self.coords[:, 1], c='lightgray', alpha=0.4, s=20)
            ax.scatter(self.coords[self.start, 0], self.coords[self.start, 1], c='red', s=250, marker='*', zorder=10, label='Start')

        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(self.ends))))

        for i, (end, path) in enumerate(paths.items()):
            color = colors[i % len(colors)]
            end_name = self._get_end_name(end)
            if is_3d:
                ax.scatter(self.coords[end, 0], self.coords[end, 1], self.coords[end, 2], c=[color], s=150, marker='X', zorder=10, label=end_name)
                ax.plot(self.coords[path, 0], self.coords[path, 1], self.coords[path, 2], c=color, linewidth=3.5, alpha=0.8, zorder=5)
            else:
                ax.scatter(self.coords[end, 0], self.coords[end, 1], c=[color], s=150, marker='X', zorder=10, label=end_name)
                ax.plot(self.coords[path, 0], self.coords[path, 1], c=color, linewidth=3.5, alpha=0.8, zorder=5)

        ax.set_title(title, fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
