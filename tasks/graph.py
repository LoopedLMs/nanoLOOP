"""Graph shortest-path tasks.

Given an undirected graph as an edge list and a source-target query,
find the shortest path (BFS). Nodes are single lowercase letters.

Format: ab,bc,cd;a>d=abcd   (edges; query = path)
        ab,cd;a>d=n          (no path)
"""

from collections import deque

import numpy as np


def graph_path(rng: np.random.Generator, level: int) -> str:
    """Graph shortest path. Level = number of nodes (3-20)."""
    n_nodes = max(3, min(level, 20))
    nodes = [chr(ord("a") + i) for i in range(n_nodes)]

    # Random undirected edges (~40% density)
    edges: list[tuple[str, str]] = []
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.4:
                edges.append((nodes[i], nodes[j]))
                adj[nodes[i]].append(nodes[j])
                adj[nodes[j]].append(nodes[i])

    # Ensure at least one edge
    if not edges:
        i, j = sorted(rng.choice(n_nodes, size=2, replace=False))
        edges.append((nodes[i], nodes[j]))
        adj[nodes[i]].append(nodes[j])
        adj[nodes[j]].append(nodes[i])

    # Pick source and target
    idxs = rng.choice(n_nodes, size=2, replace=False)
    src, tgt = nodes[idxs[0]], nodes[idxs[1]]

    path = _bfs(adj, src, tgt)

    edge_str = ",".join(f"{a}{b}" for a, b in edges)
    answer = "".join(path) if path else "n"
    return f"{edge_str};{src}>{tgt}={answer}"


def _bfs(adj: dict[str, list[str]], src: str, tgt: str) -> list[str] | None:
    """BFS shortest path. Returns node sequence or None."""
    if src == tgt:
        return [src]
    visited = {src}
    queue: deque[tuple[str, list[str]]] = deque([(src, [src])])
    while queue:
        node, path = queue.popleft()
        for neighbor in adj[node]:
            if neighbor == tgt:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None
