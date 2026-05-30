import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def prune_graph(
    edge_index: torch.Tensor,
    risk_score: torch.Tensor,
    num_nodes: int,
    beta: float,
    min_degree: int,
    lambda_theta: float = 0.0,
    undirected: bool = True,
    protect_self_loops: bool = True,
    protect_bridges: bool = False,
) -> tuple:
    """Per-node adaptive top-budget pruning.

    Args:
        edge_index: [2, E]
        risk_score: [E] P_vu risk scores
        num_nodes: number of nodes
        beta: max pruning ratio per node
        min_degree: minimum degree to preserve
        lambda_theta: threshold = mean + lambda * std
        undirected: whether graph is undirected
        protect_self_loops: if True, never remove self-loops

    Returns:
        pruned_edge_index: [2, E']
        prune_mask: [E] True = removed
        graph_stats: dict
    """
    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]

    prune_mask = torch.zeros(E, dtype=torch.bool, device=edge_index.device)

    # Get degree per node (count incoming edges for each dst)
    degree = torch.zeros(num_nodes, device=edge_index.device)
    for i in range(E):
        degree[dst[i]] += 1

    # Group edges by destination node
    edges_by_dst = {}
    for i in range(E):
        d = dst[i].item()
        if d not in edges_by_dst:
            edges_by_dst[d] = []
        edges_by_dst[d].append(i)

    # Per-node pruning
    for v, edge_indices in edges_by_dst.items():
        if len(edge_indices) <= min_degree:
            continue

        edge_indices_t = torch.tensor(edge_indices, device=edge_index.device)
        scores_v = risk_score[edge_indices_t]

        # Budget
        bv = min(
            int(beta * len(edge_indices)),
            len(edge_indices) - min_degree,
        )
        if bv <= 0:
            continue

        # Local threshold
        threshold = scores_v.mean() + lambda_theta * scores_v.std()

        # Candidates: risk > threshold
        candidate_mask = scores_v > threshold
        candidate_indices = edge_indices_t[candidate_mask]
        candidate_scores = scores_v[candidate_mask]

        if len(candidate_indices) == 0:
            continue

        # Top-bv by risk score (highest risk removed first)
        k = min(bv, len(candidate_indices))
        _, top_idx = torch.topk(candidate_scores, k)
        to_remove = candidate_indices[top_idx]

        # Check minimum degree protection
        current_degree = len(edge_indices)
        removed_count = 0
        for idx in to_remove:
            if current_degree - removed_count <= min_degree:
                break
            u = src[idx].item()
            v_node = dst[idx].item()

            # Protect self-loops
            if protect_self_loops and u == v_node:
                continue

            # Check degree of neighbor too
            neighbor_degree = degree[u].item()
            if neighbor_degree <= min_degree:
                continue

            prune_mask[idx] = True
            removed_count += 1
            degree[u] -= 1

    # Bridge protection: detect and protect bridge edges
    if protect_bridges:
        try:
            from src.graca.bridge_protection import protect_bridges_in_pruning
            prune_mask = protect_bridges_in_pruning(edge_index, prune_mask, num_nodes, protect_bridges=True)
        except ImportError:
            pass  # Bridge protection module not available

    # Build pruned edge index
    keep_mask = ~prune_mask
    pruned_edge_index = edge_index[:, keep_mask]

    # Compute graph stats
    num_edges_before = E
    num_edges_after = keep_mask.sum().item()
    isolated_nodes = (degree == 0).sum().item()
    min_deg = degree.min().item()
    mean_deg = degree.mean().item()

    # Compute largest connected component ratio (approximate)
    # Use simple BFS on pruned graph
    lcc_ratio = compute_lcc_ratio(pruned_edge_index, num_nodes)

    graph_stats = {
        "num_edges_before": num_edges_before,
        "num_edges_after": num_edges_after,
        "prune_ratio": 1.0 - num_edges_after / max(num_edges_before, 1),
        "isolated_nodes": int(isolated_nodes),
        "min_degree": float(min_deg),
        "mean_degree": float(mean_deg),
        "largest_connected_component_ratio": lcc_ratio,
    }

    return pruned_edge_index, prune_mask, graph_stats


def compute_lcc_ratio(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Compute fraction of nodes in the largest connected component."""
    if edge_index.shape[1] == 0:
        return 0.0

    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * num_nodes
    max_cc = 0

    for start in range(num_nodes):
        if visited[start]:
            continue
        # BFS
        queue = [start]
        visited[start] = True
        cc_size = 0
        while queue:
            node = queue.pop(0)
            cc_size += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        max_cc = max(max_cc, cc_size)

    return max_cc / num_nodes
