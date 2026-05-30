import torch
from collections import defaultdict


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

    For undirected graphs, edges (u,v) and (v,u) are deleted together.

    Args:
        edge_index: [2, E]
        risk_score: [E] P_vu risk scores
        num_nodes: number of nodes
        beta: max pruning ratio per node
        min_degree: minimum degree to preserve
        lambda_theta: threshold = mean + lambda * std
        undirected: whether graph is undirected
        protect_self_loops: if True, never remove self-loops
        protect_bridges: if True, protect bridge edges (default off)

    Returns:
        pruned_edge_index: [2, E']
        prune_mask: [E] True = removed
        graph_stats: dict
    """
    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]
    device = edge_index.device

    prune_mask = torch.zeros(E, dtype=torch.bool, device=device)

    # For undirected graphs, group edges by undirected pair
    # and average risk scores across both directions
    if undirected:
        # Build undirected edge mapping
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        # Average risk score per undirected edge
        undirected_risk = {}
        for key, indices in edge_key_to_indices.items():
            undirected_risk[key] = risk_score[indices].mean().item()

        # Build undirected adjacency for degree counting
        undirected_adj = defaultdict(set)
        for (u, v) in edge_key_to_indices.keys():
            if u != v:  # exclude self-loops from degree
                undirected_adj[u].add(v)
                undirected_adj[v].add(u)

        # Per-node pruning on undirected edges
        for v in range(num_nodes):
            neighbors = list(undirected_adj.get(v, set()))
            if len(neighbors) <= min_degree:
                continue

            # Get undirected edge keys and risk scores for this node
            v_edge_keys = []
            v_risk_scores = []
            for u in neighbors:
                key = (min(v, u), max(v, u))
                v_edge_keys.append(key)
                v_risk_scores.append(undirected_risk[key])

            v_risk_t = torch.tensor(v_risk_scores, device=device)

            # Budget
            bv = min(
                int(beta * len(neighbors)),
                len(neighbors) - min_degree,
            )
            if bv <= 0:
                continue

            # Local threshold
            threshold = v_risk_t.mean() + lambda_theta * v_risk_t.std()

            # Candidates: risk > threshold
            candidate_mask = v_risk_t > threshold
            candidate_keys = [k for k, m in zip(v_edge_keys, candidate_mask) if m]
            candidate_scores = v_risk_t[candidate_mask]

            if len(candidate_keys) == 0:
                continue

            # Top-bv by risk score
            k = min(bv, len(candidate_keys))
            _, top_idx = torch.topk(candidate_scores, k)
            keys_to_remove = [candidate_keys[i] for i in top_idx.tolist()]

            # Apply removal (mark both directions)
            for key in keys_to_remove:
                u = key[0] if key[1] == v else key[1]

                # Protect self-loops
                if protect_self_loops and u == v:
                    continue

                # Check min_degree for both endpoints
                v_deg = len(undirected_adj.get(v, set()))
                u_deg = len(undirected_adj.get(u, set()))
                if v_deg <= min_degree or u_deg <= min_degree:
                    continue

                # Mark all directed edges for this undirected pair
                for idx in edge_key_to_indices[key]:
                    prune_mask[idx] = True

                # Update adjacency
                undirected_adj[v].discard(u)
                undirected_adj[u].discard(v)
    else:
        # Directed graph pruning (original logic)
        edges_by_dst = defaultdict(list)
        for i in range(E):
            edges_by_dst[dst[i].item()].append(i)

        # Compute degree
        degree = torch.zeros(num_nodes, device=device)
        for i in range(E):
            degree[dst[i]] += 1

        for v, edge_indices in edges_by_dst.items():
            if len(edge_indices) <= min_degree:
                continue

            edge_indices_t = torch.tensor(edge_indices, device=device)
            scores_v = risk_score[edge_indices_t]

            bv = min(int(beta * len(edge_indices)), len(edge_indices) - min_degree)
            if bv <= 0:
                continue

            threshold = scores_v.mean() + lambda_theta * scores_v.std()
            candidate_mask = scores_v > threshold
            candidate_indices = edge_indices_t[candidate_mask]
            candidate_scores = scores_v[candidate_mask]

            if len(candidate_indices) == 0:
                continue

            k = min(bv, len(candidate_indices))
            _, top_idx = torch.topk(candidate_scores, k)
            to_remove = candidate_indices[top_idx]

            removed_count = 0
            for idx in to_remove:
                if len(edge_indices) - removed_count <= min_degree:
                    break
                u = src[idx].item()
                if protect_self_loops and u == v:
                    continue
                if degree[u].item() <= min_degree:
                    continue
                prune_mask[idx] = True
                removed_count += 1
                degree[u] -= 1

    # Bridge protection
    if protect_bridges:
        try:
            from src.graca.bridge_protection import protect_bridges_in_pruning
            prune_mask = protect_bridges_in_pruning(edge_index, prune_mask, num_nodes, True)
        except ImportError:
            pass

    # Build pruned edge index
    keep_mask = ~prune_mask
    pruned_edge_index = edge_index[:, keep_mask]

    # Compute graph stats from the FINAL pruned graph
    graph_stats = compute_graph_stats(pruned_edge_index, num_nodes, E)

    return pruned_edge_index, prune_mask, graph_stats


def compute_graph_stats(edge_index: torch.Tensor, num_nodes: int, num_edges_before: int) -> dict:
    """Compute graph statistics from the final pruned edge_index."""
    E_after = edge_index.shape[1]

    if E_after == 0:
        return {
            "num_edges_before": num_edges_before,
            "num_edges_after": 0,
            "prune_ratio": 1.0,
            "isolated_nodes": num_nodes,
            "min_degree": 0,
            "mean_degree": 0,
            "largest_connected_component_ratio": 0.0,
        }

    # Compute degree from final edge_index
    degree = torch.zeros(num_nodes)
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()

    for i in range(E_after):
        degree[dst[i]] += 1

    isolated = (degree == 0).sum().item()
    min_deg = degree.min().item()
    mean_deg = degree.mean().item()
    lcc_ratio = compute_lcc_ratio(edge_index, num_nodes)

    return {
        "num_edges_before": num_edges_before,
        "num_edges_after": E_after,
        "prune_ratio": 1.0 - E_after / max(num_edges_before, 1),
        "isolated_nodes": int(isolated),
        "min_degree": float(min_deg),
        "mean_degree": float(mean_deg),
        "largest_connected_component_ratio": lcc_ratio,
    }


def compute_lcc_ratio(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Compute fraction of nodes in the largest connected component."""
    if edge_index.shape[1] == 0:
        return 0.0

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
