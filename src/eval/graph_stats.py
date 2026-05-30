import torch
from src.graca.pruning import compute_lcc_ratio


def compute_graph_statistics(
    edge_index_before: torch.Tensor,
    edge_index_after: torch.Tensor,
    num_nodes: int,
) -> dict:
    """Compute before/after graph statistics."""
    E_before = edge_index_before.shape[1]
    E_after = edge_index_after.shape[1]

    # Degree distribution of pruned graph
    # Count both endpoints for undirected graphs stored with both directions
    degree = torch.zeros(num_nodes)
    if E_after > 0:
        for i in range(E_after):
            degree[edge_index_after[0, i]] += 1
            degree[edge_index_after[1, i]] += 1

    isolated = (degree == 0).sum().item()
    lcc_ratio = compute_lcc_ratio(edge_index_after, num_nodes)

    return {
        "num_edges_before": E_before,
        "num_edges_after": E_after,
        "prune_ratio": 1.0 - E_after / max(E_before, 1),
        "isolated_nodes": isolated,
        "min_degree": degree.min().item() if E_after > 0 else 0,
        "mean_degree": degree.mean().item() if E_after > 0 else 0,
        "largest_connected_component_ratio": lcc_ratio,
    }
