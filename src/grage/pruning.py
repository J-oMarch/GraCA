"""
GraGE Pruning: Degree-preserving pruning using GraGE edge scores.

Larger score means more harmful edge (should be pruned first).
"""
import torch
from typing import Dict, Tuple
from src.graca.pruning import prune_graph


def prune_by_grage_score(
    edge_index: torch.Tensor,
    score: torch.Tensor,
    num_nodes: int,
    prune_ratio: float = 0.20,
    min_degree: int = 1,
    undirected: bool = True,
    protect_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Prune graph using GraGE edge scores.

    Args:
        edge_index: [2, E] edge index
        score: [E] GraGE scores (higher = more harmful, should be pruned)
        num_nodes: number of nodes
        prune_ratio: fraction of edges to remove
        min_degree: minimum degree to preserve
        undirected: if True, prune undirected pairs together
        protect_self_loops: if True, never remove self-loops

    Returns:
        pruned_edge_index: [2, E'] pruned edge index
        prune_mask: [E] boolean mask, True for pruned edges
        stats: dict with pruning statistics
    """
    pruned_ei, prune_mask, stats = prune_graph(
        edge_index=edge_index,
        risk_score=score,
        num_nodes=num_nodes,
        beta=0.2,  # Not used when target_prune_ratio is set
        min_degree=min_degree,
        undirected=undirected,
        protect_self_loops=protect_self_loops,
        target_prune_ratio=prune_ratio,
    )

    return pruned_ei, prune_mask, stats
