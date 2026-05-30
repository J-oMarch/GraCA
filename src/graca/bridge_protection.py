"""
Bridge edge protection for graph pruning.
Detects bridge edges (edges whose removal disconnects the graph) and prevents their removal.
"""
import torch
import networkx as nx
from torch_geometric.utils import to_networkx


def detect_bridge_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Detect bridge edges using NetworkX.

    Args:
        edge_index: [2, E] edge indices
        num_nodes: number of nodes

    Returns:
        bridge_mask: [E] boolean tensor, True if edge is a bridge
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    edges = edge_index.t().cpu().numpy()
    edge_list = [(int(u), int(v)) for u, v in edges if u != v]
    G.add_edges_from(edge_list)

    # Find bridges
    bridges = set(nx.bridges(G))

    # Map back to edge_index
    bridge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        if (u, v) in bridges or (v, u) in bridges:
            bridge_mask[i] = True

    return bridge_mask


def protect_bridges_in_pruning(
    edge_index: torch.Tensor,
    prune_mask: torch.Tensor,
    num_nodes: int,
    protect_bridges: bool = True,
) -> torch.Tensor:
    """Remove bridge edges from prune_mask to prevent their removal.

    Args:
        edge_index: [2, E]
        prune_mask: [E] True = will be removed
        num_nodes: number of nodes
        protect_bridges: whether to enable bridge protection

    Returns:
        prune_mask: updated prune_mask with bridges protected
    """
    if not protect_bridges:
        return prune_mask

    bridge_mask = detect_bridge_edges(edge_index, num_nodes)

    # Move bridge_mask to same device as prune_mask
    bridge_mask = bridge_mask.to(prune_mask.device)

    # Remove bridges from prune set
    prune_mask = prune_mask & ~bridge_mask

    return prune_mask
