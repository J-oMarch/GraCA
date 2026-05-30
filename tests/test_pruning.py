"""Unit tests for pruning module."""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graca.pruning import prune_graph, compute_graph_stats


def test_undirected_symmetry():
    """After pruning an undirected graph, edges should remain symmetric."""
    # Create a simple undirected graph with both directions
    # 0-1, 1-2, 2-3, 3-0 (each as both directions)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0, 1, 0, 2, 1, 3, 2, 0, 3],
        [1, 0, 2, 1, 3, 2, 0, 3, 0, 1, 1, 2, 2, 3, 3, 0],
    ])

    # Give high risk to edge (0,1) to trigger pruning
    risk_score = torch.zeros(16)
    risk_score[0] = 10.0  # 0->1 high risk
    risk_score[8] = 10.0  # 1->0 high risk

    pruned, mask, stats = prune_graph(
        edge_index=edge_index, risk_score=risk_score,
        num_nodes=4, beta=0.5, min_degree=1,
        undirected=True, protect_self_loops=True,
    )

    # Check symmetry: if u->v exists, v->u must also exist
    src = pruned[0].tolist()
    dst = pruned[1].tolist()
    edges = set(zip(src, dst))
    for u, v in edges:
        if u != v:  # skip self-loops
            assert (v, u) in edges, f"Edge ({u},{v}) exists but ({v},{u}) missing"
    print("✓ test_undirected_symmetry passed")


def test_min_degree():
    """Nodes should not be pruned below min_degree."""
    # Complete graph on 4 nodes: each node has degree 3
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
    ])
    risk_score = torch.ones(12) * 10.0  # all high risk

    pruned, mask, stats = prune_graph(
        edge_index=edge_index, risk_score=risk_score,
        num_nodes=4, beta=0.9, min_degree=1,
        undirected=True, protect_self_loops=True,
    )

    # Check no node has degree < min_degree in pruned graph
    src = pruned[0].tolist()
    dst = pruned[1].tolist()
    degree = [0] * 4
    for u, v in zip(src, dst):
        if u != v:
            degree[v] += 1
    for i in range(4):
        assert degree[i] >= 1, f"Node {i} has degree {degree[i]} < min_degree=1"
    print("✓ test_min_degree passed")


def test_self_loop_protection():
    """Self-loops should not be removed when protect_self_loops=True."""
    edge_index = torch.tensor([
        [0, 1, 2, 0, 1, 2],
        [1, 2, 0, 0, 1, 2],  # includes self-loops 0->0, 1->1, 2->2
    ])
    risk_score = torch.ones(6) * 10.0  # all high risk

    pruned, mask, stats = prune_graph(
        edge_index=edge_index, risk_score=risk_score,
        num_nodes=3, beta=0.9, min_degree=0,
        undirected=False, protect_self_loops=True,
    )

    # Self-loops should remain
    src = pruned[0].tolist()
    dst = pruned[1].tolist()
    assert (0, 0) in list(zip(src, dst)), "Self-loop 0->0 removed"
    assert (1, 1) in list(zip(src, dst)), "Self-loop 1->1 removed"
    assert (2, 2) in list(zip(src, dst)), "Self-loop 2->2 removed"
    print("✓ test_self_loop_protection passed")


def test_graph_stats_from_final():
    """Graph stats should be computed from the final pruned edge_index."""
    edge_index = torch.tensor([
        [0, 1, 2, 1, 2, 0],
        [1, 2, 0, 0, 1, 2],
    ])
    stats = compute_graph_stats(edge_index, num_nodes=3, num_edges_before=10)

    assert stats["num_edges_after"] == 6
    assert stats["num_edges_before"] == 10
    assert abs(stats["prune_ratio"] - 0.4) < 1e-6
    print("✓ test_graph_stats_from_final passed")


if __name__ == "__main__":
    test_undirected_symmetry()
    test_min_degree()
    test_self_loop_protection()
    test_graph_stats_from_final()
    print("\n✓ All pruning tests passed!")
