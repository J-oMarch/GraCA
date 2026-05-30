"""
Tests for result schema validation and baseline pruning correctness.
"""
import sys
import os
import torch
import csv
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.result_writer import RESULT_FIELDS, write_result_row, read_results
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.graca.pruning import prune_graph, compute_graph_stats


def test_result_fields_schema():
    """Verify RESULT_FIELDS has all required fields."""
    required = [
        "run_id", "timestamp", "experiment_type", "seed", "dataset", "method",
        "oracle_only", "proxy_model", "downstream_model",
        "actual_prune_ratio", "num_edges_before", "num_edges_after",
        "isolated_nodes", "min_degree", "mean_degree",
        "largest_connected_component_ratio",
        "edge_homophily_before", "edge_homophily_after",
        "val_acc", "test_acc", "test_f1", "best_epoch", "runtime",
        "config_path", "graph_path",
        "noise_type", "noise_ratio", "num_injected_edges",
        "bad_edge_precision", "bad_edge_recall", "bad_edge_f1",
        "clean_edge_mistakenly_removed_ratio",
        "notes",
    ]
    for field in required:
        assert field in RESULT_FIELDS, f"Missing required field: {field}"
    print("PASS: test_result_fields_schema")


def test_write_read_roundtrip():
    """Verify write_result_row writes correct CSV that can be read back."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name

    try:
        row = {
            "run_id": "test_001", "seed": 0, "dataset": "Cora",
            "experiment_type": "clean", "method": "GraCA-lite",
            "oracle_only": False, "proxy_model": "GCN", "downstream_model": "GCN",
            "prune_ratio_target": 0.2, "actual_prune_ratio": 0.15,
            "num_edges_before": 1000, "num_edges_after": 850,
            "isolated_nodes": 0, "min_degree": 2, "mean_degree": 5.0,
            "largest_connected_component_ratio": 0.95,
            "edge_homophily_before": 0.8, "edge_homophily_after": 0.85,
            "val_acc": 0.82, "test_acc": 0.80, "test_f1": 0.79,
            "best_epoch": 100, "runtime": 10.5,
            "config_path": "test.yaml", "graph_path": "",
        }
        write_result_row(row, tmp_path)

        results = read_results(tmp_path)
        assert len(results) == 1, f"Expected 1 row, got {len(results)}"
        assert results[0]["run_id"] == "test_001"
        assert results[0]["dataset"] == "Cora"
        assert results[0]["method"] == "GraCA-lite"
        assert results[0]["actual_prune_ratio"] != ""  # should be written

        # Verify no old "prune_ratio" field
        with open(tmp_path, "r") as f:
            reader = csv.DictReader(f)
            assert "prune_ratio" not in reader.fieldnames, "Old field 'prune_ratio' found"
            assert "actual_prune_ratio" in reader.fieldnames, "Missing 'actual_prune_ratio'"

        print("PASS: test_write_read_roundtrip")
    finally:
        os.unlink(tmp_path)


def test_noise_injection_cross_class():
    """Test cross_class_oracle noise injection produces valid pairs."""
    num_nodes = 100
    # Create a simple graph: chain
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # Create labels: alternating classes
    y = torch.tensor([i % 3 for i in range(num_nodes)])

    result = inject_noise(
        edge_index=edge_index, num_nodes=num_nodes,
        noise_type="cross_class_oracle", noise_ratio=0.10,
        y=y, seed=42,
    )

    noisy_ei = result["noisy_edge_index"]
    bad_mask = result["bad_edge_mask"]

    # Check dimensions
    assert noisy_ei.shape[1] == edge_index.shape[1] + 2 * result["num_injected_edges"]
    assert bad_mask.shape[0] == noisy_ei.shape[1]
    assert bad_mask.sum().item() == 2 * result["num_injected_edges"]

    # Check that injected edges are cross-class
    E_orig = edge_index.shape[1]
    injected = noisy_ei[:, E_orig:]
    for i in range(0, injected.shape[1], 2):
        u, v = injected[0, i].item(), injected[1, i].item()
        assert y[u] != y[v], f"Injected edge ({u},{v}) is same-class"

    # Check undirected pairs
    for i in range(0, injected.shape[1], 2):
        u0, v0 = injected[0, i].item(), injected[1, i].item()
        u1, v1 = injected[0, i+1].item(), injected[1, i+1].item()
        assert u0 == v1 and v0 == u1, f"Not undirected pair at {i}"

    print("PASS: test_noise_injection_cross_class")


def test_noise_injection_low_feature():
    """Test low_feature_similarity noise injection."""
    num_nodes = 50
    num_features = 10
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # Create features: first half similar, second half different
    x = torch.zeros(num_nodes, num_features)
    x[:25] = torch.randn(1, num_features).expand(25, -1)  # similar
    x[25:] = torch.randn(1, num_features).expand(25, -1)  # different

    result = inject_noise(
        edge_index=edge_index, num_nodes=num_nodes,
        noise_type="low_feature_similarity", noise_ratio=0.10,
        x=x, seed=42,
    )

    assert result["num_injected_edges"] > 0, "No edges injected"
    assert result["noisy_edge_index"].shape[1] > edge_index.shape[1]
    print("PASS: test_noise_injection_low_feature")


def test_bad_edge_detection_perfect():
    """Test that perfect pruning of bad edges gives F1=1."""
    E = 100
    # 50 clean, 50 bad
    prune_mask = torch.zeros(E, dtype=torch.bool)
    prune_mask[50:] = True  # prune all bad edges

    bad_edge_mask = torch.zeros(E, dtype=torch.bool)
    bad_edge_mask[50:] = True  # all bad

    edge_index = torch.tensor([list(range(E)), list(range(1, E+1))], dtype=torch.long)
    edge_index[1, -1] = 0  # close the loop

    det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, edge_index)

    assert abs(det["bad_edge_precision"] - 1.0) < 0.01, f"Precision: {det['bad_edge_precision']}"
    assert abs(det["bad_edge_recall"] - 1.0) < 0.01, f"Recall: {det['bad_edge_recall']}"
    assert abs(det["bad_edge_f1"] - 1.0) < 0.01, f"F1: {det['bad_edge_f1']}"
    assert abs(det["clean_edge_mistakenly_removed_ratio"]) < 0.01

    print("PASS: test_bad_edge_detection_perfect")


def test_bad_edge_detection_random():
    """Test that random pruning on bad edges gives low F1."""
    E = 1000
    # 100 bad, 900 clean
    bad_edge_mask = torch.zeros(E, dtype=torch.bool)
    bad_edge_mask[:100] = True

    # Random prune ~10%
    prune_mask = torch.zeros(E, dtype=torch.bool)
    prune_mask[torch.randperm(E)[:100]] = True

    edge_index = torch.tensor([list(range(E)), list(range(1, E+1))], dtype=torch.long)
    edge_index[1, -1] = 0

    det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, edge_index)

    # Random should give ~0.1 F1 (low)
    assert det["bad_edge_f1"] < 0.3, f"Random F1 too high: {det['bad_edge_f1']}"
    print(f"PASS: test_bad_edge_detection_random (F1={det['bad_edge_f1']:.4f})")


def test_undirected_pruning_keeps_symmetry():
    """Test that pruning an undirected graph maintains symmetry."""
    # Create undirected graph with both directions
    edges = [(0,1), (1,0), (1,2), (2,1), (2,3), (3,2), (0,3), (3,0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Risk score: higher for edges involving node 2
    risk_score = torch.tensor([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2])

    pruned_ei, prune_mask, stats = prune_graph(
        edge_index=edge_index, risk_score=risk_score,
        num_nodes=4, beta=0.5, min_degree=1, undirected=True,
    )

    # Check symmetry of pruned graph
    src = pruned_ei[0].tolist()
    dst = pruned_ei[1].tolist()
    edges_set = set(zip(src, dst))
    for u, v in edges_set:
        assert (v, u) in edges_set, f"Asymmetric: ({u},{v}) present but ({v},{u}) missing"

    print("PASS: test_undirected_pruning_keeps_symmetry")


def test_compute_graph_stats_real():
    """Test that compute_graph_stats returns real values, not zeros."""
    edge_index = torch.tensor([[0,1,1,2,2,0], [1,0,2,1,0,2]], dtype=torch.long)
    stats = compute_graph_stats(edge_index, num_nodes=3, num_edges_before=8)

    assert stats["min_degree"] > 0, f"min_degree={stats['min_degree']}"
    assert stats["mean_degree"] > 0, f"mean_degree={stats['mean_degree']}"
    assert stats["isolated_nodes"] == 0
    assert stats["largest_connected_component_ratio"] == 1.0
    print(f"PASS: test_compute_graph_stats_real (min_deg={stats['min_degree']}, mean_deg={stats['mean_degree']:.2f})")


if __name__ == "__main__":
    test_result_fields_schema()
    test_write_read_roundtrip()
    test_noise_injection_cross_class()
    test_noise_injection_low_feature()
    test_bad_edge_detection_perfect()
    test_bad_edge_detection_random()
    test_undirected_pruning_keeps_symmetry()
    test_compute_graph_stats_real()
    print("\nAll tests passed!")
