"""
Tests for differentiable edge gate in GCN.

Verifies:
1. edge_gate=None produces identical output to old version
2. edge_gate=ones gives same output as no edge_gate
3. edge_gate requires_grad=True allows gradient computation
4. edge_gate=0 changes output relative to edge_gate=1
5. Old training flow unaffected when edge_gate not passed
"""
import torch
import pytest
from src.models.gcn import GCN


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    torch.manual_seed(42)
    num_nodes = 10
    num_features = 8
    num_classes = 3

    x = torch.randn(num_nodes, num_features)
    # Simple graph: chain + some edges
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]
    ], dtype=torch.long)

    return x, edge_index, num_features, num_classes, num_nodes


@pytest.fixture
def gcn_model(simple_graph):
    """Create a GCN model for testing."""
    _, _, num_features, num_classes, _ = simple_graph
    model = GCN(
        in_dim=num_features,
        hidden_dim=16,
        out_dim=num_classes,
        num_layers=2,
        dropout=0.0,  # Disable dropout for deterministic tests
    )
    model.eval()  # Disable dropout and BN randomness
    return model


def test_edge_gate_none_identical(simple_graph, gcn_model):
    """Test 1: edge_gate=None produces identical output to old version."""
    x, edge_index, _, _, _ = simple_graph

    # Old way (no edge_gate parameter)
    with torch.no_grad():
        out_old = gcn_model(x, edge_index)

    # New way with edge_gate=None
    with torch.no_grad():
        out_new = gcn_model(x, edge_index, edge_gate=None)

    assert torch.allclose(out_old, out_new, atol=1e-6), \
        f"edge_gate=None should produce identical output. Max diff: {(out_old - out_new).abs().max():.8f}"


def test_edge_gate_ones_identical(simple_graph, gcn_model):
    """Test 2: edge_gate=ones gives same output as no edge_gate."""
    x, edge_index, _, _, num_nodes = simple_graph
    E = edge_index.shape[1]

    with torch.no_grad():
        out_no_gate = gcn_model(x, edge_index)
        out_ones_gate = gcn_model(x, edge_index, edge_gate=torch.ones(E))

    assert torch.allclose(out_no_gate, out_ones_gate, atol=1e-6), \
        f"edge_gate=ones should match no gate. Max diff: {(out_no_gate - out_ones_gate).abs().max():.8f}"


def test_edge_gate_gradient(simple_graph, gcn_model):
    """Test 3: edge_gate requires_grad=True allows gradient computation."""
    x, edge_index, _, _, _ = simple_graph
    E = edge_index.shape[1]

    edge_gate = torch.ones(E, requires_grad=True)
    y = torch.randint(0, 3, (x.shape[0],))

    logits = gcn_model(x, edge_index, edge_gate=edge_gate)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    assert edge_gate.grad is not None, "edge_gate.grad should not be None after backward"
    assert edge_gate.grad.shape == (E,), \
        f"edge_gate.grad shape should be ({E},), got {edge_gate.grad.shape}"
    assert not torch.all(edge_gate.grad == 0), "edge_gate.grad should not be all zeros"


def test_edge_gate_zero_changes_output(simple_graph, gcn_model):
    """Test 4: edge_gate=0 changes output relative to edge_gate=1."""
    x, edge_index, _, _, _ = simple_graph
    E = edge_index.shape[1]

    with torch.no_grad():
        out_ones = gcn_model(x, edge_index, edge_gate=torch.ones(E))
        out_zeros = gcn_model(x, edge_index, edge_gate=torch.zeros(E))

    # Outputs should be different (not identical)
    assert not torch.allclose(out_ones, out_zeros, atol=1e-6), \
        "edge_gate=0 should produce different output from edge_gate=1"


def test_backward_flow_unaffected(simple_graph, gcn_model):
    """Test 5: Old training flow unaffected when edge_gate not passed."""
    x, edge_index, _, _, _ = simple_graph
    y = torch.randint(0, 3, (x.shape[0],))

    # Standard training flow (no edge_gate)
    gcn_model.train()
    logits = gcn_model(x, edge_index)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    # Check gradients exist on model parameters
    for name, param in gcn_model.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have grad"
        assert not torch.all(param.grad == 0), f"Parameter {name} grad should not be all zeros"


def test_edge_gate_autograd_grad(simple_graph, gcn_model):
    """Test 6: autograd.grad works with edge_gate."""
    x, edge_index, _, _, _ = simple_graph
    E = edge_index.shape[1]

    edge_gate = torch.ones(E, requires_grad=True)
    y = torch.randint(0, 3, (x.shape[0],))

    logits = gcn_model(x, edge_index, edge_gate=edge_gate)
    loss = torch.nn.functional.cross_entropy(logits, y)

    # Use autograd.grad instead of loss.backward()
    grads = torch.autograd.grad(loss, edge_gate, create_graph=False)[0]

    assert grads.shape == (E,), f"grads shape should be ({E},), got {grads.shape}"
    assert not torch.all(grads == 0), "grads should not be all zeros"


def test_edge_gate_partial_mask(simple_graph, gcn_model):
    """Test 7: edge_gate with partial masking (some edges gated, some not)."""
    x, edge_index, _, _, _ = simple_graph
    E = edge_index.shape[1]

    # Gate only first half of edges
    edge_gate = torch.ones(E)
    edge_gate[:E // 2] = 0.0

    with torch.no_grad():
        out_partial = gcn_model(x, edge_index, edge_gate=edge_gate)
        out_all_ones = gcn_model(x, edge_index, edge_gate=torch.ones(E))

    # Should be different since some edges are gated
    assert not torch.allclose(out_partial, out_all_ones, atol=1e-6), \
        "Partial gating should change output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
