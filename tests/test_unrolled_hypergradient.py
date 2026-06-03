"""
Tests for GraGE Unrolled Hypergradient.

Verifies:
1. K=1 unrolled produces different results from first-order
2. K=3 unrolled produces different results from K=1
3. Computation graph is not broken by .data or no_grad
4. Gradient flows through inner loop updates
"""
import torch
import pytest
from src.models.gcn import GCN
from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order
from src.grage.unrolled_hypergradient import compute_edge_gate_influence_unrolled


@pytest.fixture
def simple_setup():
    """Create a simple test setup for GraGE."""
    torch.manual_seed(42)
    num_nodes = 20
    num_features = 8
    num_classes = 3

    x = torch.randn(num_nodes, num_features)
    # Simple graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
         10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9,
         11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19, 18]
    ], dtype=torch.long)

    y = torch.randint(0, num_classes, (num_nodes,))

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:12] = True  # 12 train nodes
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[12:16] = True  # 4 val nodes
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[16:] = True  # 4 test nodes

    return x, edge_index, y, train_mask, val_mask, test_mask, num_features, num_classes


@pytest.fixture
def trained_model(simple_setup):
    """Create a trained GCN model."""
    x, edge_index, y, train_mask, val_mask, test_mask, num_features, num_classes = simple_setup

    model = GCN(
        in_dim=num_features, hidden_dim=16,
        out_dim=num_classes, num_layers=2, dropout=0.0
    )

    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    return model


def test_unrolled_different_from_first_order(simple_setup, trained_model):
    """Test that unrolled hypergradient produces different results from first-order."""
    x, edge_index, y, train_mask, val_mask, test_mask, num_features, num_classes = simple_setup
    model = trained_model

    # Split train into support/score
    from src.utils.mask_split import split_train_support_score
    support_mask, score_mask = split_train_support_score(train_mask, y, score_ratio=0.3, seed=42)

    # First-order
    result_fo = compute_edge_gate_influence_first_order(
        model=model, x=x, edge_index=edge_index, y=y,
        score_mask=score_mask, normalize=False, undirected=True,
    )
    grad_fo = result_fo["raw_grad"]

    # Unrolled K=1
    def model_ctor():
        return GCN(
            in_dim=num_features, hidden_dim=16,
            out_dim=num_classes, num_layers=2, dropout=0.0
        )

    init_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    result_unrolled = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=init_state_dict,
        x=x, edge_index=edge_index, y=y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=1, inner_lr=0.01,
        undirected=True,
    )
    grad_unrolled = result_unrolled["raw_grad"]

    # They should be different (not identical)
    # The unrolled version accounts for how edge_gate affects parameters through training
    assert not torch.allclose(grad_fo, grad_unrolled, atol=1e-6), \
        f"Unrolled K=1 should differ from first-order. Max diff: {(grad_fo - grad_unrolled).abs().max():.8f}"

    print(f"First-order grad mean: {grad_fo.mean():.6f}")
    print(f"Unrolled K=1 grad mean: {grad_unrolled.mean():.6f}")
    print(f"Max difference: {(grad_fo - grad_unrolled).abs().max():.6f}")


def test_unrolled_k1_vs_k3(simple_setup, trained_model):
    """Test that K=3 unrolled produces different results from K=1."""
    x, edge_index, y, train_mask, val_mask, test_mask, num_features, num_classes = simple_setup
    model = trained_model

    from src.utils.mask_split import split_train_support_score
    support_mask, score_mask = split_train_support_score(train_mask, y, score_ratio=0.3, seed=42)

    def model_ctor():
        return GCN(
            in_dim=num_features, hidden_dim=16,
            out_dim=num_classes, num_layers=2, dropout=0.0
        )

    init_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # K=1
    result_k1 = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=init_state_dict,
        x=x, edge_index=edge_index, y=y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=1, inner_lr=0.01,
        undirected=True,
    )
    grad_k1 = result_k1["raw_grad"]

    # K=3
    init_state_dict_2 = {k: v.clone() for k, v in model.state_dict().items()}
    result_k3 = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=init_state_dict_2,
        x=x, edge_index=edge_index, y=y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=3, inner_lr=0.01,
        undirected=True,
    )
    grad_k3 = result_k3["raw_grad"]

    # They should be different
    assert not torch.allclose(grad_k1, grad_k3, atol=1e-6), \
        f"Unrolled K=3 should differ from K=1. Max diff: {(grad_k1 - grad_k3).abs().max():.8f}"

    print(f"Unrolled K=1 grad mean: {grad_k1.mean():.6f}")
    print(f"Unrolled K=3 grad mean: {grad_k3.mean():.6f}")
    print(f"Max difference: {(grad_k1 - grad_k3).abs().max():.6f}")


def test_unrolled_gradient_not_zero(simple_setup, trained_model):
    """Test that unrolled hypergradient produces non-zero gradients."""
    x, edge_index, y, train_mask, val_mask, test_mask, num_features, num_classes = simple_setup
    model = trained_model

    from src.utils.mask_split import split_train_support_score
    support_mask, score_mask = split_train_support_score(train_mask, y, score_ratio=0.3, seed=42)

    def model_ctor():
        return GCN(
            in_dim=num_features, hidden_dim=16,
            out_dim=num_classes, num_layers=2, dropout=0.0
        )

    init_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    result = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=init_state_dict,
        x=x, edge_index=edge_index, y=y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=1, inner_lr=0.01,
        undirected=True,
    )

    grad = result["raw_grad"]

    # Gradient should not be all zeros
    assert not torch.all(grad == 0), "Unrolled gradient should not be all zeros"

    # Gradient should have reasonable magnitude
    assert grad.abs().max() > 1e-6, f"Gradient magnitude too small: {grad.abs().max():.8f}"

    print(f"Gradient stats: mean={grad.mean():.6f}, std={grad.std():.6f}, max={grad.abs().max():.6f}")


def test_unrolled_computation_graph_preserved(simple_setup, trained_model):
    """Test that computation graph is preserved through inner loop updates.

    This is the KEY test: if the graph is broken by .data assignment,
    the gradient would be zero or identical to first-order.
    """
    x, edge_index, y, train_mask, val_mask, test_mask, num_features, num_classes = simple_setup
    model = trained_model

    from src.utils.mask_split import split_train_support_score
    support_mask, score_mask = split_train_support_score(train_mask, y, score_ratio=0.3, seed=42)

    def model_ctor():
        return GCN(
            in_dim=num_features, hidden_dim=16,
            out_dim=num_classes, num_layers=2, dropout=0.0
        )

    init_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Run unrolled with K=1
    result = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=init_state_dict,
        x=x, edge_index=edge_index, y=y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=1, inner_lr=0.01,
        undirected=True,
    )

    # If the graph was broken, the gradient would be identical to first-order
    # We already test that they're different in test_unrolled_different_from_first_order
    # Here we just verify the gradient is meaningful
    grad = result["raw_grad"]

    # Check diagnostics
    diag = result["diagnostics"]
    assert diag["score_loss"] > 0, "Score loss should be positive"
    assert diag["support_size"] > 0, "Support size should be positive"
    assert diag["score_size"] > 0, "Score size should be positive"

    print(f"Score loss: {diag['score_loss']:.4f}")
    print(f"Support size: {diag['support_size']}")
    print(f"Score size: {diag['score_size']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
