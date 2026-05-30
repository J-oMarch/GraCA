"""Unit tests for scoring determinism and data leakage."""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.models.model_factory import build_model
from src.graca.gradient_collector import collect_hidden_gradients
from src.graca.edge_scoring import compute_D, compute_M, compute_edge_scores
from src.training.losses import compute_scoring_loss


def test_deterministic_scoring():
    """With deterministic=True, same model+input should give same gradients."""
    set_seed(42)

    N, F, C = 100, 32, 5
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, 200))
    y = torch.randint(0, C, (N,))
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[:20] = True
    unlabeled_mask = ~train_mask

    model = build_model("GCN", F, 16, C, num_layers=2, dropout=0.5)
    teacher_probs = torch.softmax(torch.randn(N, C), dim=-1)
    rho_train = torch.ones(N)
    rho_train[~train_mask] = 0.5

    # Run twice with deterministic=True
    result1 = collect_hidden_gradients(
        model, x, edge_index, y, teacher_probs, rho_train,
        train_mask, unlabeled_mask, lambda_s=1.0,
        collect_layer="all", deterministic=True,
    )

    # Reload same model state
    set_seed(42)
    model2 = build_model("GCN", F, 16, C, num_layers=2, dropout=0.5)
    model2.load_state_dict(model.state_dict())

    result2 = collect_hidden_gradients(
        model2, x, edge_index, y, teacher_probs, rho_train,
        train_mask, unlabeled_mask, lambda_s=1.0,
        collect_layer="all", deterministic=True,
    )

    # Gradients should be identical
    assert torch.allclose(result1["grad"], result2["grad"], atol=1e-6), \
        "Deterministic scoring produced different gradients"
    print("✓ test_deterministic_scoring passed")


def test_no_test_label_leakage():
    """Practical mode must not use test labels in any loss."""
    from src.data.leakage_check import ensure_no_test_label_leakage

    config = {"experiment": {"oracle_only": False, "method": "graca_lite"}}
    train_mask = torch.zeros(100, dtype=torch.bool)
    train_mask[:20] = True
    test_mask = torch.zeros(100, dtype=torch.bool)
    test_mask[80:] = True

    # This should pass (loss_mask = train_mask)
    ensure_no_test_label_leakage(config, train_mask, test_mask, train_mask, "practical")

    # This should fail (loss_mask overlaps with test_mask)
    bad_loss_mask = torch.zeros(100, dtype=torch.bool)
    bad_loss_mask[85:] = True  # overlaps with test
    try:
        ensure_no_test_label_leakage(config, train_mask, test_mask, bad_loss_mask, "practical")
        assert False, "Should have raised assertion"
    except AssertionError:
        pass  # Expected

    print("✓ test_no_test_label_leakage passed")


def test_signed_cosine():
    """D_vu should be signed (not absolute value)."""
    grad = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1], [2, 3]]).t()

    D = compute_D(grad, edge_index)

    # D should be in [-1, 1]
    assert D.min() >= -1.0 - 1e-6 and D.max() <= 1.0 + 1e-6

    # Create opposing gradients
    grad2 = torch.zeros(4, 8)
    grad2[0] = torch.ones(8)
    grad2[2] = -torch.ones(8)  # opposite direction
    edge_index2 = torch.tensor([[0, 2], [2, 0]])  # shape [2, 2]

    D2 = compute_D(grad2, edge_index2)
    assert D2[0] < 0, f"Opposing gradients should give negative D, got {D2[0]}"
    print("✓ test_signed_cosine passed")


if __name__ == "__main__":
    test_deterministic_scoring()
    test_no_test_label_leakage()
    test_signed_cosine()
    print("\n✓ All scoring tests passed!")
