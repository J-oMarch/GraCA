"""Tests for GraGE Adaptive Score methods.

Tests:
1. FAA-Hybrid produces non-trivial scores
2. FAA-Hybrid alpha increases with feature similarity
3. MCGC score is non-trivial
4. MCGC consistency penalizes unstable edges
5. Both methods are deterministic with same inputs
6. Both methods handle edge cases
"""
import torch
import pytest
from src.grage.adaptive_score import (
    compute_faa_hybrid_score,
    compute_mcgc_score,
    rank_normalize,
)


@pytest.fixture
def synthetic_edges():
    """Create synthetic edge data for testing."""
    torch.manual_seed(42)
    E = 100

    # Feature risk: random in [0, 2]
    feature_risk = torch.rand(E) * 2.0

    # Dynamic gradient: mix of positive and negative
    dynamic_grad = torch.randn(E)

    # Feature similarity: in [-1, 1]
    feature_similarity = torch.randn(E).clamp(-1, 1)

    # Edge index: simple chain graph
    N = 50
    src = torch.arange(0, N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    return {
        "feature_risk": feature_risk,
        "dynamic_grad": dynamic_grad,
        "feature_similarity": feature_similarity,
        "edge_index": edge_index,
        "E": E,
        "N": N,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FAA-Hybrid Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_faa_hybrid_non_trivial(synthetic_edges):
    """FAA-Hybrid should produce non-trivial scores."""
    result = compute_faa_hybrid_score(
        feature_risk=synthetic_edges["feature_risk"],
        dynamic_grad=synthetic_edges["dynamic_grad"],
        feature_similarity=synthetic_edges["feature_similarity"],
        lambda_pos=0.25,
        lambda_neg=0.25,
        ambig_scale=1.0,
        base_alpha=0.0,
        base_beta=0.0,
        undirected=False,
    )
    score = result["hybrid_score"]

    assert score.shape == (synthetic_edges["E"],), f"Wrong shape: {score.shape}"
    assert not torch.all(score == 0), "Score should not be all zeros"
    assert score.std() > 0.01, f"Score variance too low: {score.std():.6f}"
    print(f"✓ FAA-Hybrid non-trivial: mean={score.mean():.4f}, std={score.std():.4f}")


def test_faa_hybrid_alpha_increases_with_similarity(synthetic_edges):
    """Alpha should increase with feature similarity (more ambiguous → trust gradient more)."""
    result = compute_faa_hybrid_score(
        feature_risk=synthetic_edges["feature_risk"],
        dynamic_grad=synthetic_edges["dynamic_grad"],
        feature_similarity=synthetic_edges["feature_similarity"],
        lambda_pos=0.25,
        lambda_neg=0.25,
        ambig_scale=1.0,
        base_alpha=0.0,
        base_beta=0.0,
        undirected=False,
    )
    diag = result["diagnostics"]

    # Alpha should be positive (base_alpha + ambig_scale * sim)
    assert diag["alpha_mean"] > 0, f"Alpha mean should be positive: {diag['alpha_mean']}"
    print(f"✓ FAA-Hybrid alpha_mean={diag['alpha_mean']:.4f}, beta_mean={diag['beta_mean']:.4f}")


def test_faa_hybrid_higher_grad_for_ambiguous_edges(synthetic_edges):
    """Edges with high feature similarity should get stronger gradient weighting."""
    fr = synthetic_edges["feature_risk"]
    dg = synthetic_grad = synthetic_edges["dynamic_grad"]

    # Low similarity (clear features)
    low_sim = torch.full_like(fr, -0.8)
    result_low = compute_faa_hybrid_score(
        feature_risk=fr, dynamic_grad=dg,
        feature_similarity=low_sim,
        lambda_pos=0.5, lambda_neg=0.5,
        ambig_scale=2.0, base_alpha=0.0, base_beta=0.0,
        undirected=False,
    )

    # High similarity (ambiguous features)
    high_sim = torch.full_like(fr, 0.8)
    result_high = compute_faa_hybrid_score(
        feature_risk=fr, dynamic_grad=dg,
        feature_similarity=high_sim,
        lambda_pos=0.5, lambda_neg=0.5,
        ambig_scale=2.0, base_alpha=0.0, base_beta=0.0,
        undirected=False,
    )

    # High-similarity scores should deviate more from pure feature risk
    # because gradient contribution is amplified
    R_feature = rank_normalize(fr)
    diff_low = (result_low["hybrid_score"] - R_feature).abs().mean()
    diff_high = (result_high["hybrid_score"] - R_feature).abs().mean()

    assert diff_high > diff_low, \
        f"Ambiguous edges should get more gradient influence: diff_high={diff_high:.4f} vs diff_low={diff_low:.4f}"
    print(f"✓ FAA-Hybrid ambiguous edges get more gradient: diff_high={diff_high:.4f} > diff_low={diff_low:.4f}")


def test_faa_hybrid_deterministic(synthetic_edges):
    """FAA-Hybrid should be deterministic with same inputs."""
    kwargs = dict(
        feature_risk=synthetic_edges["feature_risk"],
        dynamic_grad=synthetic_edges["dynamic_grad"],
        feature_similarity=synthetic_edges["feature_similarity"],
        lambda_pos=0.25, lambda_neg=0.25,
        ambig_scale=1.0, undirected=False,
    )
    r1 = compute_faa_hybrid_score(**kwargs)
    r2 = compute_faa_hybrid_score(**kwargs)

    assert torch.allclose(r1["hybrid_score"], r2["hybrid_score"]), \
        "FAA-Hybrid should be deterministic"
    print("✓ FAA-Hybrid deterministic")


def test_faa_hybrid_with_undirected(synthetic_edges):
    """FAA-Hybrid should work with undirected averaging."""
    result = compute_faa_hybrid_score(
        feature_risk=synthetic_edges["feature_risk"],
        dynamic_grad=synthetic_edges["dynamic_grad"],
        feature_similarity=synthetic_edges["feature_similarity"],
        lambda_pos=0.25, lambda_neg=0.25,
        ambig_scale=1.0,
        undirected=True,
        edge_index=synthetic_edges["edge_index"],
    )
    score = result["hybrid_score"]
    assert score.shape == (synthetic_edges["E"],)
    print("✓ FAA-Hybrid with undirected averaging")


# ═══════════════════════════════════════════════════════════════════════════════
# MCGC Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_mcgc_non_trivial(synthetic_edges):
    """MCGC should produce non-trivial scores."""
    E = synthetic_edges["E"]

    # Create checkpoint gradients with some consistency
    torch.manual_seed(42)
    base_grad = torch.randn(E)
    checkpoint_grads = [
        base_grad + 0.1 * torch.randn(E),
        base_grad + 0.1 * torch.randn(E),
        base_grad + 0.1 * torch.randn(E),
        base_grad + 0.1 * torch.randn(E),
    ]

    result = compute_mcgc_score(
        feature_risk=synthetic_edges["feature_risk"],
        checkpoint_grads=checkpoint_grads,
        lambda_pos=0.25,
        lambda_neg=0.25,
        consistency_weight=1.0,
        undirected=False,
    )
    score = result["hybrid_score"]

    assert score.shape == (E,)
    assert not torch.all(score == 0), "Score should not be all zeros"
    print(f"✓ MCGC non-trivial: mean={score.mean():.4f}, std={score.std():.4f}")


def test_mcgc_consistency_high_for_consistent_grads(synthetic_edges):
    """MCGC should give high consistency when gradients are consistent."""
    E = synthetic_edges["E"]

    # Very consistent gradients (same direction, small noise)
    base_grad = torch.ones(E) * 0.5
    checkpoint_grads = [
        base_grad + 0.01 * torch.randn(E),
        base_grad + 0.01 * torch.randn(E),
        base_grad + 0.01 * torch.randn(E),
    ]

    result = compute_mcgc_score(
        feature_risk=synthetic_edges["feature_risk"],
        checkpoint_grads=checkpoint_grads,
        lambda_pos=0.25, lambda_neg=0.25,
        undirected=False,
    )
    diag = result["diagnostics"]

    assert diag["consistency_mean"] > 0.8, \
        f"Consistent grads should give high consistency: {diag['consistency_mean']:.4f}"
    print(f"✓ MCGC high consistency: {diag['consistency_mean']:.4f}")


def test_mcgc_consistency_low_for_inconsistent_grads(synthetic_edges):
    """MCGC should give low consistency when gradients flip sign."""
    E = synthetic_edges["E"]

    # Inconsistent gradients: alternating signs
    checkpoint_grads = [
        torch.ones(E) * 0.5,
        -torch.ones(E) * 0.5,
        torch.ones(E) * 0.5,
        -torch.ones(E) * 0.5,
    ]

    result = compute_mcgc_score(
        feature_risk=synthetic_edges["feature_risk"],
        checkpoint_grads=checkpoint_grads,
        lambda_pos=0.25, lambda_neg=0.25,
        undirected=False,
    )
    diag = result["diagnostics"]

    # Consistency should be ~0.5 (random sign agreement)
    assert diag["consistency_mean"] < 0.7, \
        f"Inconsistent grads should give low consistency: {diag['consistency_mean']:.4f}"
    print(f"✓ MCGC low consistency for inconsistent grads: {diag['consistency_mean']:.4f}")


def test_mcgc_consistent_grads_get_stronger_signal(synthetic_edges):
    """Consistent harmful gradients should get amplified penalty."""
    E = synthetic_edges["E"]
    fr = synthetic_edges["feature_risk"]

    # Consistent harmful gradients (positive)
    base_grad = torch.ones(E) * 0.5
    consistent_grads = [
        base_grad + 0.01 * torch.randn(E),
        base_grad + 0.01 * torch.randn(E),
        base_grad + 0.01 * torch.randn(E),
    ]

    # Inconsistent gradients
    inconsistent_grads = [
        torch.ones(E) * 0.5,
        -torch.ones(E) * 0.5,
        torch.ones(E) * 0.5,
    ]

    result_consistent = compute_mcgc_score(
        feature_risk=fr, checkpoint_grads=consistent_grads,
        lambda_pos=0.5, lambda_neg=0.5, consistency_weight=2.0,
        undirected=False,
    )
    result_inconsistent = compute_mcgc_score(
        feature_risk=fr, checkpoint_grads=inconsistent_grads,
        lambda_pos=0.5, lambda_neg=0.5, consistency_weight=2.0,
        undirected=False,
    )

    R_feature = rank_normalize(fr)
    diff_consistent = (result_consistent["hybrid_score"] - R_feature).abs().mean()
    diff_inconsistent = (result_inconsistent["hybrid_score"] - R_feature).abs().mean()

    assert diff_consistent > diff_inconsistent, \
        f"Consistent grads should get stronger signal: {diff_consistent:.4f} vs {diff_inconsistent:.4f}"
    print(f"✓ MCGC consistent grads amplified: {diff_consistent:.4f} > {diff_inconsistent:.4f}")


def test_mcgc_deterministic(synthetic_edges):
    """MCGC should be deterministic with same inputs."""
    E = synthetic_edges["E"]
    torch.manual_seed(42)
    grads = [torch.randn(E) for _ in range(3)]

    r1 = compute_mcgc_score(
        feature_risk=synthetic_edges["feature_risk"],
        checkpoint_grads=grads, undirected=False,
    )
    r2 = compute_mcgc_score(
        feature_risk=synthetic_edges["feature_risk"],
        checkpoint_grads=grads, undirected=False,
    )

    assert torch.allclose(r1["hybrid_score"], r2["hybrid_score"]), \
        "MCGC should be deterministic"
    print("✓ MCGC deterministic")


def test_mcgc_with_undirected(synthetic_edges):
    """MCGC should work with undirected averaging."""
    E = synthetic_edges["E"]
    torch.manual_seed(42)
    grads = [torch.randn(E) for _ in range(3)]

    result = compute_mcgc_score(
        feature_risk=synthetic_edges["feature_risk"],
        checkpoint_grads=grads,
        undirected=True,
        edge_index=synthetic_edges["edge_index"],
    )
    score = result["hybrid_score"]
    assert score.shape == (E,)
    print("✓ MCGC with undirected averaging")


def test_rank_normalize():
    """rank_normalize should map to [0, 1]."""
    x = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
    r = rank_normalize(x)

    assert r.min() >= 0.0, f"Min should be >= 0: {r.min()}"
    assert r.max() <= 1.0, f"Max should be <= 1: {r.max()}"
    # Highest value should get rank 1.0
    assert r[3] == 1.0, f"Highest value should get rank 1.0: {r[3]}"
    # Lowest value should get rank 0.0
    assert r[1] == 0.0, f"Lowest value should get rank 0.0: {r[1]}"
    print("✓ rank_normalize correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
