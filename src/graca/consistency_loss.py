"""
Consistency regularization for Full Practical GraCA.
Uses weak/strong graph augmentation to enforce prediction consistency.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, dropout_adj


def weak_augment(x: torch.Tensor, edge_index: torch.Tensor,
                 feature_dropout: float = 0.1) -> tuple:
    """Weak augmentation: light feature dropout."""
    if feature_dropout > 0 and torch.rand(1).item() < 0.5:
        mask = torch.bernoulli(torch.full_like(x, 1 - feature_dropout))
        x_weak = x * mask / (1 - feature_dropout)
    else:
        x_weak = x
    return x_weak, edge_index


def strong_augment(x: torch.Tensor, edge_index: torch.Tensor,
                   edge_dropout: float = 0.3,
                   feature_mask: float = 0.3) -> tuple:
    """Strong augmentation: edge dropout + feature masking."""
    # Edge dropout
    edge_index_strong, _ = dropout_edge(edge_index, p=edge_dropout, training=True)

    # Feature masking
    if feature_mask > 0:
        mask = torch.bernoulli(torch.full_like(x, 1 - feature_mask))
        x_strong = x * mask / (1 - feature_mask)
    else:
        x_strong = x

    return x_strong, edge_index_strong


def compute_consistency_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    rho_train: torch.Tensor,
    lambda_c: float = 1.0,
    edge_dropout: float = 0.3,
    feature_mask: float = 0.3,
    eps: float = 1e-12,
) -> tuple:
    """Compute consistency loss between weak and strong views.

    L_cons = sum_v rho_v * KL(p(v|G_weak) || p(v|G_strong))

    Returns:
        loss_cons: scalar tensor
        x_weak, edge_index_weak: weak view inputs
    """
    # Generate weak view
    x_weak, edge_index_weak = weak_augment(x, edge_index)

    # Generate strong view
    x_strong, edge_index_strong = strong_augment(x, edge_index, edge_dropout, feature_mask)

    # Forward pass on weak view (teacher-like, no dropout)
    model.eval()
    with torch.no_grad():
        logits_weak = model(x_weak, edge_index_weak)
        probs_weak = torch.softmax(logits_weak, dim=-1)

    # Forward pass on strong view (student)
    model.train()
    logits_strong = model(x_strong, edge_index_strong)
    log_probs_strong = F.log_softmax(logits_strong, dim=-1)

    # KL divergence
    kl_node = F.kl_div(log_probs_strong, probs_weak.detach(), reduction='none').sum(dim=-1)

    # Weighted by reliability
    loss_cons = (rho_train * kl_node).sum() / (rho_train.sum() + eps)
    loss_cons = lambda_c * loss_cons

    return loss_cons, x_weak, edge_index_weak
