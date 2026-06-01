"""
GraGE First-Order Edge-Gate Influence.

Computes S_e = d L_score(theta*(M), M) / d m_e using first-order approximation.

Key insight: The gradient of the score loss with respect to the edge gate
directly measures how much each edge contributes to the loss. Edges with
positive gradient are harmful (reducing their gate would reduce loss).

Usage:
    from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order

    result = compute_edge_gate_influence_first_order(
        model=model,
        x=x,
        edge_index=edge_index,
        y=y,
        score_mask=score_mask,
        normalize=True,
    )
    harmful_score = result["harmful_score"]  # [E], higher = more harmful
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def compute_edge_gate_influence_first_order(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    score_mask: torch.Tensor,
    normalize: bool = True,
    undirected: bool = True,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Compute first-order edge-gate influence scores.

    S_e = d L_score / d m_e where m_e is the edge gate.

    Args:
        model: trained GCN model (will be set to eval mode)
        x: [N, F] node features
        edge_index: [2, E] edge indices
        y: [N] node labels
        score_mask: [N] boolean mask for score loss computation
            MUST come from train-internal split, NOT val/test
        normalize: if True, apply z-score normalization
        undirected: if True, average scores for undirected edge pairs
        bad_edge_mask: [E] optional, for evaluation only (not used in scoring)

    Returns:
        dict with:
            harmful_score: [E] scores (higher = more harmful)
            raw_grad: [E] raw gradient values
            diagnostics: dict with statistics
    """
    model.eval()

    E = edge_index.shape[1]
    device = x.device

    # Create edge gate with gradient
    edge_gate = torch.ones(E, device=device, requires_grad=True)

    # Forward pass
    logits = model(x, edge_index, edge_gate=edge_gate)

    # Score loss (only on score_mask nodes)
    L_score = F.cross_entropy(logits[score_mask], y[score_mask])

    # Compute gradient: d L_score / d edge_gate
    grad = torch.autograd.grad(L_score, edge_gate, create_graph=False)[0]

    # grad > 0 means reducing edge_gate reduces loss (harmful edge)
    raw_grad = grad.detach()

    # Undirected averaging
    if undirected:
        harmful_score = _average_undirected(edge_index, raw_grad, device)
    else:
        harmful_score = raw_grad.clone()

    # Normalize
    if normalize:
        harmful_score = _normalize_scores(harmful_score)

    # Diagnostics
    diagnostics = {
        "grad_mean": float(raw_grad.mean()),
        "grad_std": float(raw_grad.std()),
        "grad_min": float(raw_grad.min()),
        "grad_max": float(raw_grad.max()),
        "positive_grad_ratio": float((raw_grad > 0).float().mean()),
        "score_loss": float(L_score.item()),
        "score_mask_size": int(score_mask.sum().item()),
    }

    # Evaluate against bad_edge_mask if provided (evaluation only)
    if bad_edge_mask is not None:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(
                bad_edge_mask.cpu().numpy(),
                harmful_score.cpu().numpy(),
            )
            diagnostics["edge_score_auc"] = auc
        except ValueError:
            diagnostics["edge_score_auc"] = 0.5

    logger.info(f"GraGE-FO: grad_mean={diagnostics['grad_mean']:.6f}, "
                f"pos_ratio={diagnostics['positive_grad_ratio']:.3f}, "
                f"score_loss={diagnostics['score_loss']:.4f}")

    return {
        "harmful_score": harmful_score,
        "raw_grad": raw_grad,
        "diagnostics": diagnostics,
    }


def _average_undirected(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Average scores for undirected edge pairs."""
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()
    E = edge_index.shape[1]

    edge_key_to_indices = defaultdict(list)
    for i in range(E):
        u, v = src[i].item(), dst[i].item()
        key = (min(u, v), max(u, v))
        edge_key_to_indices[key].append(i)

    scores_avg = scores.clone()
    for key, indices in edge_key_to_indices.items():
        if len(indices) > 1:
            mean_score = scores[indices].mean()
            for idx in indices:
                scores_avg[idx] = mean_score

    return scores_avg


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize scores using z-score, then clip extreme values."""
    mean = scores.mean()
    std = scores.std()

    if std > 1e-8:
        scores_norm = (scores - mean) / std
    else:
        scores_norm = scores - mean

    # Clip extreme values to [-3, 3]
    scores_norm = scores_norm.clamp(-3, 3)

    return scores_norm
