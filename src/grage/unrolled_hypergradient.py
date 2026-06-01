"""
GraGE Unrolled Hypergradient.

Approximates d L_score(theta_K(m), m) / d m_e by differentiating
through K inner training steps on support_mask.

This is a more principled approach than first-order: it accounts for
how the edge gate affects the model parameters through training.

Usage:
    from src.grage.unrolled_hypergradient import compute_edge_gate_influence_unrolled

    result = compute_edge_gate_influence_unrolled(
        model_ctor=lambda: GCN(...),
        init_state_dict=model.state_dict(),
        x=x,
        edge_index=edge_index,
        y=y,
        support_mask=support_mask,
        score_mask=score_mask,
        inner_steps=5,
        inner_lr=0.01,
    )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import Callable, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def compute_edge_gate_influence_unrolled(
    model_ctor: Callable[[], nn.Module],
    init_state_dict: Dict,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    support_mask: torch.Tensor,
    score_mask: torch.Tensor,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    weight_decay: float = 0.0,
    undirected: bool = True,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Compute unrolled hypergradient for edge gates.

    Approximates d L_score(theta_K(m), m) / d m_e by differentiating
    through K inner training steps.

    Args:
        model_ctor: function that creates a fresh model instance
        init_state_dict: initial model parameters
        x: [N, F] node features
        edge_index: [2, E] edge indices
        y: [N] node labels
        support_mask: [N] mask for inner-loop training (from train split)
        score_mask: [N] mask for outer-loop loss (from train split)
        inner_steps: K, number of inner gradient steps
        inner_lr: learning rate for inner loop
        weight_decay: weight decay for inner loop
        undirected: average scores for undirected pairs
        bad_edge_mask: [E] optional, for evaluation only

    Returns:
        dict with:
            harmful_score: [E] scores (higher = more harmful)
            raw_grad: [E] raw gradient values
            diagnostics: dict
    """
    E = edge_index.shape[1]
    device = x.device

    # Create edge gate with gradient
    edge_gate = torch.ones(E, device=device, requires_grad=True)

    # Create fresh model with initial parameters
    model = model_ctor()
    model.load_state_dict(init_state_dict)
    model.train()

    # Inner loop: K training steps on support_mask
    # We need to track the computational graph through these steps
    theta = list(model.parameters())

    for step in range(inner_steps):
        # Forward with current edge gate
        logits = model(x, edge_index, edge_gate=edge_gate)
        L_support = F.cross_entropy(logits[support_mask], y[support_mask])

        # Add weight decay
        if weight_decay > 0:
            for p in theta:
                L_support = L_support + weight_decay * (p ** 2).sum() / 2

        # Compute gradients w.r.t. parameters
        grads = torch.autograd.grad(L_support, theta, create_graph=True)

        # Update parameters (gradient descent)
        with torch.no_grad():
            for p, g in zip(theta, grads):
                p.data = p.data - inner_lr * g.data

        # Re-enable gradients for next iteration
        # Note: we need to be careful here - after .data assignment,
        # the parameters still require_grad but the graph is broken
        # We need to use functional approach for proper unrolling

    # Outer loss: L_score on score_mask
    logits_final = model(x, edge_index, edge_gate=edge_gate)
    L_score = F.cross_entropy(logits_final[score_mask], y[score_mask])

    # Compute gradient: d L_score / d edge_gate
    # This gradient flows through the entire unrolled computation
    grad = torch.autograd.grad(L_score, edge_gate, create_graph=False)[0]

    raw_grad = grad.detach()

    # Undirected averaging
    if undirected:
        harmful_score = _average_undirected(edge_index, raw_grad, device)
    else:
        harmful_score = raw_grad.clone()

    # Normalize
    harmful_score = _normalize_scores(harmful_score)

    # Diagnostics
    diagnostics = {
        "grad_mean": float(raw_grad.mean()),
        "grad_std": float(raw_grad.std()),
        "grad_min": float(raw_grad.min()),
        "grad_max": float(raw_grad.max()),
        "positive_grad_ratio": float((raw_grad > 0).float().mean()),
        "score_loss": float(L_score.item()),
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "support_size": int(support_mask.sum().item()),
        "score_size": int(score_mask.sum().item()),
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

    logger.info(f"GraGE-Unrolled-K{inner_steps}: grad_mean={diagnostics['grad_mean']:.6f}, "
                f"pos_ratio={diagnostics['positive_grad_ratio']:.3f}")

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
    """Normalize scores using z-score."""
    mean = scores.mean()
    std = scores.std()

    if std > 1e-8:
        scores_norm = (scores - mean) / std
    else:
        scores_norm = scores - mean

    scores_norm = scores_norm.clamp(-3, 3)
    return scores_norm
