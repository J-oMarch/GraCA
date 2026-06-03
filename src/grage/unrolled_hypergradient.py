"""
GraGE Unrolled Hypergradient (Functional Version).

Approximates d L_score(theta_K(m), m) / d m_e by differentiating
through K inner training steps on support_mask using functional_call
to preserve the computation graph.

Key fix: Uses torch.func.functional_call to keep theta_K(m) differentiable
w.r.t. edge_gate m_e. Previous version used .data assignment which broke
the computation graph.

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


def _functional_forward(model, params_dict, x, edge_index, edge_gate):
    """Functional forward pass using torch.func.functional_call.

    This preserves the computation graph through params_dict, so gradients
    can flow back through the parameter updates.
    """
    # functional_call replaces model parameters with the provided dict
    # and runs forward without modifying the model's actual parameters
    return torch.func.functional_call(
        model,
        params_dict,
        args=(x, edge_index),
        kwargs={"edge_gate": edge_gate},
    )


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
    through K inner training steps using functional parameter updates.

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

    # Create fresh model for functional_call
    model = model_ctor()
    model.load_state_dict(init_state_dict)
    model.train()

    # Extract parameters as a dict of tensors (not nn.Parameters)
    # Each tensor must have requires_grad=True for the gradient to flow
    params = {}
    for name, param in model.named_parameters():
        # Clone to detach from model's actual parameters
        # but keep requires_grad=True for gradient flow
        params[name] = param.detach().clone().requires_grad_(True)

    # Inner loop: K training steps on support_mask
    # Each step: theta_{k+1} = theta_k - alpha * grad_theta L_support(theta_k, m)
    # We use functional_call to preserve the computation graph
    for step in range(inner_steps):
        # Forward pass using functional_call with current params
        logits = _functional_forward(model, params, x, edge_index, edge_gate)

        # Support loss
        L_support = F.cross_entropy(logits[support_mask], y[support_mask])

        # Add weight decay
        if weight_decay > 0:
            for p in params.values():
                L_support = L_support + weight_decay * (p ** 2).sum() / 2

        # Compute gradients w.r.t. all parameters
        # retain_graph=False is fine since we're building a new graph each step
        grads = torch.autograd.grad(
            L_support,
            list(params.values()),
            create_graph=True,  # MUST be True to allow second-order gradients
        )

        # Update parameters: theta_{k+1} = theta_k - alpha * grad
        # This is a tensor operation that PRESERVES the computation graph
        new_params = {}
        for (name, p), g in zip(params.items(), grads):
            new_params[name] = p - inner_lr * g

        params = new_params

    # Outer loss: L_score on score_mask
    # Use the final params after K inner steps
    logits_final = _functional_forward(model, params, x, edge_index, edge_gate)
    L_score = F.cross_entropy(logits_final[score_mask], y[score_mask])

    # Compute gradient: d L_score / d edge_gate
    # This gradient flows through the entire unrolled computation:
    # d L_score / d m_e = d L_score / d theta_K * d theta_K / d m_e + d L_score / d m_e (direct)
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
