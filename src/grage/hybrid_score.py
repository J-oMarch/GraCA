"""
GraGE-Hybrid Score: Combines static feature smoothness with training-dynamics calibration.

Core formula:
    R = rank(feature_risk)
    R += lambda_pos * rank(relu(dynamic_grad))
    R -= lambda_neg * rank(relu(-dynamic_grad))

Where:
    - feature_risk: e.g. 1 - cosine(x_u, x_v), higher = more risky
    - dynamic_grad: d L_score / d m_e, higher = more harmful
    - rank: rank-normalize to [0, 1]
    - relu(dynamic_grad): edges that increase score loss (harmful)
    - relu(-dynamic_grad): edges that decrease score loss (protective)

Usage:
    from src.grage.hybrid_score import compute_grage_hybrid_score

    result = compute_grage_hybrid_score(
        feature_risk=feature_risk,  # [E]
        dynamic_grad=raw_grad,      # [E]
        lambda_pos=0.25,
        lambda_neg=0.25,
    )
    hybrid_score = result["hybrid_score"]  # [E], higher = more harmful
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def rank_normalize(score: torch.Tensor) -> torch.Tensor:
    """Rank-normalize scores to [0, 1].

    Args:
        score: [E] tensor of scores

    Returns:
        [E] tensor of rank-normalized scores in [0, 1]
    """
    # Sort and assign ranks
    sorted_indices = torch.argsort(score)
    ranks = torch.zeros_like(score)
    ranks[sorted_indices] = torch.arange(len(score), dtype=torch.float32, device=score.device)
    # Normalize to [0, 1]
    ranks = ranks / max(len(score) - 1, 1)
    return ranks


def compute_grage_hybrid_score(
    feature_risk: torch.Tensor,
    dynamic_grad: torch.Tensor,
    lambda_pos: float = 0.25,
    lambda_neg: float = 0.25,
    degree: Optional[torch.Tensor] = None,
    degree_norm: bool = False,
    mode: str = "pos_neg",
    undirected: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Compute GraGE-Hybrid score combining static and dynamic signals.

    Args:
        feature_risk: [E] static feature-based risk (e.g. 1 - cosine similarity)
        dynamic_grad: [E] training-dynamics gradient (d L_score / d m_e)
        lambda_pos: weight for positive gradient (harmful edges)
        lambda_neg: weight for negative gradient (protective edges)
        degree: [N] node degrees (optional, for degree normalization)
        degree_norm: if True, normalize by degree to avoid high-degree bias
        mode: scoring mode
        undirected: if True, average scores for undirected pairs
        edge_index: [2, E] required if undirected=True
        bad_edge_mask: [E] optional, for evaluation only

    Returns:
        dict with:
            hybrid_score: [E] scores (higher = more harmful)
            diagnostics: dict
    """
    E = feature_risk.shape[0]
    device = feature_risk.device

    # Step 1: Rank-normalize feature risk
    R_feature = rank_normalize(feature_risk)

    # Step 2: Compute gradient components
    pos_grad = F.relu(dynamic_grad)
    neg_grad = F.relu(-dynamic_grad)

    # Step 3: Rank-normalize gradient components
    R_pos = rank_normalize(pos_grad)
    R_neg = rank_normalize(neg_grad)

    # Step 4: Combine based on mode
    if mode == "feature_only":
        hybrid_score = R_feature
    elif mode == "grad_only":
        hybrid_score = R_pos
    elif mode == "neg_grad":
        hybrid_score = R_neg
    elif mode == "abs_grad":
        R_abs = rank_normalize(torch.abs(dynamic_grad))
        hybrid_score = R_abs
    elif mode == "feature_plus_grad":
        hybrid_score = R_feature + lambda_pos * R_pos
    elif mode == "feature_plus_pos":
        hybrid_score = R_feature + lambda_pos * R_pos
    elif mode == "feature_pos_neg":
        hybrid_score = R_feature + lambda_pos * R_pos - lambda_neg * R_neg
    elif mode == "feature_pos_neg_degree":
        hybrid_score = R_feature + lambda_pos * R_pos - lambda_neg * R_neg
        # Apply degree normalization
        if degree_norm and degree is not None:
            hybrid_score = _apply_degree_norm(hybrid_score, edge_index, degree, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Step 5: Undirected averaging
    if undirected and edge_index is not None:
        hybrid_score = _average_undirected(edge_index, hybrid_score, device)

    # Step 6: Clip to reasonable range
    hybrid_score = hybrid_score.clamp(-2, 3)

    # Diagnostics
    diagnostics = {
        "mode": mode,
        "lambda_pos": lambda_pos,
        "lambda_neg": lambda_neg,
        "degree_norm": degree_norm,
        "feature_risk_mean": float(feature_risk.mean()),
        "dynamic_grad_mean": float(dynamic_grad.mean()),
        "pos_grad_mean": float(pos_grad.mean()),
        "neg_grad_mean": float(neg_grad.mean()),
        "hybrid_score_mean": float(hybrid_score.mean()),
        "hybrid_score_std": float(hybrid_score.std()),
    }

    # Evaluate against bad_edge_mask if provided (evaluation only)
    if bad_edge_mask is not None:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(
                bad_edge_mask.cpu().numpy(),
                hybrid_score.cpu().numpy(),
            )
            diagnostics["edge_score_auc"] = auc
        except ValueError:
            diagnostics["edge_score_auc"] = 0.5

    logger.info(f"GraGE-Hybrid ({mode}): score_mean={diagnostics['hybrid_score_mean']:.4f}, "
                f"feature_mean={diagnostics['feature_risk_mean']:.4f}, "
                f"grad_mean={diagnostics['dynamic_grad_mean']:.6f}")

    return {
        "hybrid_score": hybrid_score,
        "diagnostics": diagnostics,
    }


def _apply_degree_norm(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    degree: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Normalize scores by node degree to avoid high-degree bias.

    Edges connected to high-degree nodes get their scores reduced,
    since high-degree nodes are more likely to have edges removed.
    """
    src = edge_index[0]
    dst = edge_index[1]

    # Average degree of endpoints
    avg_deg = (degree[src] + degree[dst]) / 2.0
    avg_deg = avg_deg.clamp(min=1)

    # Normalize: divide by log(avg_deg + 1) to dampen high-degree effect
    # This prevents systematic removal of edges to high-degree nodes
    norm_factor = torch.log(avg_deg + 1)
    norm_factor = norm_factor / norm_factor.mean()  # keep scale similar

    return score / norm_factor


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
