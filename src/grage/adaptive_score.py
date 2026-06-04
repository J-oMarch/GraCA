"""
GraGE Adaptive Score: Two new candidate methods for adaptive graph evolution.

Candidate 1: Feature-Ambiguity-Adaptive Hybrid (FAA-Hybrid)
    Amplifies gradient signal where feature similarity is high (features are
    ambiguous), and trusts static feature risk where features clearly differ.
    Clean story: "Where features are ambiguous, trust training dynamics more."

    score_e = R(feature_risk_e)
            + alpha(sim_e) * lambda_pos * R(relu(grad_e))
            - beta(sim_e)  * lambda_neg * R(relu(-grad_e))

    where alpha(sim) = base_alpha + ambig_scale * sim
          beta(sim)  = base_beta  (constant, or also scaled)

Candidate 2: Multi-Checkpoint Gradient Consistency (MCGC)
    Collects edge-gate gradients at multiple training checkpoints and uses
    sign consistency as a confidence signal. Edges with consistently harmful
    gradients across training stages are more reliably bad.
    Clean story: "Consistent gradient signals across training are more
    trustworthy than single-snapshot signals."

Usage:
    from src.grage.adaptive_score import compute_faa_hybrid_score
    from src.grage.adaptive_score import compute_mcgc_score
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def rank_normalize(score: torch.Tensor) -> torch.Tensor:
    """Rank-normalize scores to [0, 1]."""
    sorted_indices = torch.argsort(score)
    ranks = torch.zeros_like(score)
    ranks[sorted_indices] = torch.arange(len(score), dtype=torch.float32, device=score.device)
    ranks = ranks / max(len(score) - 1, 1)
    return ranks


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


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate 1: Feature-Ambiguity-Adaptive Hybrid
# ═══════════════════════════════════════════════════════════════════════════════


def compute_faa_hybrid_score(
    feature_risk: torch.Tensor,
    dynamic_grad: torch.Tensor,
    feature_similarity: torch.Tensor,
    lambda_pos: float = 0.25,
    lambda_neg: float = 0.25,
    ambig_scale: float = 1.0,
    base_alpha: float = 0.0,
    base_beta: float = 0.0,
    undirected: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Feature-Ambiguity-Adaptive Hybrid score.

    Amplifies gradient contribution where feature similarity is high (ambiguous),
    because static feature risk is less informative in that regime.

    Args:
        feature_risk: [E] 1 - cosine similarity (higher = more risky)
        dynamic_grad: [E] d L_score / d m_e (positive = harmful)
        feature_similarity: [E] cosine similarity (higher = more ambiguous)
        lambda_pos: scale for positive gradient (harmful edges)
        lambda_neg: scale for negative gradient (protective edges)
        ambig_scale: how much to amplify gradient weight for ambiguous edges
        base_alpha: minimum alpha (gradient weight for positive grad)
        base_beta: minimum beta (gradient weight for negative grad)
        undirected: average scores for undirected pairs
        edge_index: [2, E] required if undirected=True
        bad_edge_mask: [E] optional, for evaluation only

    Returns:
        dict with hybrid_score [E] and diagnostics
    """
    E = feature_risk.shape[0]
    device = feature_risk.device

    # Rank-normalize feature risk
    R_feature = rank_normalize(feature_risk)

    # Compute gradient components
    pos_grad = F.relu(dynamic_grad)
    neg_grad = F.relu(-dynamic_grad)

    # Rank-normalize gradient components
    R_pos = rank_normalize(pos_grad)
    R_neg = rank_normalize(neg_grad)

    # Feature ambiguity: high similarity = ambiguous (features don't help distinguish)
    # feature_similarity is in [-1, 1], normalize to [0, 1]
    sim_norm = (feature_similarity + 1.0) / 2.0

    # Alpha (positive gradient weight): increases with feature similarity
    # When features are similar, we trust gradient more
    alpha = base_alpha + ambig_scale * sim_norm

    # Beta (negative gradient weight): constant or also scaled
    beta = base_beta + ambig_scale * 0.5 * sim_norm

    # Adaptive hybrid score
    hybrid_score = R_feature + alpha * lambda_pos * R_pos - beta * lambda_neg * R_neg

    # Undirected averaging
    if undirected and edge_index is not None:
        hybrid_score = _average_undirected(edge_index, hybrid_score, device)

    # Clip
    hybrid_score = hybrid_score.clamp(-3, 5)

    # Diagnostics
    diagnostics = {
        "method": "faa_hybrid",
        "lambda_pos": lambda_pos,
        "lambda_neg": lambda_neg,
        "ambig_scale": ambig_scale,
        "base_alpha": base_alpha,
        "base_beta": base_beta,
        "feature_risk_mean": float(feature_risk.mean()),
        "feature_sim_mean": float(feature_similarity.mean()),
        "dynamic_grad_mean": float(dynamic_grad.mean()),
        "alpha_mean": float(alpha.mean()),
        "beta_mean": float(beta.mean()),
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

    logger.info(f"FAA-Hybrid: score_mean={diagnostics['hybrid_score_mean']:.4f}, "
                f"alpha_mean={diagnostics['alpha_mean']:.4f}, "
                f"sim_mean={diagnostics['feature_sim_mean']:.4f}")

    return {
        "hybrid_score": hybrid_score,
        "diagnostics": diagnostics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate 2: Multi-Checkpoint Gradient Consistency
# ═══════════════════════════════════════════════════════════════════════════════


def compute_mcgc_score(
    feature_risk: torch.Tensor,
    checkpoint_grads: List[torch.Tensor],
    lambda_pos: float = 0.25,
    lambda_neg: float = 0.25,
    consistency_weight: float = 1.0,
    undirected: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Multi-Checkpoint Gradient Consistency score.

    Uses gradient sign consistency across training checkpoints as a confidence
    signal. Edges with consistently harmful gradients get amplified penalty;
    edges with unstable gradients get dampened penalty.

    score_e = R(feature_risk_e)
            + consistency_e * lambda_pos * R(relu(mean_grad_e))
            - consistency_e * lambda_neg * R(relu(-mean_grad_e))

    where consistency_e = fraction of checkpoints where grad_e has same sign
          as mean_grad_e.

    Args:
        feature_risk: [E] 1 - cosine similarity
        checkpoint_grads: list of [E] gradient tensors from multiple checkpoints
        lambda_pos: scale for positive gradient
        lambda_neg: scale for negative gradient
        consistency_weight: how much to scale consistency effect
        undirected: average scores for undirected pairs
        edge_index: [2, E] required if undirected=True
        bad_edge_mask: [E] optional, for evaluation only

    Returns:
        dict with hybrid_score [E] and diagnostics
    """
    E = feature_risk.shape[0]
    device = feature_risk.device

    # Rank-normalize feature risk
    R_feature = rank_normalize(feature_risk)

    # Stack gradients: [K, E]
    grads_stack = torch.stack(checkpoint_grads, dim=0)  # [K, E]
    K = grads_stack.shape[0]

    # Mean gradient across checkpoints
    mean_grad = grads_stack.mean(dim=0)  # [E]

    # Sign consistency: fraction of checkpoints where sign matches mean sign
    mean_sign = torch.sign(mean_grad)  # [E], {-1, 0, 1}
    # For each checkpoint, check if sign matches mean sign
    checkpoint_signs = torch.sign(grads_stack)  # [K, E]
    # Agreement: checkpoint sign == mean sign (treat 0 as agreement)
    agreement = (checkpoint_signs == mean_sign).float()  # [K, E]
    # Also agree when checkpoint sign is 0
    agreement = agreement + (checkpoint_signs == 0).float()
    agreement = agreement.clamp(0, 1)
    consistency = agreement.mean(dim=0)  # [E], in [0, 1]

    # Gradient magnitude stability: low coefficient of variation = stable
    grad_std = grads_stack.std(dim=0)  # [E]
    grad_mean_abs = grads_stack.abs().mean(dim=0).clamp(min=1e-8)
    cv = grad_std / grad_mean_abs  # coefficient of variation
    stability = torch.exp(-cv)  # in (0, 1], higher = more stable

    # Combined confidence factor
    confidence = consistency * stability  # [E]
    confidence = rank_normalize(confidence)  # normalize to [0, 1]

    # Apply consistency weighting
    effective_weight = 1.0 + consistency_weight * (confidence - 0.5)

    # Gradient components
    pos_grad = F.relu(mean_grad)
    neg_grad = F.relu(-mean_grad)

    R_pos = rank_normalize(pos_grad)
    R_neg = rank_normalize(neg_grad)

    # MCGC score
    hybrid_score = (
        R_feature
        + effective_weight * lambda_pos * R_pos
        - effective_weight * lambda_neg * R_neg
    )

    # Undirected averaging
    if undirected and edge_index is not None:
        hybrid_score = _average_undirected(edge_index, hybrid_score, device)

    # Clip
    hybrid_score = hybrid_score.clamp(-3, 5)

    # Diagnostics
    diagnostics = {
        "method": "mcgc",
        "lambda_pos": lambda_pos,
        "lambda_neg": lambda_neg,
        "consistency_weight": consistency_weight,
        "num_checkpoints": K,
        "feature_risk_mean": float(feature_risk.mean()),
        "mean_grad_mean": float(mean_grad.mean()),
        "consistency_mean": float(consistency.mean()),
        "stability_mean": float(stability.mean()),
        "confidence_mean": float(confidence.mean()),
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

    logger.info(f"MCGC: score_mean={diagnostics['hybrid_score_mean']:.4f}, "
                f"consistency={diagnostics['consistency_mean']:.4f}, "
                f"confidence={diagnostics['confidence_mean']:.4f}")

    return {
        "hybrid_score": hybrid_score,
        "diagnostics": diagnostics,
    }


def collect_multi_checkpoint_grads(
    model_ctor: Callable,
    init_state_dict: Dict,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    score_mask: torch.Tensor,
    checkpoint_fractions: List[float] = [0.3, 0.5, 0.7, 0.9],
    total_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    undirected: bool = True,
) -> List[torch.Tensor]:
    """Collect edge-gate gradients at multiple training checkpoints.

    Trains a model for total_epochs and captures gradients at specified
    fractions of training.

    Args:
        model_ctor: function that creates a fresh model instance
        init_state_dict: initial model parameters
        x: [N, F] node features
        edge_index: [2, E] edge indices
        y: [N] node labels
        train_mask: [N] mask for training nodes
        score_mask: [N] mask for score loss computation
        checkpoint_fractions: list of training fractions to capture (e.g., [0.3, 0.5, 0.7, 0.9])
        total_epochs: total training epochs
        lr: learning rate
        weight_decay: weight decay
        undirected: average gradients for undirected pairs

    Returns:
        list of [E] gradient tensors, one per checkpoint
    """
    device = x.device
    E = edge_index.shape[1]

    # Create fresh model and move to correct device
    model = model_ctor()
    model.load_state_dict(init_state_dict)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Determine checkpoint epochs
    checkpoint_epochs = sorted(set(
        max(1, int(f * total_epochs)) for f in checkpoint_fractions
    ))

    collected_grads = []

    for epoch in range(1, total_epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Check if this is a checkpoint epoch
        if epoch in checkpoint_epochs:
            model.eval()
            edge_gate = torch.ones(E, device=device, requires_grad=True)
            logits_score = model(x, edge_index, edge_gate=edge_gate)
            L_score = F.cross_entropy(logits_score[score_mask], y[score_mask])
            grad = torch.autograd.grad(L_score, edge_gate, create_graph=False)[0]
            raw_grad = grad.detach()

            if undirected:
                raw_grad = _average_undirected(edge_index, raw_grad, device)

            collected_grads.append(raw_grad)

            logger.info(f"Checkpoint epoch {epoch}: grad_mean={raw_grad.mean():.6f}, "
                        f"pos_ratio={(raw_grad > 0).float().mean():.3f}")

    return collected_grads
