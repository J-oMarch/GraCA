"""
GraGE Adaptive Score: Three candidate methods for adaptive graph evolution.

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

Candidate 3: StabilityResidual-GraGE
    Trains multiple stochastic graph views (edge dropout), computes per-node
    prediction stability (entropy, JSD, variance), converts to edge scores
    via endpoint disagreement/interaction, residualizes against feature cosine,
    and uses edge-gate gradient consistency only as confidence/abstention.
    Clean story: "Prediction stability under graph perturbations provides edge
    information beyond static feature similarity."

Usage:
    from src.grage.adaptive_score import compute_faa_hybrid_score
    from src.grage.adaptive_score import compute_mcgc_score
    from src.grage.adaptive_score import compute_selective_mcgc_score
    from src.grage.adaptive_score import compute_stability_residual_score
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Callable, Literal
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


def compute_selective_mcgc_score(
    feature_risk: torch.Tensor,
    feature_similarity: torch.Tensor,
    checkpoint_grads: List[torch.Tensor],
    tau: Optional[float] = None,
    tau_quantile: float = 0.75,
    gate_type: Literal["hard", "soft"] = "hard",
    soft_k: float = 20.0,
    lambda_pos: float = 0.25,
    lambda_neg: float = 0.25,
    consistency_weight: float = 1.0,
    undirected: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """Selective MCGC score with a no-leak feature-regime gate.

    The gate activates training-dynamics terms only for feature-ambiguous edges,
    approximated by high feature similarity. Thresholds are computed from
    candidate-edge feature similarities unless ``tau`` is explicitly supplied;
    labels and oracle bad-edge masks are evaluation-only.

    score_e = R(feature_risk_e)
            + A_e C_e lambda_pos R(relu(mean_grad_e))
            - A_e C_e lambda_neg R(relu(-mean_grad_e))

    Args:
        feature_risk: [E] 1 - cosine similarity.
        feature_similarity: [E] cosine similarity, higher means more ambiguous.
        checkpoint_grads: list of [E] gradient tensors from checkpoints.
        tau: optional fixed similarity threshold. If None, uses tau_quantile.
        tau_quantile: no-leak quantile over feature_similarity for tau.
        gate_type: "hard" for indicator gate, "soft" for sigmoid gate.
        soft_k: sigmoid sharpness for soft gate.
        lambda_pos: scale for positive gradient.
        lambda_neg: scale for negative gradient.
        consistency_weight: scale for consistency confidence.
        undirected: average scores for undirected pairs.
        edge_index: [2, E] required if undirected=True.
        bad_edge_mask: [E] optional, for evaluation only.

    Returns:
        dict with hybrid_score [E], gate [E], and diagnostics.
    """
    if gate_type not in {"hard", "soft"}:
        raise ValueError(f"gate_type must be 'hard' or 'soft', got {gate_type!r}")
    if not checkpoint_grads:
        raise ValueError("checkpoint_grads must contain at least one tensor")
    if feature_risk.shape != feature_similarity.shape:
        raise ValueError("feature_risk and feature_similarity must have the same shape")

    device = feature_risk.device
    R_feature = rank_normalize(feature_risk)

    if tau is None:
        tau_tensor = torch.quantile(feature_similarity.detach(), tau_quantile)
    else:
        tau_tensor = torch.tensor(float(tau), dtype=feature_similarity.dtype, device=device)

    if gate_type == "hard":
        gate = (feature_similarity >= tau_tensor).float()
    else:
        gate = torch.sigmoid(soft_k * (feature_similarity - tau_tensor))

    grads_stack = torch.stack(checkpoint_grads, dim=0)
    K = grads_stack.shape[0]
    mean_grad = grads_stack.mean(dim=0)

    mean_sign = torch.sign(mean_grad)
    checkpoint_signs = torch.sign(grads_stack)
    agreement = (checkpoint_signs == mean_sign).float()
    agreement = agreement + (checkpoint_signs == 0).float()
    agreement = agreement.clamp(0, 1)
    consistency = agreement.mean(dim=0)

    grad_std = grads_stack.std(dim=0)
    grad_mean_abs = grads_stack.abs().mean(dim=0).clamp(min=1e-8)
    stability = torch.exp(-(grad_std / grad_mean_abs))
    confidence = rank_normalize(consistency * stability)
    effective_weight = 1.0 + consistency_weight * (confidence - 0.5)
    gated_weight = gate * effective_weight

    pos_grad = F.relu(mean_grad)
    neg_grad = F.relu(-mean_grad)
    R_pos = rank_normalize(pos_grad)
    R_neg = rank_normalize(neg_grad)

    dynamic_contribution = gated_weight * lambda_pos * R_pos - gated_weight * lambda_neg * R_neg
    hybrid_score = R_feature + dynamic_contribution

    if undirected and edge_index is not None:
        hybrid_score = _average_undirected(edge_index, hybrid_score, device)
        dynamic_contribution = _average_undirected(edge_index, dynamic_contribution, device)
        gate = _average_undirected(edge_index, gate, device)

    hybrid_score = hybrid_score.clamp(-3, 5)

    diagnostics = {
        "method": "selective_mcgc",
        "gate_type": gate_type,
        "tau": float(tau_tensor),
        "tau_quantile": tau_quantile,
        "soft_k": soft_k,
        "lambda_pos": lambda_pos,
        "lambda_neg": lambda_neg,
        "consistency_weight": consistency_weight,
        "num_checkpoints": K,
        "feature_risk_mean": float(feature_risk.mean()),
        "feature_sim_mean": float(feature_similarity.mean()),
        "mean_grad_mean": float(mean_grad.mean()),
        "consistency_mean": float(consistency.mean()),
        "stability_mean": float(stability.mean()),
        "confidence_mean": float(confidence.mean()),
        "gate_active_fraction": float((gate > 0.5).float().mean()),
        "gate_mean": float(gate.mean()),
        "dynamic_contribution_mean": float(dynamic_contribution.mean()),
        "dynamic_contribution_abs_mean": float(dynamic_contribution.abs().mean()),
        "hybrid_score_mean": float(hybrid_score.mean()),
        "hybrid_score_std": float(hybrid_score.std()),
    }

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

    logger.info(
        "Selective MCGC: gate=%s tau=%.4f active=%.3f score_mean=%.4f",
        gate_type,
        diagnostics["tau"],
        diagnostics["gate_active_fraction"],
        diagnostics["hybrid_score_mean"],
    )

    return {
        "hybrid_score": hybrid_score,
        "gate": gate,
        "dynamic_contribution": dynamic_contribution,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate 3: StabilityResidual-GraGE
# ═══════════════════════════════════════════════════════════════════════════════


def collect_multi_view_predictions(
    model_ctor: Callable,
    init_state_dict: Dict,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    num_views: int = 5,
    edge_dropout_rates: Optional[List[float]] = None,
    total_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
) -> List[torch.Tensor]:
    """Collect prediction distributions from multiple stochastic graph views.

    Each view uses a different edge dropout rate (and optionally different seed)
    to produce diverse predictions. Only training labels are used; validation
    labels are used only for early stopping (same as existing pipeline).

    Args:
        model_ctor: callable that returns a fresh model instance.
        init_state_dict: initial model state (reloaded each view for diversity).
        x: [N, F] node features.
        edge_index: [2, E] original edge index (no dropout applied externally).
        y: [N] node labels.
        train_mask: [N] training mask.
        val_mask: [N] validation mask (for early stopping only).
        num_views: number of stochastic views to collect.
        edge_dropout_rates: per-view edge dropout probabilities. If None, uses
            linearly spaced rates in [0.0, 0.3].
        total_epochs: max training epochs per view.
        lr: learning rate.
        weight_decay: weight decay.
        patience: early stopping patience.

    Returns:
        list of [N, C] softmax prediction tensors, one per view.
    """
    device = x.device
    num_classes = int(y.max().item()) + 1
    N = x.shape[0]

    if edge_dropout_rates is None:
        edge_dropout_rates = [0.05 * i for i in range(num_views)]
    assert len(edge_dropout_rates) == num_views

    predictions = []
    for view_idx, drop_rate in enumerate(edge_dropout_rates):
        view_seed = 1000 * view_idx + 7919
        torch.manual_seed(view_seed)
        np.random.seed(view_seed)

        model = model_ctor()
        model.load_state_dict(init_state_dict)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_val_acc = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, total_epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Apply edge dropout: mask out edges with probability drop_rate
            if drop_rate > 0:
                E = edge_index.shape[1]
                keep_mask = torch.rand(E, device=device) >= drop_rate
                # Always keep at least one edge per node (avoid isolated nodes)
                # by keeping self-loops unconditionally
                src, dst = edge_index[0], edge_index[1]
                is_self_loop = src == dst
                keep_mask = keep_mask | is_self_loop
                view_edge_index = edge_index[:, keep_mask]
            else:
                view_edge_index = edge_index

            logits = model(x, view_edge_index)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            # Early stopping on validation (standard, not label leakage)
            model.eval()
            with torch.no_grad():
                logits_val = model(x, edge_index)
                val_pred = logits_val[val_mask].argmax(dim=1)
                val_acc = (val_pred == y[val_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Collect predictions from best checkpoint using the ORIGINAL graph
        # (no dropout at inference — we want the model's trained belief)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_full = model(x, edge_index)
            probs = F.softmax(logits_full, dim=1)  # [N, C]

        predictions.append(probs.cpu())
        logger.info(
            f"View {view_idx}: edge_dropout={drop_rate:.2f}, "
            f"best_val_acc={best_val_acc:.4f}, "
            f"pred_entropy_mean={(-(probs * (probs + 1e-8).log()).sum(dim=1)).mean():.4f}"
        )

    return predictions


def compute_node_stability(
    predictions: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute per-node stability metrics across views.

    Args:
        predictions: list of [N, C] softmax tensors, one per view.

    Returns:
        dict with:
            node_entropy: [N] mean prediction entropy across views (higher = less certain).
            node_variance: [N] mean prediction variance across views.
            node_jsd: [N] Jensen-Shannon divergence across views (higher = less stable).
            node_confidence: [N] mean max-probability across views (higher = more confident).
            node_instability: [N] combined instability score (higher = more unstable).
    """
    stacked = torch.stack(predictions, dim=0)  # [V, N, C]
    V, N, C = stacked.shape

    # Mean prediction across views
    mean_pred = stacked.mean(dim=0)  # [N, C]

    # Per-view entropy: -sum_c p_c log p_c
    per_view_entropy = -(stacked * (stacked + 1e-8).log()).sum(dim=2)  # [V, N]
    node_entropy = per_view_entropy.mean(dim=0)  # [N]

    # Per-view variance in predicted class probabilities
    per_view_var = stacked.var(dim=0)  # [N, C]
    node_variance = per_view_var.mean(dim=1)  # [N]

    # Jensen-Shannon divergence across views
    # JSD(P_1,...,P_V) = H(mean(P)) - mean(H(P_i))
    entropy_of_mean = -(mean_pred * (mean_pred + 1e-8).log()).sum(dim=1)  # [N]
    mean_of_entropy = per_view_entropy.mean(dim=0)  # [N]
    node_jsd = (entropy_of_mean - mean_of_entropy).clamp(min=0)  # [N]

    # Confidence: mean max-probability across views
    per_view_conf = stacked.max(dim=2).values  # [V, N]
    node_confidence = per_view_conf.mean(dim=0)  # [N]

    # Combined instability: normalized combination of entropy, JSD, and inverse confidence
    # Higher = more unstable
    ent_norm = rank_normalize(node_entropy)
    jsd_norm = rank_normalize(node_jsd)
    var_norm = rank_normalize(node_variance)
    inv_conf_norm = rank_normalize(1.0 - node_confidence)

    node_instability = 0.3 * ent_norm + 0.3 * jsd_norm + 0.2 * var_norm + 0.2 * inv_conf_norm

    return {
        "node_entropy": node_entropy,
        "node_variance": node_variance,
        "node_jsd": node_jsd,
        "node_confidence": node_confidence,
        "node_instability": node_instability,
    }


def stability_to_edge_score(
    edge_index: torch.Tensor,
    node_instability: torch.Tensor,
    feature_similarity: Optional[torch.Tensor] = None,
    undirected: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert node-level instability to edge-level score.

    For edge (u, v), the stability score combines:
      - endpoint_disagreement: |instability_u - instability_v| (high = endpoints
        disagree on stability, suggesting the edge connects different regimes).
      - endpoint_interaction: instability_u * instability_v (high = both endpoints
        are unstable, edge is in a fragile region).
      - If feature_similarity is provided, edges where both endpoints are unstable
        AND features are similar get the highest score (these are the ambiguous
        edges where stability information is most valuable).

    score_e = endpoint_disagreement + endpoint_interaction

    Args:
        edge_index: [2, E] edge indices.
        node_instability: [N] per-node instability.
        feature_similarity: [E] optional cosine similarity for regime weighting.
        undirected: average scores for undirected pairs.
        device: torch device.

    Returns:
        [E] edge-level stability score (higher = more suspicious edge).
    """
    if device is None:
        device = edge_index.device

    # Ensure node_instability is on the same device as edge_index
    node_instability = node_instability.to(device)

    src = edge_index[0]
    dst = edge_index[1]

    inst_src = node_instability[src]
    inst_dst = node_instability[dst]

    # Endpoint disagreement: high when endpoints have different stability
    disagreement = (inst_src - inst_dst).abs()

    # Endpoint interaction: high when both endpoints are unstable
    interaction = inst_src * inst_dst

    # Combined edge stability score
    edge_score = disagreement + interaction

    # Optionally amplify by feature similarity (ambiguous edges)
    if feature_similarity is not None:
        sim_norm = (feature_similarity + 1.0) / 2.0  # map [-1,1] to [0,1]
        # Amplify: edges with high instability AND high similarity get boosted
        edge_score = edge_score * (1.0 + sim_norm)

    if undirected:
        edge_score = _average_undirected(edge_index, edge_score, device)

    return edge_score


def residualize_stability_score(
    stability_score: torch.Tensor,
    feature_risk: torch.Tensor,
    feature_similarity: Optional[torch.Tensor] = None,
    degree: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Residualize stability score against static feature risk.

    Removes the component of stability score that is explained by feature
    similarity, leaving only the residual signal that goes beyond features.

    The residual is computed by:
    1. Rank-normalize both stability_score and feature_risk.
    2. Compute the projection of stability onto feature_risk.
    3. Subtract the projection to get the residual.
    4. Add back feature_risk as the base score.

    final_score = R(feature_risk) + alpha * residual(stability | feature_risk)

    Args:
        stability_score: [E] raw stability edge score.
        feature_risk: [E] 1 - cosine similarity.
        feature_similarity: [E] optional, for diagnostics.
        degree: [N] optional node degrees for degree effect control.
        edge_index: [2, E] optional, required if degree is provided.

    Returns:
        dict with:
            residualized_score: [E] final score combining feature_risk and stability residual.
            residual: [E] pure stability residual after removing feature component.
            projection_ratio: float, fraction of stability variance explained by features.
            diagnostics: dict.
    """
    R_stability = rank_normalize(stability_score)
    R_feature = rank_normalize(feature_risk)

    # Compute linear projection: residual = R_stability - beta * R_feature
    # beta = cov(stability, feature) / var(feature)
    stab_mean = R_stability.mean()
    feat_mean = R_feature.mean()
    cov = ((R_stability - stab_mean) * (R_feature - feat_mean)).mean()
    feat_var = ((R_feature - feat_mean) ** 2).mean().clamp(min=1e-8)
    beta = cov / feat_var

    # Residual: what's left after removing feature-similarity component
    residual = R_stability - beta * R_feature

    # Normalize residual to [0, 1]
    residual = rank_normalize(residual)

    # Projection ratio: how much of stability is explained by features
    proj_ratio = float((beta ** 2 * feat_var) / (R_stability.var().clamp(min=1e-8)))

    # Final score: feature_risk base + scaled residual
    # The residual adds information beyond feature_risk
    alpha = 0.5  # scale for residual contribution
    residualized_score = R_feature + alpha * residual
    residualized_score = residualized_score.clamp(-2, 5)

    diagnostics = {
        "projection_beta": float(beta),
        "projection_ratio": float(proj_ratio),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "alpha": alpha,
    }

    if feature_similarity is not None:
        diagnostics["feature_sim_mean"] = float(feature_similarity.mean())
        # Correlation between residual and feature similarity
        sim_rank = rank_normalize(feature_similarity)
        residual_corr = float(((residual - residual.mean()) * (sim_rank - sim_rank.mean())).mean()
                              / (residual.std().clamp(min=1e-8) * sim_rank.std().clamp(min=1e-8)))
        diagnostics["residual_feature_sim_corr"] = residual_corr

    return {
        "residualized_score": residualized_score,
        "residual": residual,
        "projection_ratio": proj_ratio,
        "diagnostics": diagnostics,
    }


def compute_stability_residual_score(
    model_ctor: Callable,
    init_state_dict: Dict,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    feature_risk: torch.Tensor,
    feature_similarity: torch.Tensor,
    checkpoint_grads: Optional[List[torch.Tensor]] = None,
    num_views: int = 5,
    edge_dropout_rates: Optional[List[float]] = None,
    total_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    use_gradient_confidence: bool = True,
    gradient_abstention_threshold: float = 0.1,
    undirected: bool = True,
    bad_edge_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """StabilityResidual-GraGE: prediction-stability residual edge score.

    Trains multiple stochastic graph views, computes node-level prediction
    stability, converts to edge scores, residualizes against feature similarity,
    and optionally uses edge-gate gradient consistency as confidence/abstention.

    The scoring pipeline:
    1. Collect multi-view predictions (stochastic graph views).
    2. Compute per-node instability (entropy, JSD, variance).
    3. Convert to edge score: endpoint disagreement + interaction.
    4. Residualize against feature_risk to get signal beyond features.
    5. Combine: score = R(feature_risk) + alpha * residual(stability).
    6. (Optional) Abstain to feature-only when gradient confidence is low.

    Args:
        model_ctor: callable returning fresh model.
        init_state_dict: initial model state.
        x: [N, F] features.
        edge_index: [2, E] edges.
        y: [N] labels.
        train_mask: [N] training mask.
        val_mask: [N] validation mask (early stopping only).
        feature_risk: [E] 1 - cosine similarity.
        feature_similarity: [E] cosine similarity.
        checkpoint_grads: optional list of [E] gradient tensors for gradient
            confidence. If None, gradient confidence/abstention is skipped.
        num_views: number of stochastic views.
        edge_dropout_rates: per-view dropout rates. If None, uses defaults.
        total_epochs: max training epochs per view.
        lr: learning rate.
        weight_decay: weight decay.
        patience: early stopping patience.
        use_gradient_confidence: if True and checkpoint_grads provided, use
            gradient sign consistency as abstention mechanism.
        gradient_abstention_threshold: minimum gradient confidence to apply
            stability residual; below this, fall back to feature-only.
        undirected: average scores for undirected pairs.
        bad_edge_mask: [E] optional, for diagnostics only.

    Returns:
        dict with:
            edge_score: [E] final edge score (higher = more suspicious).
            residual: [E] pure stability residual.
            node_stability: dict of [N] node stability metrics.
            diagnostics: dict.
    """
    device = x.device

    # Step 1: Collect multi-view predictions
    predictions = collect_multi_view_predictions(
        model_ctor=model_ctor,
        init_state_dict=init_state_dict,
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        num_views=num_views,
        edge_dropout_rates=edge_dropout_rates,
        total_epochs=total_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    # Step 2: Compute node stability
    node_stability = compute_node_stability(predictions)

    # Step 3: Convert to edge score
    raw_edge_score = stability_to_edge_score(
        edge_index=edge_index,
        node_instability=node_stability["node_instability"],
        feature_similarity=feature_similarity,
        undirected=undirected,
        device=device,
    )

    # Step 4: Residualize against feature risk
    residual_result = residualize_stability_score(
        stability_score=raw_edge_score,
        feature_risk=feature_risk,
        feature_similarity=feature_similarity,
        edge_index=edge_index,
    )
    edge_score = residual_result["residualized_score"]
    residual = residual_result["residual"]

    # Step 5: Optional gradient confidence / abstention
    gradient_confidence = None
    abstention_fraction = 0.0

    if use_gradient_confidence and checkpoint_grads is not None:
        grads_stack = torch.stack(checkpoint_grads, dim=0)  # [K, E]
        K = grads_stack.shape[0]

        # Gradient sign consistency
        mean_grad = grads_stack.mean(dim=0)
        mean_sign = torch.sign(mean_grad)
        checkpoint_signs = torch.sign(grads_stack)
        agreement = (checkpoint_signs == mean_sign).float()
        agreement = agreement + (checkpoint_signs == 0).float()
        agreement = agreement.clamp(0, 1)
        gradient_confidence = agreement.mean(dim=0)  # [E]

        # Gradient magnitude (normalized)
        grad_magnitude = grads_stack.abs().mean(dim=0)
        grad_mag_norm = rank_normalize(grad_magnitude)

        # Combined gradient confidence: consistency × magnitude
        combined_confidence = gradient_confidence * grad_mag_norm

        # Abstention: where gradient confidence is low, fall back to feature-only
        abstain_mask = combined_confidence < gradient_abstention_threshold
        R_feature = rank_normalize(feature_risk)
        edge_score = torch.where(abstain_mask, R_feature, edge_score)
        abstention_fraction = float(abstain_mask.float().mean())

        # Undirected averaging for abstention mask
        if undirected:
            abstain_mask = _average_undirected(
                edge_index, abstain_mask.float(), device
            ) > 0.5

    # Undirected averaging for final score
    if undirected:
        edge_score = _average_undirected(edge_index, edge_score, device)

    edge_score = edge_score.clamp(-2, 5)

    # Diagnostics
    diagnostics = {
        "method": "stability_residual",
        "num_views": num_views,
        "edge_dropout_rates": edge_dropout_rates or "default",
        "use_gradient_confidence": use_gradient_confidence,
        "gradient_abstention_threshold": gradient_abstention_threshold,
        "feature_risk_mean": float(feature_risk.mean()),
        "feature_sim_mean": float(feature_similarity.mean()),
        "raw_edge_score_mean": float(raw_edge_score.mean()),
        "raw_edge_score_std": float(raw_edge_score.std()),
        "node_instability_mean": float(node_stability["node_instability"].mean()),
        "node_entropy_mean": float(node_stability["node_entropy"].mean()),
        "node_jsd_mean": float(node_stability["node_jsd"].mean()),
        "node_confidence_mean": float(node_stability["node_confidence"].mean()),
        "projection_ratio": residual_result["projection_ratio"],
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "abstention_fraction": abstention_fraction,
        "edge_score_mean": float(edge_score.mean()),
        "edge_score_std": float(edge_score.std()),
    }

    if gradient_confidence is not None:
        diagnostics["gradient_confidence_mean"] = float(gradient_confidence.mean())
        diagnostics["gradient_confidence_std"] = float(gradient_confidence.std())

    diagnostics.update(residual_result["diagnostics"])

    # Evaluate against bad_edge_mask (diagnostics only)
    if bad_edge_mask is not None:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(
                bad_edge_mask.cpu().numpy(),
                edge_score.cpu().numpy(),
            )
            diagnostics["edge_score_auc"] = auc

            # Also compute residual-only AUC
            residual_auc = roc_auc_score(
                bad_edge_mask.cpu().numpy(),
                residual.cpu().numpy(),
            )
            diagnostics["residual_auc"] = residual_auc
        except ValueError:
            diagnostics["edge_score_auc"] = 0.5
            diagnostics["residual_auc"] = 0.5

    logger.info(
        "StabilityResidual: views=%d node_instab=%.4f proj_ratio=%.3f "
        "abstain=%.3f score_mean=%.4f",
        num_views,
        diagnostics["node_instability_mean"],
        diagnostics["projection_ratio"],
        abstention_fraction,
        diagnostics["edge_score_mean"],
    )

    return {
        "edge_score": edge_score,
        "residual": residual,
        "node_stability": node_stability,
        "diagnostics": diagnostics,
    }
