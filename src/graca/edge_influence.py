"""
EdgeInfluence: Edge-level influence scoring via leave-one-out loss change.

Supports multiple scoring variants:
- L: basic vectorized LOO (approximate)
- L_raw: without rho weighting
- L_oracle: LOO with true labels (labeled nodes only)
- delta_softmax: P(u_class@v) change
- norm_D: gradient magnitude × direction consistency
- loo_sampling: full forward pass LOO (oracle, sampling-based)
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from src.utils.logger import get_logger


def compute_edge_influence_scores(
    teacher,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    teacher_probs: torch.Tensor,
    rho_score: torch.Tensor,
    num_nodes: int,
    undirected: bool = True,
    eps: float = 1e-12,
) -> dict:
    """Compute all edge influence score variants.

    Returns dict with keys:
        L, L_raw, L_oracle, delta_softmax, delta_entropy,
        L_undirected, L_raw_undirected, rho_vu, diagnostics
    """
    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]

    teacher.teacher.eval()

    with torch.no_grad():
        internal = teacher.teacher
        logits_full, hidden_list = internal(x, edge_index, return_hidden=True)

        # First layer hidden (after BN+ReLU+Dropout, but dropout is off in eval)
        h_full = hidden_list[0] if hidden_list else None

        # Degree per node
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, dst, torch.ones(E, device=edge_index.device))
        deg = deg.clamp(min=1)

        # Ablated hidden: h^{e\\}(v) = ReLU((d_v * h(v) - h(u)) / (d_v - 1))
        d_v = deg[dst]
        d_m1 = (d_v - 1).clamp(min=1)
        h_ablated = F.relu(
            (d_v.unsqueeze(1) * h_full[dst] - h_full[src]) / d_m1.unsqueeze(1)
        )

        # Output layer weights
        last_conv = internal.convs[-1]
        W_out = getattr(last_conv, 'lin', last_conv)
        W = W_out.weight.data
        b = W_out.bias.data if W_out.bias is not None else torch.zeros(W.shape[0], device=W.device)

        # Ablated logits
        logits_ablated = h_ablated @ W.T + b  # [E, C]

        # Full probabilities
        probs_full = torch.softmax(logits_full, dim=-1)  # [N, C]
        probs_ablated = torch.softmax(logits_ablated, dim=-1)  # [E, C]

        # === Score Variant 1: L (pseudo label target) ===
        target_pseudo = teacher_probs.clone()
        log_full = F.log_softmax(logits_full, dim=-1)
        log_ablated = F.log_softmax(logits_ablated, dim=-1)
        loss_full_pseudo = -(target_pseudo * log_full).sum(dim=-1)  # [N]
        loss_ablated_pseudo = -(target_pseudo[dst] * log_ablated).sum(dim=-1)  # [E]
        L = loss_ablated_pseudo - loss_full_pseudo[dst]  # [E]

        # === Score Variant 2: L_oracle (true labels for labeled nodes) ===
        y_one_hot = F.one_hot(y, probs_full.shape[1]).float()
        target_oracle = target_pseudo.clone()
        target_oracle[train_mask] = y_one_hot[train_mask]
        loss_full_oracle = -(target_oracle * log_full).sum(dim=-1)
        loss_ablated_oracle = -(target_oracle[dst] * log_ablated).sum(dim=-1)
        L_oracle = loss_ablated_oracle - loss_full_oracle[dst]

        # === Score Variant 3a: delta_softmax ORACLE (uses true labels y[src]) ===
        # DIAGNOSTIC ONLY - uses all labels including test labels
        p_uc_full_oracle = probs_full[dst].gather(1, y[src].unsqueeze(1)).squeeze(1)  # [E]
        p_uc_ablated_oracle = probs_ablated.gather(1, y[src].unsqueeze(1)).squeeze(1)  # [E]
        delta_softmax_oracle = p_uc_full_oracle - p_uc_ablated_oracle

        # === Score Variant 3b: delta_softmax PRACTICAL (uses pseudo labels) ===
        # For labeled nodes: use true labels (train_mask only)
        # For unlabeled nodes: use teacher's argmax prediction
        node_label = teacher_probs.argmax(dim=1)  # [N]
        node_label[train_mask] = y[train_mask]  # override with true labels for train nodes
        p_uc_full_pseudo = probs_full[dst].gather(1, node_label[src].unsqueeze(1)).squeeze(1)  # [E]
        p_uc_ablated_pseudo = probs_ablated.gather(1, node_label[src].unsqueeze(1)).squeeze(1)  # [E]
        delta_softmax_pseudo = p_uc_full_pseudo - p_uc_ablated_pseudo

        # === Score Variant 4: delta_entropy ===
        ent_full = -(probs_full[dst] * probs_full[dst].clamp(min=eps).log()).sum(-1)
        ent_ablated = -(probs_ablated * probs_ablated.clamp(min=eps).log()).sum(-1)
        delta_entropy = ent_ablated - ent_full  # positive = removing edge increases entropy

        # === Score Variant 5: prediction change indicator ===
        pred_full = probs_full[dst].argmax(-1)
        pred_ablated = probs_ablated.argmax(-1)
        pred_changed = (pred_full != pred_ablated).float()

        # === Score Variant 6: confidence change ===
        conf_full = probs_full[dst].max(dim=-1).values
        conf_ablated = probs_ablated.max(dim=-1).values
        delta_conf = conf_full - conf_ablated  # positive = removing edge decreases confidence

    # Apply rho weighting
    rho_v = rho_score[dst]
    rho_u = rho_score[src].clamp(min=0.05, max=1.0)
    rho_vu = rho_v * rho_u
    L_weighted = L * rho_vu
    L_oracle_weighted = L_oracle * rho_vu

    # Undirected averaging
    L_undirected = _average_undirected(edge_index, L_weighted, num_nodes) if undirected else L_weighted
    L_raw_undirected = _average_undirected(edge_index, L, num_nodes) if undirected else L
    L_oracle_undirected = _average_undirected(edge_index, L_oracle_weighted, num_nodes) if undirected else L_oracle_weighted
    delta_softmax_oracle_undirected = _average_undirected(edge_index, delta_softmax_oracle, num_nodes) if undirected else delta_softmax_oracle
    delta_softmax_pseudo_undirected = _average_undirected(edge_index, delta_softmax_pseudo, num_nodes) if undirected else delta_softmax_pseudo

    diagnostics = {
        "L_mean": float(L.mean()),
        "L_std": float(L.std()),
        "L_oracle_mean": float(L_oracle.mean()),
        "delta_softmax_oracle_mean": float(delta_softmax_oracle.mean()),
        "delta_softmax_pseudo_mean": float(delta_softmax_pseudo.mean()),
        "pred_change_frac": float(pred_changed.mean()),
        "rho_vu_mean": float(rho_vu.mean()),
        "deg_mean": float(d_v.float().mean()),
    }

    return {
        "L": L,
        "L_raw": L,
        "L_oracle": L_oracle,
        "L_weighted": L_weighted,
        "delta_softmax_oracle": delta_softmax_oracle,
        "delta_softmax_pseudo": delta_softmax_pseudo,
        "delta_entropy": delta_entropy,
        "delta_conf": delta_conf,
        "pred_changed": pred_changed,
        "L_undirected": L_undirected,
        "L_raw_undirected": L_raw_undirected,
        "L_oracle_undirected": L_oracle_undirected,
        "delta_softmax_oracle_undirected": delta_softmax_oracle_undirected,
        "delta_softmax_pseudo_undirected": delta_softmax_pseudo_undirected,
        "rho_vu": rho_vu,
        "diagnostics": diagnostics,
    }


def compute_loo_sampling_scores(
    teacher,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
    sample_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute LOO scores via full forward pass (oracle, sampling-based).

    This is the gold-standard LOO: for each sampled edge, do a full forward
    pass with that edge removed. Uses true labels.

    Returns dict with:
        L_loo: [sample_size] LOO scores
        edge_indices: [sample_size] indices of sampled edges
        cross_class: [sample_size] whether each edge is cross-class
    """
    import numpy as np

    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(E, min(sample_size, E), replace=False)

    teacher.teacher.eval()
    L_loo = []
    cross_class = []

    with torch.no_grad():
        for idx in sample_idx:
            u, v = src[idx].item(), dst[idx].item()
            is_cross = (y[u].item() != y[v].item())
            cross_class.append(int(is_cross))

            # Full loss at v
            logits_full = teacher.teacher(x, edge_index)
            loss_full = F.cross_entropy(logits_full[v:v+1], y[v:v+1]).item()

            # Ablated: remove edge (u,v)
            mask = torch.ones(E, dtype=torch.bool, device=edge_index.device)
            mask[idx] = False
            ei_ablated = edge_index[:, mask]
            logits_ablated = teacher.teacher(x, ei_ablated)
            loss_ablated = F.cross_entropy(logits_ablated[v:v+1], y[v:v+1]).item()

            L_loo.append(loss_ablated - loss_full)

    return {
        "L_loo": np.array(L_loo),
        "edge_indices": sample_idx,
        "cross_class": np.array(cross_class),
    }


def compute_loss_gradient_scores(
    teacher,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_nodes: int,
    undirected: bool = True,
) -> dict:
    """Compute edge scores via loss gradient aggregation.

    For each edge (u,v): score = g_v^T · h_u where g_v = ∂ℓ/∂h_v.
    This is the first-order approximation of the LOO loss change.

    Returns dict with grad_scores, grad_scores_undirected
    """
    src = edge_index[0]
    dst = edge_index[1]

    # Enable gradients on input x for gradient computation
    x_req = x.detach().requires_grad_(True)

    teacher.teacher.eval()
    teacher.teacher.zero_grad()

    # Forward pass
    logits = teacher.teacher(x_req, edge_index)

    # Loss on labeled nodes
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()

    # g_v = gradient of loss w.r.t. input x at each node
    g = x_req.grad.detach()  # [N, F]

    # Score per edge: g_v^T · x_u (first-order LOO approximation)
    g_v = g[dst]  # [E, F]
    x_u = x.detach()[src]  # [E, F]
    grad_scores = (g_v * x_u).sum(dim=-1)  # [E]

    teacher.teacher.zero_grad()

    grad_scores_ud = None
    if undirected:
        grad_scores_ud = _average_undirected(edge_index, grad_scores, num_nodes)

    return {
        "grad_scores": grad_scores,
        "grad_scores_undirected": grad_scores_ud,
    }


def _average_undirected(edge_index, scores, num_nodes):
    """Average scores for undirected edge pairs."""
    src = edge_index[0]
    dst = edge_index[1]
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
