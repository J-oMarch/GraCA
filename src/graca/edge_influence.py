"""
EdgeInfluence: Edge-level influence scoring via leave-one-out loss change.

For each edge (u,v):
  L_{vu} = ℓ(v; G) - ℓ(v; G \\ e)

where ℓ(v; G \\ e) is the loss at node v when edge (u,v) is removed.

Efficient computation for a 2-layer GCN:
  1. Compute full hidden h = ReLU(Â X W1)  [with all edges]
  2. For each edge (u,v), compute ablated hidden:
     h^{e\\}(v) = ReLU( (d_v · h(v) - h(u)) / (d_v - 1) )
     This correctly handles the ReLU nonlinearity.
  3. Compute ablated logits: z^{e\\}(v) = W2 · h^{e\\}(v) + b2
  4. Compute loss change: L_{vu} = ℓ(z(v)) - ℓ(z^{e\\}(v))

For undirected graphs, average both directions.
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
    """Compute edge influence scores for all edges.

    Args:
        teacher: trained EMA teacher (with .teacher attribute)
        x: node features [N, F]
        edge_index: [2, E]
        y: labels
        train_mask: boolean mask for labeled nodes
        unlabeled_mask: boolean mask for unlabeled nodes
        teacher_probs: [N, C] teacher output probabilities
        rho_score: [N] per-node reliability weights
        num_nodes: number of nodes
        undirected: whether to average scores for undirected graphs
        eps: numerical stability

    Returns:
        dict with L, L_weighted, L_undirected, diagnostics
    """
    logger = get_logger("edge_influence")
    src = edge_index[0]  # neighbor u
    dst = edge_index[1]  # target v
    E = edge_index.shape[1]

    teacher.teacher.eval()

    with torch.no_grad():
        # Step 1: Get full hidden representations and logits
        internal = teacher.teacher
        logits_full, hidden_list = internal(x, edge_index, return_hidden=True)

        # hidden_list[0] = first layer output (after BN+ReLU+Dropout)
        # logits_full = final output
        if len(hidden_list) > 1:
            h_full = hidden_list[0]  # [N, H] - first layer hidden
        else:
            h_full = hidden_list[0]

        # Step 2: Compute degree per node
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, dst, torch.ones(E, device=edge_index.device))
        deg = deg.clamp(min=1)

        # Step 3: For each edge (u,v), compute ablated hidden at v
        # h^{e\\}(v) = ReLU( (d_v · h(v) - h(u)) / (d_v - 1) )
        H_dim = h_full.shape[1]
        d_v = deg[dst]  # [E]
        d_minus_1 = (d_v - 1).clamp(min=1)

        # Ablated hidden: properly handles ReLU by applying it AFTER the correction
        # h_full[dst] is h(v) with all edges, shape [E, H]
        # h_full[src] is h(u), shape [E, H]
        h_ablated = F.relu(
            (d_v.unsqueeze(1) * h_full[dst] - h_full[src]) / d_minus_1.unsqueeze(1)
        )  # [E, H]

        # Step 4: Compute ablated logits using the output layer
        # For GCN, the output layer is: logits = h @ W_out.T + b_out
        # We need to extract W_out and b_out from the last conv layer
        last_conv = internal.convs[-1]

        if hasattr(last_conv, 'lin'):
            W_out = last_conv.lin.weight.data  # [C_out, H_in]
            b_out = last_conv.lin.bias.data if last_conv.lin.bias is not None else torch.zeros(W_out.shape[0], device=W_out.device)
        elif hasattr(last_conv, 'weight'):
            W_out = last_conv.weight.data
            b_out = last_conv.bias.data if last_conv.bias is not None else torch.zeros(W_out.shape[0], device=W_out.device)
        else:
            logger.warning("Cannot extract output layer weights")
            W_out = None

        if W_out is not None:
            # Ablated logits for each edge's destination node
            logits_ablated = h_ablated @ W_out.T + b_out  # [E, C]

            # Step 5: Compute loss change
            # Full loss at each node
            log_probs_full = F.log_softmax(logits_full, dim=-1)  # [N, C]

            # Use teacher's own predictions as targets (soft pseudo labels)
            # This measures: how much does removing edge (u,v) change v's prediction
            # relative to what the teacher predicts?
            target = teacher_probs.clone()  # [N, C]

            # Full loss per node: ℓ(v) = -Σ_c target_c(v) log p_c(v)
            loss_full = -(target * log_probs_full).sum(dim=-1)  # [N]

            # Ablated loss per edge
            log_probs_ablated = F.log_softmax(logits_ablated, dim=-1)  # [E, C]
            target_v = target[dst]  # [E, C]
            loss_ablated = -(target_v * log_probs_ablated).sum(dim=-1)  # [E]

            # Influence: L_{vu} = ℓ(v) - ℓ^{e\\}(v)
            # Positive = removing edge INCREASES loss (edge is helpful)
            # Negative = removing edge DECREASES loss (edge is harmful)
            # We want: HIGH score = HARMFUL edge
            # So: L = ℓ^{e\\}(v) - ℓ(v) = loss_ablated - loss_full[dst]
            L = loss_ablated - loss_full[dst]  # [E]

            # Also compute a version using cross-entropy with true labels (for labeled nodes)
            # This gives a cleaner signal for labeled nodes
            ce_full = F.cross_entropy(logits_full, y, reduction='none')  # [N]
            ce_ablated = F.cross_entropy(logits_ablated, y[dst], reduction='none')  # [E]
            L_ce = ce_ablated - ce_full[dst]  # [E]

            # Blend: use true labels for labeled endpoints, pseudo labels for unlabeled
            both_labeled = train_mask[src] & train_mask[dst]
            L_blended = torch.where(both_labeled, L_ce, L)
        else:
            # Fallback: use cosine similarity
            L = -F.cosine_similarity(h_full[src], h_full[dst], dim=-1, eps=eps)
            L_blended = L

    # Apply reliability weighting
    rho_v = rho_score[dst]
    rho_u = rho_score[src].clamp(min=0.05, max=1.0)
    rho_vu = rho_v * rho_u
    L_weighted = L_blended * rho_vu

    # For undirected graphs, average scores of both directions
    L_undirected = None
    if undirected:
        L_undirected = _average_undirected(edge_index, L_weighted, num_nodes)

    diagnostics = {
        "L_mean": float(L_blended.mean()),
        "L_std": float(L_blended.std()),
        "L_min": float(L_blended.min()),
        "L_max": float(L_blended.max()),
        "L_positive_frac": float((L_blended > 0).float().mean()),
        "rho_vu_mean": float(rho_vu.mean()),
        "deg_mean": float(d_v.float().mean()),
    }

    return {
        "L": L_blended,
        "L_weighted": L_weighted,
        "L_undirected": L_undirected,
        "rho_vu": rho_vu,
        "loss_full": loss_full if 'loss_full' in dir() else None,
        "diagnostics": diagnostics,
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
