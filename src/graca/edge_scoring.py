import torch
import torch.nn.functional as F


def compute_D(grad: torch.Tensor, edge_index: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute gradient direction consistency D_vu = cos(g_v, g_u).

    Args:
        grad: [N, H] node gradients
        edge_index: [2, E] where src=u (neighbor), dst=v (target)
        eps: numerical stability

    Returns:
        D: [E] cosine similarity per edge
    """
    src = edge_index[0]  # neighbor u
    dst = edge_index[1]  # target v
    g_u = grad[src]
    g_v = grad[dst]
    D = F.cosine_similarity(g_v, g_u, dim=-1, eps=eps)
    return D


def compute_M(
    grad: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute relative gradient strength M_vu = ||g_u|| / mean_{j in N(v)} ||g_j||.

    Args:
        grad: [N, H] node gradients
        edge_index: [2, E]
        num_nodes: total number of nodes
        eps: numerical stability

    Returns:
        M: [E] relative gradient strength per edge
    """
    src = edge_index[0]
    dst = edge_index[1]

    grad_norm = torch.norm(grad, p=2, dim=-1)  # [N]
    src_norm = grad_norm[src]  # [E]

    # scatter mean: mean neighbor norm per destination node
    sum_norm = torch.zeros(num_nodes, device=grad.device).scatter_add_(
        0, dst, src_norm
    )
    deg = torch.zeros(num_nodes, device=grad.device).scatter_add_(
        0, dst, torch.ones_like(src_norm)
    )
    mean_norm_per_dst = sum_norm / deg.clamp(min=1)

    M = src_norm / (mean_norm_per_dst[dst] + eps)
    return M


def compute_rho_score(
    teacher_probs: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    tau: float,
    alpha: float,
    epsilon_rho: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute scoring-phase reliability rho_score for each node.

    Returns:
        rho_score: [N]
    """
    confidence = teacher_probs.max(dim=-1).values
    log_q = torch.log(teacher_probs + eps)
    entropy = -(teacher_probs * log_q).sum(dim=-1)
    num_classes = teacher_probs.shape[1]
    max_entropy = torch.log(torch.tensor(float(num_classes)))

    rho_score = torch.zeros(teacher_probs.shape[0], device=teacher_probs.device)

    # Labeled: rho = 1
    rho_score[train_mask] = 1.0

    # Unlabeled with sufficient confidence
    high_conf = unlabeled_mask & (confidence >= tau)
    normalized_entropy = entropy / (max_entropy + eps)
    rho_vals = (confidence ** alpha) * (1.0 - normalized_entropy)
    rho_vals = rho_vals.clamp(min=0.0, max=1.0)
    rho_score[high_conf] = rho_vals[high_conf]

    # Unlabeled below threshold: epsilon_rho
    low_conf = unlabeled_mask & (confidence < tau)
    rho_score[low_conf] = epsilon_rho

    return rho_score


def compute_edge_scores(
    grad: torch.Tensor,
    edge_index: torch.Tensor,
    rho_score: torch.Tensor,
    num_nodes: int,
    eta: float,
    epsilon_rho: float,
    eps: float = 1e-12,
) -> dict:
    """Compute all edge-level scores: D, M, rho_vu, H, R, P.

    Returns:
        dict with D, M, rho_vu, H, R, P tensors of shape [E]
    """
    src = edge_index[0]  # neighbor u
    dst = edge_index[1]  # target v

    # Direction consistency
    D = compute_D(grad, edge_index, eps)

    # Relative strength
    M = compute_M(grad, edge_index, num_nodes, eps)

    # Edge reliability: rho_vu = rho_v * clip(rho_u, eps_rho, 1)
    rho_v = rho_score[dst]
    rho_u = rho_score[src].clamp(min=epsilon_rho, max=1.0)
    rho_vu = rho_v * rho_u

    # Helpful and harmful scores
    H = rho_vu * torch.clamp(D, min=0.0) * M
    R = rho_vu * torch.clamp(-D, min=0.0) * M

    # Risk score
    P = R - eta * H

    return {
        "D": D,
        "M": M,
        "rho_vu": rho_vu,
        "H": H,
        "R": R,
        "P": P,
    }


def average_undirected_scores(
    edge_index: torch.Tensor,
    P: torch.Tensor,
) -> torch.Tensor:
    """For undirected graphs, average risk scores of both directions.

    Args:
        edge_index: [2, E] with both (u,v) and (v,u) present
        P: [E] risk scores

    Returns:
        P_avg: [E] averaged risk scores
    """
    # Create undirected key for each edge
    src = edge_index[0]
    dst = edge_index[1]

    # For each edge (u,v), find its reverse (v,u)
    # Build a mapping from (min,max) -> list of edge indices
    edge_keys = torch.stack([torch.min(src, dst), torch.max(src, dst)], dim=0)  # [2, E]
    edge_keys_t = edge_keys.t()  # [E, 2]

    # Use a dict to group edges by undirected key
    key_to_edges = {}
    for i in range(edge_keys_t.shape[0]):
        key = (edge_keys_t[i, 0].item(), edge_keys_t[i, 1].item())
        if key not in key_to_edges:
            key_to_edges[key] = []
        key_to_edges[key].append(i)

    P_avg = P.clone()
    for key, indices in key_to_edges.items():
        if len(indices) > 1:
            mean_P = P[indices].mean()
            for idx in indices:
                P_avg[idx] = mean_P

    return P_avg
