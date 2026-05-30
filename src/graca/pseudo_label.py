import torch


def compute_soft_pseudo_labels(
    teacher_probs: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    tau: float,
    alpha: float,
    eps: float = 1e-12,
) -> tuple:
    """Compute soft pseudo labels, confidence, entropy, and reliability.

    Args:
        teacher_probs: teacher softmax output [N, C]
        train_mask: boolean mask for labeled training nodes
        unlabeled_mask: boolean mask for unlabeled nodes
        tau: confidence threshold
        alpha: reliability exponent
        eps: numerical stability

    Returns:
        q: teacher_probs (same reference)
        confidence: [N] max class probability
        entropy: [N] prediction entropy
        rho_train: [N] training-phase reliability weight
    """
    q = teacher_probs
    confidence = q.max(dim=-1).values  # [N]

    # Entropy: H(q_v) = -sum_k q_k * log(q_k)
    log_q = torch.log(q + eps)
    entropy = -(q * log_q).sum(dim=-1)  # [N]

    num_classes = q.shape[1]
    max_entropy = torch.log(torch.tensor(float(num_classes)))

    # Training-phase reliability
    rho_train = torch.zeros(q.shape[0], device=q.device)

    # Labeled nodes: rho = 1
    rho_train[train_mask] = 1.0

    # Unlabeled nodes with sufficient confidence
    unlabeled_high_conf = unlabeled_mask & (confidence >= tau)
    normalized_entropy = entropy / (max_entropy + eps)
    rho_unlabeled = (confidence ** alpha) * (1.0 - normalized_entropy)
    rho_unlabeled = rho_unlabeled.clamp(min=0.0, max=1.0)
    rho_train[unlabeled_high_conf] = rho_unlabeled[unlabeled_high_conf]

    # Unlabeled nodes below threshold: rho = 0 (default)
    return q, confidence, entropy, rho_train


def compute_pseudo_coverage(
    unlabeled_mask: torch.Tensor,
    confidence: torch.Tensor,
    tau: float,
) -> dict:
    """Compute pseudo label coverage statistics."""
    total_unlabeled = unlabeled_mask.sum().item()
    if total_unlabeled == 0:
        return {"coverage": 0.0, "mean_confidence": 0.0}
    high_conf = (unlabeled_mask & (confidence >= tau)).sum().item()
    mean_conf = confidence[unlabeled_mask].mean().item()
    return {
        "coverage": high_conf / total_unlabeled,
        "mean_confidence": mean_conf,
    }
