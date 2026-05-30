import torch
import torch.nn.functional as F


def supervised_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute cross-entropy loss on masked nodes."""
    return F.cross_entropy(logits[mask], y[mask])


def soft_pseudo_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    rho_train: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    eps: float = 1e-12,
) -> tuple:
    """Compute KL-divergence soft pseudo label loss on unlabeled nodes.

    Returns:
        loss_soft: scalar tensor
        weights_sum: sum of reliability weights (for normalization)
    """
    log_probs = F.log_softmax(student_logits, dim=-1)
    loss_soft_node = F.kl_div(
        log_probs,
        teacher_probs.detach(),
        reduction="none",
    ).sum(dim=-1)

    weights = rho_train[unlabeled_mask]
    loss_soft = (weights * loss_soft_node[unlabeled_mask]).sum()
    weights_sum = weights.sum() + eps
    loss_soft = loss_soft / weights_sum
    return loss_soft, weights_sum


def compute_scoring_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    teacher_probs: torch.Tensor,
    rho_train: torch.Tensor,
    lambda_s: float,
    eps: float = 1e-12,
) -> tuple:
    """Compute deterministic scoring loss for gradient collection.

    Returns:
        loss_score: total loss for backward
        loss_sup_det: supervised component
        loss_soft_det: soft pseudo component
    """
    loss_sup_det = supervised_loss(logits, y, train_mask, eps)
    loss_soft_det, _ = soft_pseudo_loss(
        logits, teacher_probs, rho_train, unlabeled_mask, eps
    )
    loss_score = loss_sup_det + lambda_s * loss_soft_det
    return loss_score, loss_sup_det, loss_soft_det
