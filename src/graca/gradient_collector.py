import torch
from src.training.losses import compute_scoring_loss


def collect_hidden_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    teacher_probs: torch.Tensor,
    rho_train: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    lambda_s: float,
    collect_layer: str = "all",
    deterministic: bool = True,
) -> dict:
    """Collect hidden representation gradients via scoring loss.

    Args:
        model: ProxyGNN
        x: node features
        edge_index: edge indices
        y: labels
        teacher_probs: detached teacher soft labels [N, C]
        rho_train: reliability weights [N]
        train_mask: boolean mask
        unlabeled_mask: boolean mask
        lambda_s: soft pseudo loss weight
        collect_layer: 'first', 'last', or 'all' (average all layers)
        deterministic: if True, use model.eval() to disable dropout during scoring

    Returns:
        dict with 'hidden', 'grad', 'logits', 'loss_score'
    """
    # Deterministic scoring: eval mode disables dropout
    if deterministic:
        model.eval()
    else:
        model.train()

    model.zero_grad(set_to_none=True)

    logits, hidden_list = model(
        x, edge_index, return_hidden=True, retain_hidden_grad=True
    )

    # Select layers to collect
    # 'last' = last hidden layer before output (not logits)
    # 'first' = first hidden layer
    # 'all' = all hidden layers except logits, averaged
    if collect_layer == "first":
        target_hidden = [hidden_list[0]]
    elif collect_layer == "last":
        target_hidden = [hidden_list[-2] if len(hidden_list) > 1 else hidden_list[-1]]
    else:  # 'all'
        target_hidden = hidden_list[:-1] if len(hidden_list) > 1 else hidden_list

    # Retain grad on all target layers
    for h in target_hidden:
        h.retain_grad()

    loss_score, loss_sup, loss_soft = compute_scoring_loss(
        logits=logits,
        y=y,
        train_mask=train_mask,
        unlabeled_mask=unlabeled_mask,
        teacher_probs=teacher_probs,
        rho_train=rho_train,
        lambda_s=lambda_s,
    )

    loss_score.backward()

    # Collect and average gradients across layers
    grads = []
    hiddens = []
    for h in target_hidden:
        grads.append(h.grad.detach().clone())
        hiddens.append(h.detach().clone())

    grad = torch.stack(grads, dim=0).mean(dim=0)
    hidden = torch.stack(hiddens, dim=0).mean(dim=0)

    # Sanity checks
    assert grad.shape == hidden.shape, "Grad/hidden shape mismatch"
    assert torch.isfinite(grad).all(), "Non-finite values in gradient"
    assert grad.abs().sum() > 0, "Zero gradient — scoring loss has no signal"

    model.zero_grad(set_to_none=True)

    return {
        "hidden": hidden,
        "grad": grad,
        "logits": logits.detach(),
        "loss_score": loss_score.item(),
    }


def collect_multi_checkpoint_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    teacher_probs: torch.Tensor,
    rho_train: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    lambda_s: float,
    checkpoints: list,
    deterministic: bool = True,
) -> dict:
    """Collect gradients at multiple checkpoints and average.

    Each checkpoint gets its own forward pass. teacher_probs is fixed across checkpoints.
    """
    all_grads = []
    all_hiddens = []

    for ckpt in checkpoints:
        model.load_state_dict(ckpt)
        result = collect_hidden_gradients(
            model, x, edge_index, y, teacher_probs, rho_train,
            train_mask, unlabeled_mask, lambda_s,
            collect_layer="all", deterministic=deterministic,
        )
        all_grads.append(result["grad"])
        all_hiddens.append(result["hidden"])

    grad = torch.stack(all_grads, dim=0).mean(dim=0)
    hidden = torch.stack(all_hiddens, dim=0).mean(dim=0)

    return {
        "hidden": hidden,
        "grad": grad,
        "logits": result["logits"],
        "loss_score": result["loss_score"],
    }
