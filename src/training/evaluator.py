import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compute accuracy and macro-F1 on masked nodes."""
    pred = logits[mask].argmax(dim=-1).cpu().numpy()
    true = y[mask].cpu().numpy()
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    return {"accuracy": acc, "macro_f1": f1}
