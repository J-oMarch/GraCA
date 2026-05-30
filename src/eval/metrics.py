import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=-1).cpu().numpy()
    true = y[mask].cpu().numpy()
    return accuracy_score(true, pred)


def compute_macro_f1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=-1).cpu().numpy()
    true = y[mask].cpu().numpy()
    return f1_score(true, pred, average="macro")
