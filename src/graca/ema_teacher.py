import torch
import torch.nn as nn
from copy import deepcopy


class EMATeacher:
    """Exponential Moving Average teacher for stable pseudo labels."""

    def __init__(self, student_model: nn.Module, decay: float = 0.99):
        self.teacher = deepcopy(student_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, student_model: nn.Module):
        """Update teacher parameters with EMA of student."""
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), student_model.parameters()
            ):
                t_param.data.mul_(self.decay).add_(
                    s_param.data, alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get teacher softmax predictions."""
        self.teacher.eval()
        logits = self.teacher(x, edge_index)
        return torch.softmax(logits, dim=-1)

    def eval(self):
        self.teacher.eval()

    def to(self, device):
        self.teacher = self.teacher.to(device)
        return self
