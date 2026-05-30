import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


class BaseGNN(nn.Module):
    """Base class for all GNN models with unified interface."""

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_hidden: bool = False,
        retain_hidden_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: node features [N, F]
            edge_index: edge indices [2, E]
            return_hidden: if True, also return hidden representations
            retain_hidden_grad: if True, call retain_grad() on hidden tensors

        Returns:
            logits [N, C] if return_hidden=False
            (logits [N, C], hidden_list) if return_hidden=True
        """
        raise NotImplementedError
