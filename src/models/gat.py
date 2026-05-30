import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Tuple, Union
from src.models.base import BaseGNN


class GAT(BaseGNN):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, dropout=dropout))
        else:
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_hidden: bool = False,
        retain_hidden_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        hidden_list = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if return_hidden:
                    if retain_hidden_grad:
                        x.retain_grad()
                    hidden_list.append(x)
            else:
                if return_hidden:
                    if retain_hidden_grad:
                        x.retain_grad()
                    hidden_list.append(x)

        if return_hidden:
            return x, hidden_list
        return x
