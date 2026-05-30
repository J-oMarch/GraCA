import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import List, Tuple, Union
from src.models.base import BaseGNN


class GraphSAGE(BaseGNN):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))

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
                x = self.bns[i](x)
                x = F.relu(x)
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
