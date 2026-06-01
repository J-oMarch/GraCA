import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Optional, Tuple, Union
from src.models.base import BaseGNN


class GCN(BaseGNN):
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
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_gate: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        retain_hidden_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass with optional edge gating.

        Args:
            x: [N, F] node features
            edge_index: [2, E] edge indices
            edge_gate: [E] optional edge gate weights. If None, standard GCN.
                If provided, passed as edge_weight to GCNConv.
                edge_gate.requires_grad=True enables hypergradient computation.
            return_hidden: if True, also return hidden layer outputs
            retain_hidden_grad: if True, call retain_grad() on hidden outputs

        Returns:
            logits: [N, C] output logits
            hidden_list: list of hidden tensors (if return_hidden=True)

        Note:
            When edge_gate is None, output is identical to the old version.
            When edge_gate is provided, it is passed as edge_weight to GCNConv,
            which scales message passing by edge_gate[e] for each edge e.
        """
        hidden_list = []

        for i, conv in enumerate(self.convs):
            if edge_gate is not None:
                x = conv(x, edge_index, edge_weight=edge_gate)
            else:
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
