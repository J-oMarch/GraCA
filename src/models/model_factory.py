from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE


MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
}


def build_model(
    name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    **kwargs,
):
    """Build a GNN model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs,
    )
