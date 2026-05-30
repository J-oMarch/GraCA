import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, WebKB
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
from pathlib import Path


def load_dataset(config: dict) -> tuple:
    """Load dataset and return (data, num_features, num_classes)."""
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    root = ds_cfg.get("root", "data/")

    if name in ("Cora", "CiteSeer", "PubMed"):
        dataset = Planetoid(root=root, name=name, split="public")
        data = dataset[0]
    elif name == "Actor":
        dataset = Actor(root=f"{root}Actor")
        data = dataset[0]
        # Actor has multi-split masks [N, 10]; select split 0
        if data.train_mask.dim() == 2:
            split_idx = ds_cfg.get("split_idx", 0)
            data.train_mask = data.train_mask[:, split_idx]
            data.val_mask = data.val_mask[:, split_idx]
            data.test_mask = data.test_mask[:, split_idx]
    elif name in ("Texas", "Cornell", "Wisconsin"):
        dataset = WebKB(root=f"{root}WebKB", name=name)
        data = dataset[0]
        # WebKB has multi-split masks [N, 10]; select split 0
        if data.train_mask.dim() == 2:
            split_idx = ds_cfg.get("split_idx", 0)
            data.train_mask = data.train_mask[:, split_idx]
            data.val_mask = data.val_mask[:, split_idx]
            data.test_mask = data.test_mask[:, split_idx]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Ensure undirected
    if ds_cfg.get("undirected", True):
        data.edge_index = to_undirected(data.edge_index)

    # Handle self-loops
    if ds_cfg.get("add_self_loops", False):
        data.edge_index, _ = add_self_loops(
            data.edge_index, num_nodes=data.num_nodes
        )

    # Normalize features
    if ds_cfg.get("normalize_features", False):
        row_sum = data.x.sum(dim=1, keepdim=True).clamp(min=1e-12)
        data.x = data.x / row_sum

    # Generate splits if missing
    if not hasattr(data, "train_mask") or data.train_mask is None:
        from src.data.splits import generate_splits
        data = generate_splits(data, config)

    num_features = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1

    return data, num_features, num_classes
