import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import to_undirected, add_self_loops, homophily
from pathlib import Path


def load_dataset(config: dict) -> tuple:
    """Load dataset and return (data, num_features, num_classes).

    Supported datasets:
        Homophilic: Cora, CiteSeer, PubMed, AmazonComputers, AmazonPhoto,
                     CoauthorCS, CoauthorPhysics, WikiCS
        Heterophilic: Actor, Texas, Cornell, Wisconsin,
                      Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions
        Large: ogbn-arxiv (requires ogb package)
    """
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    root = ds_cfg.get("root", "data/")

    # --- Planetoid ---
    if name in ("Cora", "CiteSeer", "PubMed"):
        dataset = Planetoid(root=root, name=name, split="public")
        data = dataset[0]

    # --- Actor ---
    elif name == "Actor":
        dataset = Actor(root=f"{root}Actor")
        data = dataset[0]
        if data.train_mask.dim() == 2:
            split_idx = ds_cfg.get("split_idx", 0)
            data.train_mask = data.train_mask[:, split_idx]
            data.val_mask = data.val_mask[:, split_idx]
            data.test_mask = data.test_mask[:, split_idx]

    # --- WebKB ---
    elif name in ("Texas", "Cornell", "Wisconsin"):
        dataset = WebKB(root=f"{root}WebKB", name=name)
        data = dataset[0]
        if data.train_mask.dim() == 2:
            split_idx = ds_cfg.get("split_idx", 0)
            data.train_mask = data.train_mask[:, split_idx]
            data.val_mask = data.val_mask[:, split_idx]
            data.test_mask = data.test_mask[:, split_idx]

    # --- Amazon ---
    elif name in ("AmazonComputers", "AmazonPhoto"):
        amz_name = name.replace("Amazon", "")
        dataset = Amazon(root=f"{root}Amazon", name=amz_name)
        data = dataset[0]
        if not hasattr(data, "train_mask") or data.train_mask is None:
            from src.data.splits import generate_splits
            data = generate_splits(data, config)

    # --- Coauthor ---
    elif name in ("CoauthorCS", "CoauthorPhysics"):
        co_name = name.replace("Coauthor", "")
        dataset = Coauthor(root=f"{root}Coauthor", name=co_name)
        data = dataset[0]
        if not hasattr(data, "train_mask") or data.train_mask is None:
            from src.data.splits import generate_splits
            data = generate_splits(data, config)

    # --- WikiCS ---
    elif name == "WikiCS":
        dataset = WikiCS(root=f"{root}WikiCS")
        data = dataset[0]
        if data.train_mask.dim() == 2:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask

    # --- OGB ---
    elif name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{root}ogb")
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask[split_idx["valid"]] = True
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze()

    # --- Heterophilic datasets from arXiv 2302.11275 ---
    elif name in ("Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"):
        data = load_heterophilic_dataset(name, root)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Ensure undirected
    if ds_cfg.get("undirected", True):
        data.edge_index = to_undirected(data.edge_index)

    # Handle self-loops
    if ds_cfg.get("add_self_loops", False):
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # Normalize features
    if ds_cfg.get("normalize_features", False):
        row_sum = data.x.sum(dim=1, keepdim=True).clamp(min=1e-12)
        data.x = data.x / row_sum

    num_features = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1

    # Log dataset info
    if ds_cfg.get("verbose", False):
        homo = compute_edge_homophily(data.edge_index, data.y)
        print(f"Dataset: {name}")
        print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
        print(f"  Features: {num_features}, Classes: {num_classes}")
        print(f"  Train: {data.train_mask.sum().item()}, "
              f"Val: {data.val_mask.sum().item()}, "
              f"Test: {data.test_mask.sum().item()}")
        print(f"  Edge Homophily: {homo:.4f}")

    return data, num_features, num_classes


def load_heterophilic_dataset(name: str, root: str) -> Data:
    """Load heterophilic datasets from arXiv 2302.11275.

    Downloads from GitHub if not cached.
    """
    import os
    import numpy as np

    url_base = "https://raw.githubusercontent.com/yandex-research/heterophil/main/data/"
    cache_dir = f"{root}heterophilic/{name}"
    os.makedirs(cache_dir, exist_ok=True)

    # Download if needed
    for fname in ["edges.npy", "features.npy", "labels.npy",
                   "train_masks.npy", "val_masks.npy", "test_masks.npy"]:
        fpath = f"{cache_dir}/{fname}"
        if not os.path.exists(fpath):
            import urllib.request
            url = f"{url_base}{name}/{fname}"
            try:
                urllib.request.urlretrieve(url, fpath)
            except Exception as e:
                raise RuntimeError(f"Failed to download {url}: {e}")

    # Load
    edges = np.load(f"{cache_dir}/edges.npy")
    features = np.load(f"{cache_dir}/features.npy")
    labels = np.load(f"{cache_dir}/labels.npy")
    train_masks = np.load(f"{cache_dir}/train_masks.npy")
    val_masks = np.load(f"{cache_dir}/val_masks.npy")
    test_masks = np.load(f"{cache_dir}/test_masks.npy")

    # Build Data object
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Use first split
    if train_masks.ndim == 2:
        train_mask = torch.tensor(train_masks[0], dtype=torch.bool)
        val_mask = torch.tensor(val_masks[0], dtype=torch.bool)
        test_mask = torch.tensor(test_masks[0], dtype=torch.bool)
    else:
        train_mask = torch.tensor(train_masks, dtype=torch.bool)
        val_mask = torch.tensor(val_masks, dtype=torch.bool)
        test_mask = torch.tensor(test_masks, dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


def compute_edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """Compute edge homophily ratio."""
    src = edge_index[0]
    dst = edge_index[1]
    same = (y[src] == y[dst]).float().mean().item()
    return same
