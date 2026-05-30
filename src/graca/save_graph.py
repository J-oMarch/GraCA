import torch
from pathlib import Path
from src.utils.io import ensure_dir


def save_sanitized_graph(
    edge_index: torch.Tensor,
    prune_mask: torch.Tensor,
    graph_stats: dict,
    save_dir: str,
    filename: str,
):
    """Save sanitized graph to disk."""
    ensure_dir(save_dir)
    path = f"{save_dir}/{filename}.pt"
    torch.save(
        {
            "edge_index": edge_index,
            "prune_mask": prune_mask,
            "graph_stats": graph_stats,
        },
        path,
    )
    return path


def load_sanitized_graph(path: str) -> dict:
    """Load sanitized graph from disk."""
    return torch.load(path, weights_only=False)
