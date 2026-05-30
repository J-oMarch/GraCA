import torch
import torch.nn.functional as F
from src.training.train_downstream import train_downstream
from src.utils.seed import set_seed
from collections import defaultdict


def run_jaccard_pruning(data, config, num_features, num_classes, device, seed=42,
                        prune_ratio=None, match_graca_ratio=None):
    """Jaccard similarity pruning: remove edges between nodes with low Jaccard similarity.
    Suitable for bag-of-words features.
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    edge_index = data.edge_index.cpu()
    x = data.x.cpu()
    E = edge_index.shape[1]

    # Compute Jaccard similarity for each edge
    src = edge_index[0]
    dst = edge_index[1]

    # Binarize features (positive = 1, else 0)
    x_bin = (x > 0).float()

    # Jaccard = |A ∩ B| / |A ∪ B|
    intersection = (x_bin[src] * x_bin[dst]).sum(dim=1)
    union = ((x_bin[src] + x_bin[dst]) > 0).float().sum(dim=1)
    jaccard = intersection / union.clamp(min=1)

    # Sort edges by Jaccard similarity (ascending) and remove lowest
    num_remove = int(E * prune_ratio)
    _, sorted_indices = torch.sort(jaccard)
    to_remove = sorted_indices[:num_remove]

    keep_mask = torch.ones(E, dtype=torch.bool)
    keep_mask[to_remove] = False

    # Protect self-loops
    self_loop_mask = src == dst
    keep_mask = keep_mask | self_loop_mask

    pruned_edge_index = edge_index[:, keep_mask]

    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        results[model_name] = train_downstream(
            model_name=model_name, data=data, edge_index=pruned_edge_index,
            config=config, num_features=num_features, num_classes=num_classes,
            device=device, seed=seed,
        )

    graph_stats = {
        "num_edges_before": E,
        "num_edges_after": keep_mask.sum().item(),
        "prune_ratio": 1.0 - keep_mask.sum().item() / E,
        "isolated_nodes": 0,
        "min_degree": 0,
        "mean_degree": 0,
    }

    return results, graph_stats


def run_cosine_pruning(data, config, num_features, num_classes, device, seed=42,
                       prune_ratio=None, match_graca_ratio=None):
    """Cosine similarity pruning: remove edges between nodes with low cosine similarity.
    Suitable for continuous features.
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    edge_index = data.edge_index.cpu()
    x = data.x.cpu()
    E = edge_index.shape[1]

    src = edge_index[0]
    dst = edge_index[1]

    # Cosine similarity per edge
    cosine_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)

    # Sort by cosine similarity (ascending) and remove lowest
    num_remove = int(E * prune_ratio)
    _, sorted_indices = torch.sort(cosine_sim)
    to_remove = sorted_indices[:num_remove]

    keep_mask = torch.ones(E, dtype=torch.bool)
    keep_mask[to_remove] = False

    self_loop_mask = src == dst
    keep_mask = keep_mask | self_loop_mask

    pruned_edge_index = edge_index[:, keep_mask]

    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        results[model_name] = train_downstream(
            model_name=model_name, data=data, edge_index=pruned_edge_index,
            config=config, num_features=num_features, num_classes=num_classes,
            device=device, seed=seed,
        )

    graph_stats = {
        "num_edges_before": E,
        "num_edges_after": keep_mask.sum().item(),
        "prune_ratio": 1.0 - keep_mask.sum().item() / E,
        "isolated_nodes": 0,
        "min_degree": 0,
        "mean_degree": 0,
    }

    return results, graph_stats
