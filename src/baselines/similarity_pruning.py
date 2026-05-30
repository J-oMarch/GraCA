import torch
import torch.nn.functional as F
from src.training.train_downstream import train_downstream
from src.utils.seed import set_seed
from src.graca.pruning import compute_graph_stats
from collections import defaultdict


def run_jaccard_pruning(data, config, num_features, num_classes, device, seed=42,
                        prune_ratio=None, match_graca_ratio=None):
    """Jaccard similarity pruning: remove edges between nodes with low Jaccard similarity.
    Suitable for bag-of-words features.

    For undirected graphs, edges are deleted in pairs.
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    undirected = config.get("dataset", {}).get("undirected", True)
    edge_index = data.edge_index.cpu()
    x = data.x.cpu()
    E = edge_index.shape[1]
    num_nodes = data.num_nodes

    src = edge_index[0]
    dst = edge_index[1]

    # Binarize features (positive = 1, else 0)
    x_bin = (x > 0).float()

    # Jaccard = |A ∩ B| / |A ∪ B|
    intersection = (x_bin[src] * x_bin[dst]).sum(dim=1)
    union = ((x_bin[src] + x_bin[dst]) > 0).float().sum(dim=1)
    jaccard = intersection / union.clamp(min=1)

    if undirected:
        # Average Jaccard per undirected pair
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        pair_jaccard = {}
        for key, indices in edge_key_to_indices.items():
            pair_jaccard[key] = jaccard[indices].mean().item()

        # Sort pairs by Jaccard (ascending) and remove lowest
        sorted_pairs = sorted(pair_jaccard.items(), key=lambda x: x[1])
        num_remove_pairs = int(len(sorted_pairs) * prune_ratio)
        removed_keys = set(k for k, _ in sorted_pairs[:num_remove_pairs])

        keep_mask = torch.ones(E, dtype=torch.bool)
        for key in removed_keys:
            for idx in edge_key_to_indices[key]:
                keep_mask[idx] = False
    else:
        num_remove = int(E * prune_ratio)
        _, sorted_indices = torch.sort(jaccard)
        to_remove = sorted_indices[:num_remove]

        keep_mask = torch.ones(E, dtype=torch.bool)
        keep_mask[to_remove] = False

    # Protect self-loops
    self_loop_mask = src == dst
    keep_mask = keep_mask | self_loop_mask

    pruned_edge_index = edge_index[:, keep_mask]

    # Compute real graph stats
    graph_stats = compute_graph_stats(pruned_edge_index, num_nodes, E)

    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        results[model_name] = train_downstream(
            model_name=model_name, data=data, edge_index=pruned_edge_index,
            config=config, num_features=num_features, num_classes=num_classes,
            device=device, seed=seed,
        )

    return results, graph_stats


def run_cosine_pruning(data, config, num_features, num_classes, device, seed=42,
                       prune_ratio=None, match_graca_ratio=None):
    """Cosine similarity pruning: remove edges between nodes with low cosine similarity.
    Suitable for continuous features.

    For undirected graphs, edges are deleted in pairs.
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    undirected = config.get("dataset", {}).get("undirected", True)
    edge_index = data.edge_index.cpu()
    x = data.x.cpu()
    E = edge_index.shape[1]
    num_nodes = data.num_nodes

    src = edge_index[0]
    dst = edge_index[1]

    # Cosine similarity per edge
    cosine_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)

    if undirected:
        # Average cosine per undirected pair
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        pair_cosine = {}
        for key, indices in edge_key_to_indices.items():
            pair_cosine[key] = cosine_sim[indices].mean().item()

        sorted_pairs = sorted(pair_cosine.items(), key=lambda x: x[1])
        num_remove_pairs = int(len(sorted_pairs) * prune_ratio)
        removed_keys = set(k for k, _ in sorted_pairs[:num_remove_pairs])

        keep_mask = torch.ones(E, dtype=torch.bool)
        for key in removed_keys:
            for idx in edge_key_to_indices[key]:
                keep_mask[idx] = False
    else:
        num_remove = int(E * prune_ratio)
        _, sorted_indices = torch.sort(cosine_sim)
        to_remove = sorted_indices[:num_remove]

        keep_mask = torch.ones(E, dtype=torch.bool)
        keep_mask[to_remove] = False

    self_loop_mask = src == dst
    keep_mask = keep_mask | self_loop_mask

    pruned_edge_index = edge_index[:, keep_mask]

    # Compute real graph stats
    graph_stats = compute_graph_stats(pruned_edge_index, num_nodes, E)

    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        results[model_name] = train_downstream(
            model_name=model_name, data=data, edge_index=pruned_edge_index,
            config=config, num_features=num_features, num_classes=num_classes,
            device=device, seed=seed,
        )

    return results, graph_stats
