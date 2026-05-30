import torch
from src.training.train_downstream import train_downstream
from src.utils.seed import set_seed


def run_homophily_pruning(data, config, num_features, num_classes, device, seed=42, prune_ratio=None):
    """Homophily pruning baseline: remove edges between nodes with different labels.

    IMPORTANT: Only uses train_mask labels to avoid test label leakage.
    High-confidence pseudo labels could be used as an extension.
    """
    set_seed(seed)

    if prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    edge_index = data.edge_index.cpu()
    y = data.y.cpu()
    train_mask = data.train_mask.cpu()
    E = edge_index.shape[1]

    src = edge_index[0]
    dst = edge_index[1]

    # Only use train labels for homophily check
    # If either endpoint is not in train, keep the edge (conservative)
    both_labeled = train_mask[src] & train_mask[dst]
    same_class = y[src] == y[dst]

    # Heterophilic edges (both labeled, different class) are candidates for removal
    hetero_candidates = both_labeled & ~same_class
    candidate_indices = torch.where(hetero_candidates)[0]

    # Remove up to prune_ratio * E edges
    num_remove = min(int(E * prune_ratio), len(candidate_indices))

    if num_remove > 0:
        # Randomly select among heterophilic edges
        perm = torch.randperm(len(candidate_indices))[:num_remove]
        to_remove = candidate_indices[perm]

        keep_mask = torch.ones(E, dtype=torch.bool)
        keep_mask[to_remove] = False

        # Protect self-loops
        self_loop_mask = src == dst
        keep_mask = keep_mask | self_loop_mask
    else:
        keep_mask = torch.ones(E, dtype=torch.bool)

    pruned_edge_index = edge_index[:, keep_mask]

    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        results[model_name] = train_downstream(
            model_name=model_name,
            data=data,
            edge_index=pruned_edge_index,
            config=config,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )

    graph_stats = {
        "num_edges_before": E,
        "num_edges_after": keep_mask.sum().item(),
        "prune_ratio": 1.0 - keep_mask.sum().item() / E,
    }

    return results, graph_stats
