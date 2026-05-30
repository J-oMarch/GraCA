import torch
from src.training.train_downstream import train_downstream
from src.utils.seed import set_seed


def run_random_pruning(data, config, num_features, num_classes, device, seed=42, prune_ratio=None):
    """Random pruning baseline: randomly remove edges at the same ratio as GraCA."""
    set_seed(seed)

    if prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    edge_index = data.edge_index.cpu()
    E = edge_index.shape[1]
    num_remove = int(E * prune_ratio)

    # Random selection
    perm = torch.randperm(E)
    keep_mask = torch.ones(E, dtype=torch.bool)
    keep_mask[perm[:num_remove]] = False

    # Protect self-loops
    self_loop_mask = edge_index[0] == edge_index[1]
    keep_mask = keep_mask | self_loop_mask

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
