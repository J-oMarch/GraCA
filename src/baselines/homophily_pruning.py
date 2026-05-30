import torch
from src.training.train_downstream import train_downstream
from src.utils.seed import set_seed
from src.graca.pruning import compute_graph_stats
from collections import defaultdict


def run_homophily_pruning(data, config, num_features, num_classes, device, seed=42,
                          prune_ratio=None, match_graca_ratio=None, oracle_mode=False):
    """Homophily pruning baseline.

    For undirected graphs, edges are deleted in pairs (both directions together).

    Args:
        oracle_mode: if True, use all labels (oracle/diagnostic only).
                     if False, only use train_mask labels (legal for practical).
        prune_ratio: fixed ratio
        match_graca_ratio: if provided, use this ratio to match GraCA
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    undirected = config.get("dataset", {}).get("undirected", True)
    edge_index = data.edge_index.cpu()
    y = data.y.cpu()
    train_mask = data.train_mask.cpu()
    E = edge_index.shape[1]
    num_nodes = data.num_nodes

    src = edge_index[0]
    dst = edge_index[1]

    if oracle_mode:
        # Oracle: use all labels
        same_class = y[src] == y[dst]
        hetero_candidates = ~same_class & (src != dst)
    else:
        # Legal: only use train labels
        both_labeled = train_mask[src] & train_mask[dst]
        same_class = y[src] == y[dst]
        hetero_candidates = both_labeled & ~same_class

    candidate_indices = torch.where(hetero_candidates)[0]

    if undirected:
        # Group candidate edges by undirected pair
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        # Get unique candidate pairs
        candidate_pairs = set()
        for idx in candidate_indices:
            u, v = src[idx].item(), dst[idx].item()
            candidate_pairs.add((min(u, v), max(u, v)))

        candidate_pairs = list(candidate_pairs)
        num_remove_pairs = min(int(E * prune_ratio / 2), len(candidate_pairs))

        if num_remove_pairs > 0:
            perm = torch.randperm(len(candidate_pairs))[:num_remove_pairs]
            removed_keys = set()
            for idx in perm:
                removed_keys.add(candidate_pairs[idx])

            keep_mask = torch.ones(E, dtype=torch.bool)
            for key in removed_keys:
                for idx in edge_key_to_indices[key]:
                    keep_mask[idx] = False

            # Protect self-loops
            self_loop_mask = src == dst
            keep_mask = keep_mask | self_loop_mask
        else:
            keep_mask = torch.ones(E, dtype=torch.bool)
    else:
        num_remove = min(int(E * prune_ratio), len(candidate_indices))

        if num_remove > 0:
            perm = torch.randperm(len(candidate_indices))[:num_remove]
            to_remove = candidate_indices[perm]

            keep_mask = torch.ones(E, dtype=torch.bool)
            keep_mask[to_remove] = False

            self_loop_mask = src == dst
            keep_mask = keep_mask | self_loop_mask
        else:
            keep_mask = torch.ones(E, dtype=torch.bool)

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
