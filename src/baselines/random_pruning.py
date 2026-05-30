import torch
from src.training.train_downstream import train_downstream
from src.utils.seed import set_seed
from src.graca.pruning import compute_graph_stats
from collections import defaultdict


def run_random_pruning(data, config, num_features, num_classes, device, seed=42,
                       prune_ratio=None, match_graca_ratio=None):
    """Random pruning baseline.

    For undirected graphs, edges are deleted in pairs (both directions together).

    Args:
        prune_ratio: fixed ratio (default: config beta)
        match_graca_ratio: if provided, use this ratio to match GraCA's actual pruning
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    undirected = config.get("dataset", {}).get("undirected", True)
    edge_index = data.edge_index.cpu()
    E = edge_index.shape[1]
    num_nodes = data.num_nodes

    if undirected:
        # Group edges by undirected pair
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        # Randomly select undirected pairs to remove
        pair_keys = list(edge_key_to_indices.keys())
        num_pairs = len(pair_keys)
        num_remove_pairs = int(num_pairs * prune_ratio)

        perm = torch.randperm(num_pairs)
        removed_keys = set()
        for idx in perm[:num_remove_pairs]:
            removed_keys.add(pair_keys[idx])

        # Build prune mask
        prune_mask = torch.zeros(E, dtype=torch.bool)
        for key in removed_keys:
            for idx in edge_key_to_indices[key]:
                prune_mask[idx] = True

        # Protect self-loops
        self_loop_mask = edge_index[0] == edge_index[1]
        prune_mask = prune_mask & ~self_loop_mask
    else:
        num_remove = int(E * prune_ratio)
        perm = torch.randperm(E)
        prune_mask = torch.zeros(E, dtype=torch.bool)
        prune_mask[perm[:num_remove]] = True

        # Protect self-loops
        self_loop_mask = edge_index[0] == edge_index[1]
        prune_mask = prune_mask & ~self_loop_mask

    keep_mask = ~prune_mask
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


def run_degree_aware_random(data, config, num_features, num_classes, device, seed=42,
                            prune_ratio=None, match_graca_ratio=None):
    """Degree-aware random pruning: each node removes ~same number of edges as GraCA.

    For undirected graphs, edges are deleted in pairs.
    """
    set_seed(seed)

    if match_graca_ratio is not None:
        prune_ratio = match_graca_ratio
    elif prune_ratio is None:
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)

    undirected = config.get("dataset", {}).get("undirected", True)
    edge_index = data.edge_index.cpu()
    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]
    num_nodes = data.num_nodes

    prune_mask = torch.zeros(E, dtype=torch.bool)
    rng = torch.Generator()
    rng.manual_seed(seed)

    if undirected:
        # Build undirected adjacency
        edge_key_to_indices = defaultdict(list)
        undirected_adj = defaultdict(set)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)
            if u != v:
                undirected_adj[u].add(v)
                undirected_adj[v].add(u)

        # Per-node budget on undirected edges
        removed_keys = set()
        for v in range(num_nodes):
            neighbors = list(undirected_adj.get(v, set()))
            bv = int(prune_ratio * len(neighbors))
            if bv <= 0 or len(neighbors) == 0:
                continue
            # Only consider edges not yet removed
            remaining = [n for n in neighbors if (min(v, n), max(v, n)) not in removed_keys]
            if len(remaining) == 0:
                continue
            bv = min(bv, len(remaining))
            perm = torch.randperm(len(remaining), generator=rng)[:bv]
            for idx in perm:
                n = remaining[idx]
                removed_keys.add((min(v, n), max(v, n)))

        for key in removed_keys:
            for idx in edge_key_to_indices[key]:
                prune_mask[idx] = True
    else:
        # Group edges by destination (directed)
        edges_by_dst = defaultdict(list)
        for i in range(E):
            edges_by_dst[dst[i].item()].append(i)

        for v, indices in edges_by_dst.items():
            bv = int(prune_ratio * len(indices))
            if bv <= 0:
                continue
            indices_t = torch.tensor(indices)
            perm = torch.randperm(len(indices_t), generator=rng)[:bv]
            prune_mask[indices_t[perm]] = True

    # Protect self-loops
    self_loop_mask = src == dst
    prune_mask = prune_mask & ~self_loop_mask

    keep_mask = ~prune_mask
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
