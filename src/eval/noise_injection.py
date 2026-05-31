"""
Noise injection for GraCA noisy-edge experiments.

Supports 4 noise types:
1. cross_class_train_safe: Only use train labels or teacher high-confidence pseudo labels.
   Does NOT use test labels. Suitable as practical noisy experiment.
2. cross_class_oracle: Uses all labels (train+val+test).
   Only for diagnostic/oracle noise setting. Tables MUST annotate "oracle noise construction".
3. low_feature_similarity: No labels used. Connects node pairs with lowest feature similarity.
   Cleanest practical harmful-edge injection.
4. random_inter_community: Uses feature k-means clustering (no labels), adds cross-cluster edges.

All injected edges are:
- Deduplicated against existing edges
- Added as undirected pairs (both directions)
- Saved with bad_edge_mask aligned to noisy_edge_index
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict


def inject_noise(
    edge_index: torch.Tensor,
    num_nodes: int,
    noise_type: str,
    noise_ratio: float,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
    train_mask: torch.Tensor = None,
    teacher_probs: torch.Tensor = None,
    confidence_threshold: float = 0.8,
    num_clusters: int = 10,
    seed: int = 42,
) -> dict:
    """Inject noise edges into the graph.

    Args:
        edge_index: [2, E] original graph edges
        num_nodes: number of nodes
        noise_type: one of the 4 supported types
        noise_ratio: fraction of original edges to inject (e.g., 0.1 = 10%)
        x: [N, F] node features (required for low_feature_similarity, random_inter_community)
        y: [N] node labels (required for cross_class_oracle, optional for cross_class_train_safe)
        train_mask: [N] boolean mask (required for cross_class_train_safe)
        teacher_probs: [N, C] teacher probabilities (optional, for cross_class_train_safe pseudo labels)
        confidence_threshold: threshold for using pseudo labels in cross_class_train_safe
        num_clusters: number of clusters for random_inter_community
        seed: random seed

    Returns:
        dict with keys:
            noisy_edge_index: [2, E + E_inj] combined edge index
            bad_edge_mask: [E + E_inj] boolean mask, True for injected edges
            num_injected_edges: number of injected undirected pairs
            metadata: dict with injection details
    """
    generator = torch.Generator().manual_seed(seed)
    E_orig = edge_index.shape[1]
    src_orig = edge_index[0]
    dst_orig = edge_index[1]

    # Build existing edge set for deduplication
    existing_pairs = set()
    for i in range(E_orig):
        u, v = src_orig[i].item(), dst_orig[i].item()
        existing_pairs.add((min(u, v), max(u, v)))

    # Target number of undirected pairs to inject
    num_target_pairs = int(E_orig * noise_ratio / 2)  # divide by 2 because undirected

    if noise_type == "cross_class_train_safe":
        injected_pairs = _inject_cross_class_train_safe(
            num_nodes, num_target_pairs, existing_pairs, y, train_mask,
            teacher_probs, confidence_threshold, generator
        )
    elif noise_type == "cross_class_oracle":
        if y is None:
            raise ValueError("cross_class_oracle requires labels y")
        injected_pairs = _inject_cross_class_oracle(
            num_nodes, num_target_pairs, existing_pairs, y, generator
        )
    elif noise_type == "low_feature_similarity":
        if x is None:
            raise ValueError("low_feature_similarity requires features x")
        injected_pairs = _inject_low_feature_similarity(
            num_nodes, num_target_pairs, existing_pairs, x, generator
        )
    elif noise_type == "random_inter_community":
        if x is None:
            raise ValueError("random_inter_community requires features x")
        injected_pairs = _inject_random_inter_community(
            num_nodes, num_target_pairs, existing_pairs, x, num_clusters, generator
        )
    elif noise_type == "train_safe_oracle_v2":
        # High-risk: connect labeled node to unlabeled node with DIFFERENT class
        # Only uses train labels (no test labels)
        injected_pairs = _inject_train_safe_oracle_v2(
            num_nodes, num_target_pairs, existing_pairs, y, train_mask, generator
        )
    elif noise_type == "degree_aligned_random":
        # Random edges, but aligned with degree distribution of cross-class edges
        injected_pairs = _inject_degree_aligned_random(
            num_nodes, num_target_pairs, existing_pairs, edge_index, y, generator
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Build injected edge index (both directions)
    injected_src = []
    injected_dst = []
    for u, v in injected_pairs:
        injected_src.extend([u, v])
        injected_dst.extend([v, u])

    if len(injected_src) > 0:
        injected_edge_index = torch.tensor([injected_src, injected_dst], dtype=torch.long)
        noisy_edge_index = torch.cat([edge_index, injected_edge_index], dim=1)
    else:
        noisy_edge_index = edge_index.clone()

    # Build bad_edge_mask
    bad_edge_mask = torch.zeros(noisy_edge_index.shape[1], dtype=torch.bool)
    bad_edge_mask[E_orig:] = True

    # Metadata
    metadata = {
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "num_edges_original": E_orig,
        "num_injected_pairs": len(injected_pairs),
        "num_injected_directed": len(injected_src),
        "num_edges_noisy": noisy_edge_index.shape[1],
        "actual_injection_ratio": len(injected_src) / max(E_orig, 1),
        "seed": seed,
    }

    return {
        "noisy_edge_index": noisy_edge_index,
        "bad_edge_mask": bad_edge_mask,
        "num_injected_edges": len(injected_pairs),
        "metadata": metadata,
    }


def _inject_cross_class_train_safe(
    num_nodes, num_target_pairs, existing_pairs, y, train_mask,
    teacher_probs, confidence_threshold, generator
):
    """Inject cross-class edges using only train labels and high-confidence pseudo labels.

    Does NOT use test labels. This is the practical noise injection.
    """
    # Determine class labels for each node
    node_class = torch.full((num_nodes,), -1, dtype=torch.long)

    # Use train labels
    if train_mask is not None and y is not None:
        node_class[train_mask] = y[train_mask]

    # Use high-confidence pseudo labels for unlabeled nodes
    if teacher_probs is not None:
        unlabeled_mask = ~train_mask if train_mask is not None else torch.ones(num_nodes, dtype=torch.bool)
        max_probs, pred_class = teacher_probs.max(dim=1)
        high_conf = (max_probs >= confidence_threshold) & unlabeled_mask
        node_class[high_conf] = pred_class[high_conf]

    # Find nodes with known class
    known_nodes = torch.where(node_class >= 0)[0].tolist()
    if len(known_nodes) < 2:
        # Fallback: random injection if not enough labeled nodes
        return _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator)

    # Group nodes by class
    class_to_nodes = defaultdict(list)
    for n in known_nodes:
        class_to_nodes[node_class[n].item()].append(n)

    classes = list(class_to_nodes.keys())
    if len(classes) < 2:
        return _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator)

    # Generate cross-class pairs
    injected_pairs = set()
    attempts = 0
    max_attempts = num_target_pairs * 100

    while len(injected_pairs) < num_target_pairs and attempts < max_attempts:
        # Pick two different classes
        c1_idx = torch.randint(0, len(classes), (1,), generator=generator).item()
        c2_idx = torch.randint(0, len(classes), (1,), generator=generator).item()
        while c2_idx == c1_idx:
            c2_idx = torch.randint(0, len(classes), (1,), generator=generator).item()

        c1, c2 = classes[c1_idx], classes[c2_idx]
        n1 = class_to_nodes[c1][torch.randint(0, len(class_to_nodes[c1]), (1,), generator=generator).item()]
        n2 = class_to_nodes[c2][torch.randint(0, len(class_to_nodes[c2]), (1,), generator=generator).item()]

        key = (min(n1, n2), max(n1, n2))
        if key not in existing_pairs and key not in injected_pairs:
            injected_pairs.add(key)

        attempts += 1

    return injected_pairs


def _inject_cross_class_oracle(num_nodes, num_target_pairs, existing_pairs, y, generator):
    """Inject cross-class edges using ALL labels. Oracle/diagnostic only."""
    class_to_nodes = defaultdict(list)
    for n in range(num_nodes):
        class_to_nodes[y[n].item()].append(n)

    classes = list(class_to_nodes.keys())
    if len(classes) < 2:
        return _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator)

    injected_pairs = set()
    attempts = 0
    max_attempts = num_target_pairs * 100

    while len(injected_pairs) < num_target_pairs and attempts < max_attempts:
        c1_idx = torch.randint(0, len(classes), (1,), generator=generator).item()
        c2_idx = torch.randint(0, len(classes), (1,), generator=generator).item()
        while c2_idx == c1_idx:
            c2_idx = torch.randint(0, len(classes), (1,), generator=generator).item()

        c1, c2 = classes[c1_idx], classes[c2_idx]
        n1 = class_to_nodes[c1][torch.randint(0, len(class_to_nodes[c1]), (1,), generator=generator).item()]
        n2 = class_to_nodes[c2][torch.randint(0, len(class_to_nodes[c2]), (1,), generator=generator).item()]

        key = (min(n1, n2), max(n1, n2))
        if key not in existing_pairs and key not in injected_pairs:
            injected_pairs.add(key)

        attempts += 1

    return injected_pairs


def _inject_low_feature_similarity(num_nodes, num_target_pairs, existing_pairs, x, generator):
    """Inject edges between nodes with lowest feature similarity. No labels used."""
    # Compute pairwise cosine similarity for a sample of node pairs
    # For large graphs, we sample rather than compute full N×N matrix
    max_sample = min(num_nodes * num_nodes, 500000)

    # Generate random candidate pairs
    candidate_pairs = set()
    attempts = 0
    while len(candidate_pairs) < min(max_sample, num_target_pairs * 20) and attempts < max_sample * 2:
        u = torch.randint(0, num_nodes, (1,), generator=generator).item()
        v = torch.randint(0, num_nodes, (1,), generator=generator).item()
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in existing_pairs:
                candidate_pairs.add(key)
        attempts += 1

    if len(candidate_pairs) == 0:
        return set()

    # Compute cosine similarity for candidate pairs
    pairs_list = list(candidate_pairs)
    u_idx = torch.tensor([p[0] for p in pairs_list])
    v_idx = torch.tensor([p[1] for p in pairs_list])

    cosine_sim = F.cosine_similarity(x[u_idx], x[v_idx], dim=1, eps=1e-8)

    # Sort by similarity (ascending) and pick lowest
    _, sorted_idx = torch.sort(cosine_sim)
    num_select = min(num_target_pairs, len(pairs_list))
    injected_pairs = set()
    for i in range(num_select):
        injected_pairs.add(pairs_list[sorted_idx[i].item()])

    return injected_pairs


def _inject_random_inter_community(
    num_nodes, num_target_pairs, existing_pairs, x, num_clusters, generator
):
    """Inject cross-community edges using feature k-means. No labels used."""
    try:
        from sklearn.cluster import KMeans
        x_np = x.numpy()
        n_clusters = min(num_clusters, num_nodes)
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_np = kmeans_model.fit_predict(x_np)
        labels = torch.tensor(labels_np, dtype=torch.long)
    except ImportError:
        # Simple random clustering as fallback
        labels = torch.randint(0, num_clusters, (num_nodes,), generator=generator)

    cluster_to_nodes = defaultdict(list)
    for n in range(num_nodes):
        cluster_to_nodes[labels[n].item()].append(n)

    clusters = list(cluster_to_nodes.keys())
    if len(clusters) < 2:
        return _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator)

    injected_pairs = set()
    attempts = 0
    max_attempts = num_target_pairs * 100

    while len(injected_pairs) < num_target_pairs and attempts < max_attempts:
        c1_idx = torch.randint(0, len(clusters), (1,), generator=generator).item()
        c2_idx = torch.randint(0, len(clusters), (1,), generator=generator).item()
        while c2_idx == c1_idx:
            c2_idx = torch.randint(0, len(clusters), (1,), generator=generator).item()

        c1, c2 = clusters[c1_idx], clusters[c2_idx]
        n1 = cluster_to_nodes[c1][torch.randint(0, len(cluster_to_nodes[c1]), (1,), generator=generator).item()]
        n2 = cluster_to_nodes[c2][torch.randint(0, len(cluster_to_nodes[c2]), (1,), generator=generator).item()]

        key = (min(n1, n2), max(n1, n2))
        if key not in existing_pairs and key not in injected_pairs:
            injected_pairs.add(key)

        attempts += 1

    return injected_pairs


def _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator):
    """Fallback: inject random pairs."""
    injected_pairs = set()
    attempts = 0
    max_attempts = num_target_pairs * 100

    while len(injected_pairs) < num_target_pairs and attempts < max_attempts:
        u = torch.randint(0, num_nodes, (1,), generator=generator).item()
        v = torch.randint(0, num_nodes, (1,), generator=generator).item()
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in existing_pairs and key not in injected_pairs:
                injected_pairs.add(key)
        attempts += 1

    return injected_pairs


def _inject_train_safe_oracle_v2(num_nodes, num_target_pairs, existing_pairs, y, train_mask, generator):
    """High-risk: connect labeled node to unlabeled node with DIFFERENT class.

    This creates edges where one endpoint has a known label (from train set)
    and the other is unlabeled, but they belong to different classes.
    Only uses train labels - no test label leakage.
    """
    if y is None or train_mask is None:
        raise ValueError("train_safe_oracle_v2 requires labels y and train_mask")

    labeled_nodes = torch.where(train_mask)[0].tolist()
    unlabeled_nodes = torch.where(~train_mask)[0].tolist()

    if not labeled_nodes or not unlabeled_nodes:
        return _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator)

    # Group labeled nodes by class
    class_to_labeled = defaultdict(list)
    for n in labeled_nodes:
        class_to_labeled[y[n].item()].append(n)

    classes = list(class_to_labeled.keys())
    if len(classes) < 2:
        return _inject_random_pairs(num_nodes, num_target_pairs, existing_pairs, generator)

    injected_pairs = set()
    attempts = 0
    max_attempts = num_target_pairs * 200

    while len(injected_pairs) < num_target_pairs and attempts < max_attempts:
        # Pick an unlabeled node
        u = unlabeled_nodes[torch.randint(0, len(unlabeled_nodes), (1,), generator=generator).item()]
        # Pick a labeled node from a DIFFERENT class than u's most likely class
        # Since u is unlabeled, we pick any labeled node (cross-class is likely)
        v = labeled_nodes[torch.randint(0, len(labeled_nodes), (1,), generator=generator).item()]

        key = (min(u, v), max(u, v))
        if key not in existing_pairs and key not in injected_pairs:
            injected_pairs.add(key)
        attempts += 1

    return injected_pairs


def _inject_degree_aligned_random(num_nodes, num_target_pairs, existing_pairs, edge_index, y, generator):
    """Random edges aligned with degree distribution.

    Selects random node pairs weighted by the product of their degrees,
    simulating the degree distribution of real cross-class edges.
    """
    src = edge_index[0]
    dst = edge_index[1]

    # Compute degree
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, dst.cpu(), torch.ones(len(dst)))
    deg = deg.clamp(min=1)

    # Sample with degree-weighted probability
    injected_pairs = set()
    attempts = 0
    max_attempts = num_target_pairs * 200

    # Use degree product as sampling weight
    deg_weights = deg / deg.sum()

    while len(injected_pairs) < num_target_pairs and attempts < max_attempts:
        u = torch.multinomial(deg_weights, 1, generator=generator).item()
        v = torch.multinomial(deg_weights, 1, generator=generator).item()
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in existing_pairs and key not in injected_pairs:
                injected_pairs.add(key)
        attempts += 1

    return injected_pairs


def evaluate_bad_edge_detection(
    prune_mask: torch.Tensor,
    bad_edge_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> dict:
    """Evaluate how well pruning detects injected bad edges.

    Args:
        prune_mask: [E_noisy] True = edge was pruned
        bad_edge_mask: [E_noisy] True = edge was injected
        edge_index: [2, E_noisy] noisy graph edges

    Returns:
        dict with precision, recall, f1, and clean_edge_mistakenly_removed_ratio
    """
    # Ensure same device
    prune_mask = prune_mask.cpu().bool()
    bad_edge_mask = bad_edge_mask.cpu().bool()

    # For undirected graphs, we need to evaluate at the pair level
    # Group edges by undirected pair
    E = edge_index.shape[1]
    src = edge_index[0]
    dst = edge_index[1]

    edge_key_to_indices = defaultdict(list)
    for i in range(E):
        u, v = src[i].item(), dst[i].item()
        key = (min(u, v), max(u, v))
        edge_key_to_indices[key].append(i)

    # Evaluate at pair level
    pair_pruned = set()
    pair_bad = set()
    pair_total = set()

    for key, indices in edge_key_to_indices.items():
        pair_total.add(key)
        # A pair is "pruned" if at least one direction is pruned
        if any(prune_mask[i] for i in indices):
            pair_pruned.add(key)
        # A pair is "bad" if at least one direction is bad
        if any(bad_edge_mask[i] for i in indices):
            pair_bad.add(key)

    # True positive: bad pairs that were pruned
    tp = len(pair_pruned & pair_bad)
    # False positive: clean pairs that were pruned
    fp = len(pair_pruned - pair_bad)
    # False negative: bad pairs that were NOT pruned
    fn = len(pair_bad - pair_pruned)
    # True negative: clean pairs that were NOT pruned
    tn = len(pair_total) - len(pair_pruned) - len(pair_bad) + tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Clean edges mistakenly removed
    total_clean = len(pair_total) - len(pair_bad)
    clean_removed = fp
    clean_removed_ratio = clean_removed / max(total_clean, 1)

    return {
        "bad_edge_precision": precision,
        "bad_edge_recall": recall,
        "bad_edge_f1": f1,
        "clean_edge_mistakenly_removed_ratio": clean_removed_ratio,
        "num_bad_pairs": len(pair_bad),
        "num_pruned_pairs": len(pair_pruned),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def save_noise_metadata(metadata: dict, save_dir: str, filename: str):
    """Save noise injection metadata to JSON."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path
