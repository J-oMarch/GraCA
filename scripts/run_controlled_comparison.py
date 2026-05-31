"""
Controlled comparison: EdgeInfluence vs Random vs Original.

All methods use:
- Same data split (same seed for data loading)
- Same downstream seed
- Same prune ratio (20%)
- Same downstream models (GCN, GAT, GraphSAGE)

Usage:
    python scripts/run_controlled_comparison.py --config configs/graca_lite_cora.yaml --seed 0
    python scripts/run_controlled_comparison.py --config configs/graca_lite_citeseer.yaml --seed 0
    python scripts/run_controlled_comparison.py --config configs/graca_lite_pubmed.yaml --seed 0
"""
import sys
import os
import argparse
import torch
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset, compute_edge_homophily
from src.training.train_proxy import train_proxy
from src.training.train_downstream import train_downstream
from src.graca.edge_influence import compute_edge_influence_scores
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.edge_scoring import compute_rho_score
from src.graca.pruning import prune_graph, compute_graph_stats
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
import torch.nn.functional as F
from scipy.stats import zscore as sp_zscore
from collections import defaultdict


def run_original(data, edge_index, config, num_features, num_classes, device, seed):
    """Run downstream on original graph."""
    set_seed(seed)
    t0 = time.time()
    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        set_seed(seed)
        res = train_downstream(model_name, data, edge_index, config, num_features, num_classes, device, seed)
        results[model_name] = res
    return results


def run_random_pruning(edge_index, num_nodes, prune_ratio, undirected, seed, device):
    """Random pruning with exact ratio control."""
    set_seed(seed)
    E = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]

    if undirected:
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        pair_keys = list(edge_key_to_indices.keys())
        num_remove = int(len(pair_keys) * prune_ratio)

        rng = np.random.RandomState(seed)
        remove_indices = rng.choice(len(pair_keys), num_remove, replace=False)
        removed_keys = set(pair_keys[i] for i in remove_indices)

        prune_mask = torch.zeros(E, dtype=torch.bool, device='cpu')
        for key in removed_keys:
            for idx in edge_key_to_indices[key]:
                prune_mask[idx] = True
    else:
        num_remove = int(E * prune_ratio)
        perm = torch.randperm(E)
        prune_mask = torch.zeros(E, dtype=torch.bool)
        prune_mask[perm[:num_remove]] = True

    # Protect self-loops
    self_loop_mask = (src == dst).cpu()
    prune_mask = prune_mask.cpu() & ~self_loop_mask

    keep_mask = ~prune_mask
    pruned_ei = edge_index.cpu()[:, keep_mask]
    stats = compute_graph_stats(pruned_ei, num_nodes, E)
    return pruned_ei, prune_mask, stats


def run_homophily_pruning(edge_index, y, train_mask, num_nodes, prune_ratio, undirected, seed):
    """Homophily pruning: remove cross-class edges using train labels only."""
    set_seed(seed)
    E = edge_index.shape[1]
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    y_cpu = y.cpu()
    train_mask_cpu = train_mask.cpu()

    # Find cross-class edges with both endpoints labeled
    both_labeled = train_mask_cpu[src] & train_mask_cpu[dst]
    same_class = y_cpu[src] == y_cpu[dst]
    hetero_candidates = both_labeled & ~same_class

    candidate_indices = torch.where(hetero_candidates)[0]

    if undirected:
        edge_key_to_indices = defaultdict(list)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            key = (min(u, v), max(u, v))
            edge_key_to_indices[key].append(i)

        candidate_pairs = set()
        for idx in candidate_indices:
            u, v = src[idx].item(), dst[idx].item()
            candidate_pairs.add((min(u, v), max(u, v)))

        candidate_pairs = list(candidate_pairs)
        num_remove = min(int(E * prune_ratio / 2), len(candidate_pairs))

        rng = np.random.RandomState(seed)
        remove_indices = rng.choice(len(candidate_pairs), num_remove, replace=False)
        removed_keys = set(candidate_pairs[i] for i in remove_indices)

        prune_mask = torch.zeros(E, dtype=torch.bool)
        for key in removed_keys:
            for idx in edge_key_to_indices[key]:
                prune_mask[idx] = True
    else:
        num_remove = min(int(E * prune_ratio), len(candidate_indices))
        prune_mask = torch.zeros(E, dtype=torch.bool)
        if num_remove > 0:
            perm = torch.randperm(len(candidate_indices))[:num_remove]
            prune_mask[candidate_indices[perm]] = True

    self_loop_mask = (src == dst).cpu()
    prune_mask = prune_mask.cpu() & ~self_loop_mask

    keep_mask = ~prune_mask
    pruned_ei = edge_index.cpu()[:, keep_mask]
    stats = compute_graph_stats(pruned_ei, num_nodes, E)
    return pruned_ei, prune_mask, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prune_ratio", type=float, default=0.20)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--noise_type", type=str, default="cross_class_oracle")
    parser.add_argument("--noise_ratio", type=float, default=0.20)
    parser.add_argument("--output_dir", type=str, default="results_clean/controlled/")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("controlled_comparison")

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    undirected = config.get("dataset", {}).get("undirected", True)

    # Single data load - all methods use same split
    set_seed(args.seed)
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    num_nodes = data.num_nodes
    E_orig = data.edge_index.shape[1]
    y_cpu = data.y.cpu()

    # Inject noise if needed
    noisy = args.noisy
    if noisy:
        logger.info(f"Injecting {args.noise_type} noise at {args.noise_ratio}...")
        noise_result = inject_noise(
            edge_index=data.edge_index.cpu(), num_nodes=num_nodes,
            noise_type=args.noise_type, noise_ratio=args.noise_ratio,
            x=data.x.cpu(), y=y_cpu, train_mask=data.train_mask.cpu(),
            seed=args.seed,
        )
        edge_index = noise_result["noisy_edge_index"].to(device)
        bad_edge_mask = noise_result["bad_edge_mask"]
    else:
        edge_index = data.edge_index
        bad_edge_mask = None

    homo_before = compute_edge_homophily(edge_index.cpu(), y_cpu)
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    exp_type = "noisy_edge" if noisy else "clean"

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = f"{args.output_dir}/controlled_results.csv"

    def _write(method, prune_ratio_actual, stats, pruned_ei, prune_mask, ds_model_name, res, det=None):
        homo_after = compute_edge_homophily(pruned_ei.cpu(), y_cpu) if pruned_ei is not None else homo_before
        write_result_row({
            "run_id": f"{method}_{ds_name}_{exp_type}_{ds_model_name}_seed{args.seed}",
            "seed": args.seed, "dataset": ds_name,
            "experiment_type": exp_type,
            "method": method, "oracle_only": False,
            "proxy_model": "-", "downstream_model": ds_model_name,
            "prune_ratio_target": args.prune_ratio,
            "actual_prune_ratio": prune_ratio_actual,
            "num_edges_before": stats.get("num_edges_before", E_orig) if stats else E_orig,
            "num_edges_after": stats.get("num_edges_after", E_orig) if stats else E_orig,
            "isolated_nodes": stats.get("isolated_nodes", 0) if stats else 0,
            "min_degree": stats.get("min_degree", 0) if stats else 0,
            "mean_degree": stats.get("mean_degree", 0) if stats else 0,
            "largest_connected_component_ratio": stats.get("largest_connected_component_ratio", 1.0) if stats else 1.0,
            "edge_homophily_before": homo_before,
            "edge_homophily_after": homo_after,
            "val_acc": res["val_acc"],
            "test_acc": res["test_acc"], "test_f1": res["test_f1"],
            "best_epoch": res["best_epoch"], "runtime": res["runtime"],
            "config_path": args.config, "graph_path": "",
            "noise_type": args.noise_type if noisy else "",
            "noise_ratio": args.noise_ratio if noisy else 0,
            "num_injected_edges": noise_result["num_injected_edges"] if noisy else 0,
            "bad_edge_precision": det["bad_edge_precision"] if det else 0,
            "bad_edge_recall": det["bad_edge_recall"] if det else 0,
            "bad_edge_f1": det["bad_edge_f1"] if det else 0,
            "clean_edge_mistakenly_removed_ratio": det["clean_edge_mistakenly_removed_ratio"] if det else 0,
        }, csv_path)

    # ============ 1. Original ============
    logger.info("Running Original...")
    for ds_model_name in downstream_names:
        set_seed(args.seed)
        t0 = time.time()
        res = train_downstream(ds_model_name, data, edge_index, config, num_features, num_classes, device, args.seed)
        res["runtime"] = time.time() - t0
        stats = {"num_edges_before": edge_index.shape[1], "num_edges_after": edge_index.shape[1],
                 "prune_ratio": 0, "isolated_nodes": 0, "min_degree": 0, "mean_degree": 0,
                 "largest_connected_component_ratio": 1.0}
        _write("Original", 0, stats, edge_index, None, ds_model_name, res)

    # ============ 2. Random-Matched ============
    logger.info("Running Random-Matched...")
    pruned_ei_rand, mask_rand, stats_rand = run_random_pruning(
        edge_index, num_nodes, args.prune_ratio, undirected, args.seed, device
    )
    det_rand = None
    if noisy and bad_edge_mask is not None:
        det_rand = evaluate_bad_edge_detection(mask_rand, bad_edge_mask, edge_index)
    for ds_model_name in downstream_names:
        set_seed(args.seed)
        t0 = time.time()
        res = train_downstream(ds_model_name, data, pruned_ei_rand, config, num_features, num_classes, device, args.seed)
        res["runtime"] = time.time() + t0 - t0  # just use downstream time
        _write("Random-Matched", stats_rand["prune_ratio"], stats_rand, pruned_ei_rand, mask_rand, ds_model_name, res, det_rand)

    # ============ 3. Homophily-TrainOnly ============
    logger.info("Running Homophily-TrainOnly...")
    pruned_ei_homo, mask_homo, stats_homo = run_homophily_pruning(
        edge_index, data.y, data.train_mask, num_nodes, args.prune_ratio, undirected, args.seed
    )
    det_homo = None
    if noisy and bad_edge_mask is not None:
        det_homo = evaluate_bad_edge_detection(mask_homo, bad_edge_mask, edge_index)
    for ds_model_name in downstream_names:
        set_seed(args.seed)
        t0 = time.time()
        res = train_downstream(ds_model_name, data, pruned_ei_homo, config, num_features, num_classes, device, args.seed)
        res["runtime"] = time.time() - t0
        _write("Homophily-TrainOnly", stats_homo["prune_ratio"], stats_homo, pruned_ei_homo, mask_homo, ds_model_name, res, det_homo)

    # ============ 4. EdgeInfluence ============
    logger.info("Running EdgeInfluence...")
    t0 = time.time()
    model, teacher, _, _ = train_proxy(config, data, num_features, num_classes, device, args.seed)
    teacher_time = time.time() - t0

    x = data.x.to(device)
    y = data.y
    train_mask = data.train_mask
    unlabeled_mask = ~train_mask
    teacher_probs = teacher.predict(x, data.edge_index)
    pseudo_cfg = config.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.6)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)
    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )
    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

    ei_result = compute_edge_influence_scores(
        teacher, x, edge_index, y, train_mask, unlabeled_mask,
        teacher_probs, rho_score, num_nodes, undirected=undirected,
    )

    # Combined score
    ds_scores = ei_result["delta_softmax_undirected"].cpu().numpy()
    src_n, dst_n = edge_index[0].cpu(), edge_index[1].cpu()
    cos_sim = F.cosine_similarity(x[src_n].cpu(), x[dst_n].cpu(), dim=1).numpy()
    combined_scores = sp_zscore(ds_scores) + sp_zscore(-cos_sim)
    P = torch.from_numpy(combined_scores).float().to(device)

    pruned_ei_ei, mask_ei, stats_ei = prune_graph(
        edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
        beta=0.2, min_degree=1, lambda_theta=0.0,
        undirected=undirected, protect_self_loops=True,
        target_prune_ratio=args.prune_ratio,
    )

    det_ei = None
    if noisy and bad_edge_mask is not None:
        det_ei = evaluate_bad_edge_detection(mask_ei, bad_edge_mask, edge_index)

    for ds_model_name in downstream_names:
        set_seed(args.seed)
        t0 = time.time()
        res = train_downstream(ds_model_name, data, pruned_ei_ei, config, num_features, num_classes, device, args.seed)
        res["runtime"] = teacher_time + (time.time() - t0)
        _write("EdgeInfluence", stats_ei["prune_ratio"], stats_ei, pruned_ei_ei, mask_ei, ds_model_name, res, det_ei)

    logger.info(f"Controlled comparison complete! Results in {csv_path}")
    logger.info(f"  Original GCN: see CSV")
    logger.info(f"  Random-Matched prune ratio: {stats_rand['prune_ratio']:.4f}")
    logger.info(f"  Homophily-TrainOnly prune ratio: {stats_homo['prune_ratio']:.4f}")
    logger.info(f"  EdgeInfluence prune ratio: {stats_ei['prune_ratio']:.4f}")
    if det_ei:
        logger.info(f"  EdgeInfluence bad-edge F1: {det_ei['bad_edge_f1']:.4f}")


if __name__ == "__main__":
    main()
