"""
Run EdgeInfluence pipeline: Teacher training → edge scoring → pruning → downstream evaluation.

Uses combined scoring (delta_softmax + feature_cosine) for edge influence.

Usage:
    python scripts/run_edge_influence.py --config configs/graca_lite_cora.yaml --seed 0
    python scripts/run_edge_influence.py --config configs/graca_lite_cora.yaml --seed 0 --noisy --noise_type cross_class_oracle --noise_ratio 0.20
"""
import sys
import os
import argparse
import torch
import time

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
from src.graca.save_graph import save_sanitized_graph
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger
import torch.nn.functional as F
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noisy", action="store_true", help="Run on noisy graph")
    parser.add_argument("--noise_type", type=str, default="cross_class_oracle")
    parser.add_argument("--noise_ratio", type=float, default=0.20)
    parser.add_argument("--output_dir", type=str, default="results_clean/main/")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_edge_influence")

    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    undirected = config.get("dataset", {}).get("undirected", True)
    experiment_type = "noisy_edge" if args.noisy else "clean"

    for seed in seeds:
        set_seed(seed)
        logger.info(f"=== EdgeInfluence on {ds_name}, seed={seed}, noisy={args.noisy} ===")

        # 1. Load data
        data, num_features, num_classes = load_dataset(config)
        data = data.to(device)
        num_nodes = data.num_nodes

        # 2. Inject noise if needed
        if args.noisy:
            logger.info(f"Injecting {args.noise_type} noise at {args.noise_ratio}...")
            noise_result = inject_noise(
                edge_index=data.edge_index.cpu(), num_nodes=num_nodes,
                noise_type=args.noise_type, noise_ratio=args.noise_ratio,
                x=data.x.cpu(), y=data.y.cpu(), train_mask=data.train_mask.cpu(),
                seed=seed,
            )
            edge_index = noise_result["noisy_edge_index"].to(device)
            bad_edge_mask = noise_result["bad_edge_mask"]
            E_orig = data.edge_index.shape[1]
        else:
            edge_index = data.edge_index
            bad_edge_mask = None
            E_orig = edge_index.shape[1]

        # 3. Train teacher
        logger.info("Training teacher...")
        t0 = time.time()
        model, teacher, train_log, _ = train_proxy(config, data, num_features, num_classes, device, seed)
        teacher_time = time.time() - t0

        # 4. Compute edge influence scores
        logger.info("Computing edge influence scores...")
        x = data.x.to(device)
        y = data.y
        train_mask = data.train_mask
        unlabeled_mask = ~train_mask

        teacher_probs = teacher.predict(x, data.edge_index)  # use clean graph for teacher
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

        # 5. Compute combined score: delta_softmax + feature_cosine
        from scipy.stats import zscore as sp_zscore
        ds_scores = ei_result["delta_softmax_undirected"].cpu().numpy()
        src_n, dst_n = edge_index[0].cpu(), edge_index[1].cpu()
        cos_sim = F.cosine_similarity(x[src_n].cpu(), x[dst_n].cpu(), dim=1).numpy()
        combined_scores = sp_zscore(ds_scores) + sp_zscore(-cos_sim)
        P = torch.from_numpy(combined_scores).float().to(device)

        logger.info(f"Score stats: delta_sm mean={ds_scores.mean():.4f}, cos mean={cos_sim.mean():.4f}")

        # 6. Prune with exact ratio control
        pruning_cfg = config.get("pruning", {})
        target_ratio = pruning_cfg.get("target_prune_ratio", 0.20)
        pruned_edge_index, prune_mask, graph_stats = prune_graph(
            edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
            beta=pruning_cfg.get("beta", 0.2),
            min_degree=pruning_cfg.get("min_degree", 1),
            lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
            undirected=undirected,
            protect_self_loops=pruning_cfg.get("protect_self_loops", True),
            target_prune_ratio=target_ratio,
        )

        y_cpu = y.cpu()
        homo_before = compute_edge_homophily(edge_index.cpu(), y_cpu)
        homo_after = compute_edge_homophily(pruned_edge_index.cpu(), y_cpu)

        logger.info(f"Pruning: {graph_stats['num_edges_before']} -> {graph_stats['num_edges_after']} "
                     f"(ratio={graph_stats['prune_ratio']:.3f})")
        logger.info(f"Homophily: {homo_before:.4f} -> {homo_after:.4f}")

        # 7. Evaluate bad-edge detection if noisy
        det = {"bad_edge_precision": 0, "bad_edge_recall": 0, "bad_edge_f1": 0,
               "clean_edge_mistakenly_removed_ratio": 0}
        if args.noisy and bad_edge_mask is not None:
            det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask.to(device), edge_index)
            logger.info(f"Bad-edge: P={det['bad_edge_precision']:.4f} R={det['bad_edge_recall']:.4f} F1={det['bad_edge_f1']:.4f}")

        # 8. Save sanitized graph
        graph_dir = config.get("logging", {}).get("graph_dir", "sanitized_graphs_clean/edge_influence/")
        graph_path = save_sanitized_graph(
            pruned_edge_index, prune_mask, graph_stats, graph_dir, f"{ds_name}_seed{seed}"
        )

        # 9. Downstream evaluation
        logger.info("Training downstream models...")
        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])

        for ds_model_name in downstream_names:
            set_seed(seed)
            t0 = time.time()
            ds_result = train_downstream(
                model_name=ds_model_name, data=data,
                edge_index=pruned_edge_index, config=config,
                num_features=num_features, num_classes=num_classes,
                device=device, seed=seed,
            )
            runtime = time.time() - t0

            write_result_row({
                "run_id": f"edge_influence_{ds_name}_{args.noise_type if args.noisy else 'clean'}_{ds_model_name}_seed{seed}",
                "seed": seed, "dataset": ds_name,
                "experiment_type": experiment_type,
                "method": "EdgeInfluence",
                "oracle_only": False,
                "proxy_model": "GCN", "downstream_model": ds_model_name,
                "prune_ratio_target": pruning_cfg.get("beta", 0.2),
                "actual_prune_ratio": graph_stats["prune_ratio"],
                "num_edges_before": graph_stats["num_edges_before"],
                "num_edges_after": graph_stats["num_edges_after"],
                "isolated_nodes": graph_stats.get("isolated_nodes", 0),
                "min_degree": graph_stats.get("min_degree", 0),
                "mean_degree": graph_stats.get("mean_degree", 0),
                "largest_connected_component_ratio": graph_stats.get("largest_connected_component_ratio", 0),
                "edge_homophily_before": homo_before,
                "edge_homophily_after": homo_after,
                "val_acc": ds_result["val_acc"],
                "test_acc": ds_result["test_acc"], "test_f1": ds_result["test_f1"],
                "best_epoch": ds_result["best_epoch"], "runtime": runtime,
                "config_path": args.config, "graph_path": graph_path,
                "noise_type": args.noise_type if args.noisy else "",
                "noise_ratio": args.noise_ratio if args.noisy else 0,
                "num_injected_edges": noise_result["num_injected_edges"] if args.noisy else 0,
                "bad_edge_precision": det["bad_edge_precision"],
                "bad_edge_recall": det["bad_edge_recall"],
                "bad_edge_f1": det["bad_edge_f1"],
                "clean_edge_mistakenly_removed_ratio": det["clean_edge_mistakenly_removed_ratio"],
            }, f"{args.output_dir}/results.csv")

    logger.info("EdgeInfluence pipeline complete!")


if __name__ == "__main__":
    main()
