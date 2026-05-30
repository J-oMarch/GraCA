"""
Unified noisy-edge experiment runner.

Runs GraCA and baselines on noisy graphs, evaluating:
1. Downstream accuracy on noisy graph
2. Bad-edge detection (precision/recall/F1)
3. Structural graph metrics before/after pruning

Usage:
    python scripts/run_noisy_edge_experiment.py --config configs/graca_lite_cora.yaml --seed 0 \
        --noise_type low_feature_similarity --noise_ratio 0.10
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
from src.graca.gradient_collector import collect_hidden_gradients
from src.graca.edge_scoring import compute_edge_scores, average_undirected_scores, compute_rho_score
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.pruning import prune_graph, compute_graph_stats
from src.training.train_downstream import train_downstream
from src.baselines.random_pruning import run_random_pruning, run_degree_aware_random
from src.baselines.homophily_pruning import run_homophily_pruning
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection, save_noise_metadata
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", type=str, default="low_feature_similarity",
                        choices=["cross_class_train_safe", "cross_class_oracle",
                                 "low_feature_similarity", "random_inter_community"])
    parser.add_argument("--noise_ratio", type=float, default=0.10)
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "graca", "random", "degree_random",
                                 "homophily", "original"])
    parser.add_argument("--output_dir", type=str, default="results_clean/noisy/")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_noisy_edge")
    set_seed(args.seed)

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    undirected = config.get("dataset", {}).get("undirected", True)
    experiment_type = "noisy_edge"

    # Load data
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    num_nodes = data.num_nodes
    E_orig = data.edge_index.shape[1]
    y_for_homo = data.y.cpu()
    homo_before_clean = compute_edge_homophily(data.edge_index.cpu(), y_for_homo)

    # Inject noise
    logger.info(f"Injecting {args.noise_type} noise at {args.noise_ratio}...")
    noise_result = inject_noise(
        edge_index=data.edge_index.cpu(),
        num_nodes=num_nodes,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        x=data.x.cpu(),
        y=data.y.cpu(),
        train_mask=data.train_mask.cpu(),
        seed=args.seed,
    )
    noisy_edge_index = noise_result["noisy_edge_index"].to(device)
    bad_edge_mask = noise_result["bad_edge_mask"].to(device)
    E_noisy = noisy_edge_index.shape[1]

    logger.info(f"Original: {E_orig} edges, Noisy: {E_noisy} edges, "
                f"Injected pairs: {noise_result['num_injected_edges']}")

    homo_before_noisy = compute_edge_homophily(noisy_edge_index.cpu(), y_for_homo)

    # Save noise metadata
    noise_meta = noise_result["metadata"]
    noise_meta["dataset"] = ds_name
    save_noise_metadata(noise_meta, args.output_dir, f"noise_meta_{ds_name}_{args.noise_type}_{args.noise_ratio}_{args.seed}")

    def _write_result(method_name, oracle_only, ds_model_name, val_acc, test_acc, test_f1,
                      best_epoch, runtime, prune_mask, graph_stats, prune_target=0):
        """Helper to write a single result row."""
        # Evaluate bad-edge detection
        det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)

        # Compute homophily after pruning
        keep_mask = ~prune_mask.cpu()
        pruned_ei = noisy_edge_index.cpu()[:, keep_mask]
        homo_after = compute_edge_homophily(pruned_ei, y_for_homo)

        run_id = f"{method_name}_{ds_name}_{args.noise_type}_{args.noise_ratio}_{ds_model_name}_seed{args.seed}"
        write_result_row({
            "run_id": run_id, "seed": args.seed, "dataset": ds_name,
            "experiment_type": experiment_type,
            "method": method_name, "oracle_only": oracle_only,
            "proxy_model": "-", "downstream_model": ds_model_name,
            "prune_ratio_target": prune_target,
            "actual_prune_ratio": graph_stats.get("prune_ratio", 0),
            "num_edges_before": graph_stats.get("num_edges_before", E_noisy),
            "num_edges_after": graph_stats.get("num_edges_after", E_noisy),
            "isolated_nodes": graph_stats.get("isolated_nodes", 0),
            "min_degree": graph_stats.get("min_degree", 0),
            "mean_degree": graph_stats.get("mean_degree", 0),
            "largest_connected_component_ratio": graph_stats.get("largest_connected_component_ratio", 1.0),
            "edge_homophily_before": homo_before_noisy,
            "edge_homophily_after": homo_after,
            "val_acc": val_acc, "test_acc": test_acc, "test_f1": test_f1,
            "best_epoch": best_epoch, "runtime": runtime,
            "config_path": args.config, "graph_path": "",
            "noise_type": args.noise_type, "noise_ratio": args.noise_ratio,
            "num_injected_edges": noise_result["num_injected_edges"],
            "bad_edge_precision": det["bad_edge_precision"],
            "bad_edge_recall": det["bad_edge_recall"],
            "bad_edge_f1": det["bad_edge_f1"],
            "clean_edge_mistakenly_removed_ratio": det["clean_edge_mistakenly_removed_ratio"],
        }, f"{args.output_dir}/noisy_edge_results.csv")

    # ============ Original + Noise (no pruning) ============
    if args.method in ("all", "original"):
        logger.info("Running Original+Noise baseline...")
        # Save original edge_index, use noisy for downstream
        orig_edge_index_backup = data.edge_index.clone()
        data.edge_index = noisy_edge_index

        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
        for model_name in downstream_names:
            set_seed(args.seed)
            t0 = time.time()
            res = train_downstream(
                model_name=model_name, data=data, edge_index=noisy_edge_index,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=args.seed,
            )
            runtime = time.time() - t0

            # Original doesn't prune, so all edges kept
            full_prune_mask = torch.zeros(E_noisy, dtype=torch.bool, device="cpu")
            stats = {"prune_ratio": 0, "num_edges_before": E_noisy, "num_edges_after": E_noisy,
                     "isolated_nodes": 0, "min_degree": 0, "mean_degree": 0,
                     "largest_connected_component_ratio": 1.0}

            _write_result("Original+Noise", False, model_name,
                          res["val_acc"], res["test_acc"], res["test_f1"],
                          res["best_epoch"], runtime, full_prune_mask, stats)

        data.edge_index = orig_edge_index_backup

    # ============ GraCA-lite on noisy graph ============
    if args.method in ("all", "graca"):
        logger.info("Running GraCA-lite on noisy graph...")

        # Train proxy on NOISY graph
        set_seed(args.seed)
        data_for_proxy = data.clone()
        data_for_proxy.edge_index = noisy_edge_index

        model, teacher, train_log, saved_checkpoints = train_proxy(
            config, data_for_proxy, num_features, num_classes, device, args.seed
        )

        # Collect gradients on noisy graph
        x = data.x.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        unlabeled_mask = ~train_mask

        teacher_probs = teacher.predict(x, noisy_edge_index)
        pseudo_cfg = config.get("pseudo", {})
        tau = pseudo_cfg.get("tau", 0.6)
        alpha = pseudo_cfg.get("alpha", 1.0)
        eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

        q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
            teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
        )

        scoring_cfg = config.get("scoring", {})
        deterministic = scoring_cfg.get("deterministic", True)

        grad_result = collect_hidden_gradients(
            model=model, x=x, edge_index=noisy_edge_index, y=y,
            teacher_probs=teacher_probs, rho_train=rho_train,
            train_mask=train_mask, unlabeled_mask=unlabeled_mask,
            lambda_s=scoring_cfg.get("lambda_s", 1.0),
            collect_layer=scoring_cfg.get("collect_layer", "all"),
            deterministic=deterministic,
        )

        grad = grad_result["grad"]
        rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

        edge_scores = compute_edge_scores(
            grad=grad, edge_index=noisy_edge_index, rho_score=rho_score,
            num_nodes=num_nodes, eta=scoring_cfg.get("eta", 1.0), epsilon_rho=eps_rho,
        )

        P = edge_scores["P"]
        if undirected:
            P = average_undirected_scores(noisy_edge_index, P)

        pruning_cfg = config.get("pruning", {})
        pruned_edge_index, prune_mask, graph_stats = prune_graph(
            edge_index=noisy_edge_index, risk_score=P, num_nodes=num_nodes,
            beta=pruning_cfg.get("beta", 0.2),
            min_degree=pruning_cfg.get("min_degree", 1),
            lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
            undirected=undirected,
            protect_self_loops=pruning_cfg.get("protect_self_loops", True),
        )

        det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)
        logger.info(f"GraCA bad-edge detection: P={det['bad_edge_precision']:.4f} "
                     f"R={det['bad_edge_recall']:.4f} F1={det['bad_edge_f1']:.4f}")

        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
        for model_name in downstream_names:
            set_seed(args.seed)
            t0 = time.time()
            res = train_downstream(
                model_name=model_name, data=data, edge_index=pruned_edge_index,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=args.seed,
            )
            runtime = time.time() - t0

            _write_result("GraCA-lite", False, model_name,
                          res["val_acc"], res["test_acc"], res["test_f1"],
                          res["best_epoch"], runtime, prune_mask, graph_stats,
                          pruning_cfg.get("beta", 0.2))

    # ============ Random-Matched pruning ============
    if args.method in ("all", "random"):
        logger.info("Running Random-Matched pruning...")

        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
        for model_name in downstream_names:
            set_seed(args.seed)
            t0 = time.time()

            # Run random pruning on noisy graph
            edge_index_cpu = noisy_edge_index.cpu()
            prune_ratio = config.get("pruning", {}).get("beta", 0.2)
            E = edge_index_cpu.shape[1]

            # Undirected random pruning
            if undirected:
                edge_key_to_indices = defaultdict(list)
                for i in range(E):
                    u, v = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
                    key = (min(u, v), max(u, v))
                    edge_key_to_indices[key].append(i)

                pair_keys = list(edge_key_to_indices.keys())
                num_remove_pairs = int(len(pair_keys) * prune_ratio)
                perm = torch.randperm(len(pair_keys))
                removed_keys = set(pair_keys[i] for i in perm[:num_remove_pairs])

                prune_mask = torch.zeros(E, dtype=torch.bool)
                for key in removed_keys:
                    for idx in edge_key_to_indices[key]:
                        prune_mask[idx] = True

                self_loop_mask = edge_index_cpu[0] == edge_index_cpu[1]
                prune_mask = prune_mask & ~self_loop_mask
            else:
                num_remove = int(E * prune_ratio)
                perm = torch.randperm(E)
                prune_mask = torch.zeros(E, dtype=torch.bool)
                prune_mask[perm[:num_remove]] = True

            keep_mask = ~prune_mask
            pruned_ei = edge_index_cpu[:, keep_mask].to(device)
            graph_stats = compute_graph_stats(pruned_ei, num_nodes, E)

            res = train_downstream(
                model_name=model_name, data=data, edge_index=pruned_ei,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=args.seed,
            )
            runtime = time.time() - t0

            _write_result("Random-Matched", False, model_name,
                          res["val_acc"], res["test_acc"], res["test_f1"],
                          res["best_epoch"], runtime, prune_mask, graph_stats,
                          prune_ratio)

    # ============ Homophily-TrainOnly ============
    if args.method in ("all", "homophily"):
        logger.info("Running Homophily-TrainOnly pruning...")

        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
        for model_name in downstream_names:
            set_seed(args.seed)
            t0 = time.time()

            # Homophily pruning on noisy graph (train labels only)
            edge_index_cpu = noisy_edge_index.cpu()
            y_cpu = data.y.cpu()
            train_mask_cpu = data.train_mask.cpu()
            E = edge_index_cpu.shape[1]
            src = edge_index_cpu[0]
            dst = edge_index_cpu[1]

            both_labeled = train_mask_cpu[src] & train_mask_cpu[dst]
            same_class = y_cpu[src] == y_cpu[dst]
            hetero_candidates = both_labeled & ~same_class

            candidate_indices = torch.where(hetero_candidates)[0]
            prune_ratio = config.get("pruning", {}).get("beta", 0.2)

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
                num_remove_pairs = min(int(E * prune_ratio / 2), len(candidate_pairs))
                perm = torch.randperm(len(candidate_pairs))[:num_remove_pairs]
                removed_keys = set(candidate_pairs[i] for i in perm.tolist())

                prune_mask = torch.zeros(E, dtype=torch.bool)
                for key in removed_keys:
                    for idx in edge_key_to_indices[key]:
                        prune_mask[idx] = True

                self_loop_mask = src == dst
                prune_mask = prune_mask & ~self_loop_mask
            else:
                num_remove = min(int(E * prune_ratio), len(candidate_indices))
                prune_mask = torch.zeros(E, dtype=torch.bool)
                if num_remove > 0:
                    perm = torch.randperm(len(candidate_indices))[:num_remove]
                    prune_mask[candidate_indices[perm]] = True

            keep_mask = ~prune_mask
            pruned_ei = edge_index_cpu[:, keep_mask].to(device)
            graph_stats = compute_graph_stats(pruned_ei, num_nodes, E)

            res = train_downstream(
                model_name=model_name, data=data, edge_index=pruned_ei,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=args.seed,
            )
            runtime = time.time() - t0

            _write_result("Homophily-TrainOnly", False, model_name,
                          res["val_acc"], res["test_acc"], res["test_f1"],
                          res["best_epoch"], runtime, prune_mask, graph_stats,
                          prune_ratio)

    logger.info("Noisy-edge experiment complete!")


if __name__ == "__main__":
    main()
