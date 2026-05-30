"""
Ablation experiments on noisy-edge graphs.

Runs 15 ablation variants on Cora/CiteSeer/PubMed noisy-edge 10%.
Reports test_acc, bad_edge_precision/recall/f1, actual_prune_ratio.

Usage:
    python scripts/run_ablation_noisy.py --config configs/graca_lite_cora.yaml --seed 0 \
        --noise_type low_feature_similarity --noise_ratio 0.10
"""
import sys
import os
import argparse
import torch
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.training.train_proxy import train_proxy
from src.graca.gradient_collector import collect_hidden_gradients
from src.graca.edge_scoring import compute_edge_scores, average_undirected_scores, compute_rho_score
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.pruning import prune_graph
from src.training.train_downstream import train_downstream
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


ABLATION_VARIANTS = [
    "full_GraCA-lite",
    "no_gradient_direction",
    "direction_only",
    "harmful_only",
    "helpful_only",
    "no_relative_strength",
    "no_uncertainty",
    "no_ema",
    "hard_pseudo",
    "global_threshold",
    "no_local_budget",
    "first_layer",
    "last_layer",
    "all_layers",
    "deterministic_off",
]


def modify_config_for_ablation(config, variant):
    """Modify config for a specific ablation variant."""
    cfg = copy.deepcopy(config)

    if variant == "full_GraCA-lite":
        pass  # default config
    elif variant == "no_gradient_direction":
        cfg["_ablation_no_dir"] = True
    elif variant == "direction_only":
        cfg["_ablation_dir_only"] = True
    elif variant == "harmful_only":
        cfg["_ablation_harmful_only"] = True
    elif variant == "helpful_only":
        cfg["_ablation_helpful_only"] = True
    elif variant == "no_relative_strength":
        cfg["_ablation_no_M"] = True
    elif variant == "no_uncertainty":
        cfg["_ablation_no_rho"] = True
    elif variant == "no_ema":
        cfg.setdefault("teacher", {})["use_ema"] = False
    elif variant == "hard_pseudo":
        cfg.setdefault("pseudo", {})["hard"] = True
    elif variant == "global_threshold":
        cfg.setdefault("pruning", {})["lambda_theta"] = 0.0
        cfg["_ablation_global"] = True
    elif variant == "no_local_budget":
        cfg["_ablation_no_local"] = True
    elif variant == "first_layer":
        cfg.setdefault("scoring", {})["collect_layer"] = "first"
    elif variant == "last_layer":
        cfg.setdefault("scoring", {})["collect_layer"] = "last"
    elif variant == "all_layers":
        cfg.setdefault("scoring", {})["collect_layer"] = "all"
    elif variant == "deterministic_off":
        cfg.setdefault("scoring", {})["deterministic"] = False

    return cfg


def apply_ablation_to_scores(edge_scores, config):
    """Modify edge scores based on ablation flags."""
    P = edge_scores["P"]

    if config.get("_ablation_no_dir"):
        # Replace D contribution with zero: P = R (no direction penalty)
        P = edge_scores["R"]
    elif config.get("_ablation_dir_only"):
        # P = max(-D, 0) only
        P = torch.clamp(-edge_scores["D"], min=0)
    elif config.get("_ablation_harmful_only"):
        # P = R only (harmful score)
        P = edge_scores["R"]
    elif config.get("_ablation_helpful_only"):
        # P = -H only (helpful score, inverted)
        P = -edge_scores["H"]
    elif config.get("_ablation_no_M"):
        # P without relative strength: use D and H only
        P = torch.clamp(-edge_scores["D"], min=0) - edge_scores["H"]

    return P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", type=str, default="low_feature_similarity")
    parser.add_argument("--noise_ratio", type=float, default=0.10)
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all"] + ABLATION_VARIANTS)
    parser.add_argument("--output_dir", type=str, default="results_clean/ablation/")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("ablation_noisy")
    set_seed(args.seed)

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    undirected = config.get("dataset", {}).get("undirected", True)

    # Load data and inject noise
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    num_nodes = data.num_nodes

    logger.info(f"Injecting {args.noise_type} noise at {args.noise_ratio}...")
    noise_result = inject_noise(
        edge_index=data.edge_index.cpu(),
        num_nodes=num_nodes,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        x=data.x.cpu(), y=data.y.cpu(),
        train_mask=data.train_mask.cpu(),
        seed=args.seed,
    )
    noisy_edge_index = noise_result["noisy_edge_index"].to(device)
    bad_edge_mask = noise_result["bad_edge_mask"]

    variants = ABLATION_VARIANTS if args.variant == "all" else [args.variant]

    for variant in variants:
        logger.info(f"\n=== Ablation: {variant} ===")
        set_seed(args.seed)

        cfg = modify_config_for_ablation(config, variant)

        # Train proxy
        data_for_proxy = data.clone()
        data_for_proxy.edge_index = noisy_edge_index

        use_ema = cfg.get("teacher", {}).get("use_ema", True)
        if variant == "no_ema":
            use_ema = False

        model, teacher, train_log, saved_checkpoints = train_proxy(
            cfg, data_for_proxy, num_features, num_classes, device, args.seed
        )

        # Collect gradients
        x = data.x.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        unlabeled_mask = ~train_mask

        teacher_probs = teacher.predict(x, noisy_edge_index)
        pseudo_cfg = cfg.get("pseudo", {})
        tau = pseudo_cfg.get("tau", 0.6)
        alpha = pseudo_cfg.get("alpha", 1.0)
        eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

        q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
            teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
        )

        scoring_cfg = cfg.get("scoring", {})
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

        # Apply ablation modifications to scores
        P = apply_ablation_to_scores(edge_scores, cfg)

        if undirected:
            P = average_undirected_scores(noisy_edge_index, P)

        # Prune
        pruning_cfg = cfg.get("pruning", {})
        # For no_local_budget, use global threshold (no per-node budget)
        if variant == "no_local_budget":
            # Global: prune all edges above median P
            threshold = P.median()
            prune_mask = P > threshold
            keep_mask = ~prune_mask
            pruned_edge_index = noisy_edge_index[:, keep_mask]
            from src.graca.pruning import compute_graph_stats
            graph_stats = compute_graph_stats(pruned_edge_index, num_nodes, noisy_edge_index.shape[1])
        else:
            pruned_edge_index, prune_mask, graph_stats = prune_graph(
                edge_index=noisy_edge_index, risk_score=P, num_nodes=num_nodes,
                beta=pruning_cfg.get("beta", 0.2),
                min_degree=pruning_cfg.get("min_degree", 1),
                lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
                undirected=undirected,
                protect_self_loops=pruning_cfg.get("protect_self_loops", True),
            )

        # Evaluate bad-edge detection
        det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)

        logger.info(f"Prune ratio: {graph_stats['prune_ratio']:.3f}")
        logger.info(f"Bad-edge: P={det['bad_edge_precision']:.4f} R={det['bad_edge_recall']:.4f} F1={det['bad_edge_f1']:.4f}")

        # Downstream evaluation
        downstream_names = cfg.get("downstream_model", {}).get("names", ["GCN"])
        for model_name in downstream_names:
            set_seed(args.seed)
            t0 = time.time()
            res = train_downstream(
                model_name=model_name, data=data, edge_index=pruned_edge_index,
                config=cfg, num_features=num_features, num_classes=num_classes,
                device=device, seed=args.seed,
            )
            runtime = time.time() - t0

            write_result_row({
                "run_id": f"ablation_{variant}_{ds_name}_{args.noise_type}_{model_name}_seed{args.seed}",
                "seed": args.seed, "dataset": ds_name,
                "experiment_type": "ablation",
                "method": variant, "oracle_only": False,
                "proxy_model": "GCN", "downstream_model": model_name,
                "prune_ratio_target": pruning_cfg.get("beta", 0.2),
                "actual_prune_ratio": graph_stats["prune_ratio"],
                "num_edges_before": graph_stats["num_edges_before"],
                "num_edges_after": graph_stats["num_edges_after"],
                "isolated_nodes": graph_stats.get("isolated_nodes", 0),
                "min_degree": graph_stats.get("min_degree", 0),
                "mean_degree": graph_stats.get("mean_degree", 0),
                "largest_connected_component_ratio": graph_stats.get("largest_connected_component_ratio", 0),
                "edge_homophily_before": 0, "edge_homophily_after": 0,
                "val_acc": res["val_acc"], "test_acc": res["test_acc"],
                "test_f1": res["test_f1"], "best_epoch": res["best_epoch"],
                "runtime": runtime,
                "config_path": args.config, "graph_path": "",
                "noise_type": args.noise_type, "noise_ratio": args.noise_ratio,
                "num_injected_edges": noise_result["num_injected_edges"],
                "bad_edge_precision": det["bad_edge_precision"],
                "bad_edge_recall": det["bad_edge_recall"],
                "bad_edge_f1": det["bad_edge_f1"],
                "clean_edge_mistakenly_removed_ratio": det["clean_edge_mistakenly_removed_ratio"],
            }, f"{args.output_dir}/ablation_results.csv")

    logger.info("Ablation experiments complete!")


if __name__ == "__main__":
    main()
