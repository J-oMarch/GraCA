"""
Run GraCA pipeline: ProxyGNN training -> gradient collection -> edge scoring -> pruning.
Supports both GraCA-lite and Full Practical GraCA.
Saves sanitized graph and downstream results.
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
from src.data.load_data import load_dataset
from src.training.train_proxy import train_proxy
from src.graca.gradient_collector import collect_hidden_gradients, collect_multi_checkpoint_gradients
from src.graca.edge_scoring import compute_edge_scores, average_undirected_scores, compute_rho_score
from src.graca.pruning import prune_graph
from src.graca.save_graph import save_sanitized_graph
from src.training.train_downstream import train_downstream
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_graca")

    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    proxy_name = config["proxy_model"]["name"]
    result_dir = config.get("logging", {}).get("result_dir", "results/main/")
    graph_dir = config.get("logging", {}).get("graph_dir", "sanitized_graphs/graca_lite/")
    method = config.get("experiment", {}).get("method", "graca_lite")

    for seed in seeds:
        set_seed(seed)
        logger.info(f"=== {method} on {ds_name}, seed={seed} ===")

        # 1. Load data
        data, num_features, num_classes = load_dataset(config)
        data = data.to(device)
        logger.info(f"Data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, "
                     f"{num_features} features, {num_classes} classes")

        # 2. Train ProxyGNN
        logger.info("Training ProxyGNN...")
        model, teacher, train_log, saved_checkpoints = train_proxy(
            config, data, num_features, num_classes, device, seed
        )

        # 3. Collect gradients
        logger.info("Collecting gradients...")
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        unlabeled_mask = ~train_mask

        teacher_probs = teacher.predict(x, edge_index)
        pseudo_cfg = config.get("pseudo", {})
        tau = pseudo_cfg.get("tau", 0.6)
        alpha = pseudo_cfg.get("alpha", 1.0)
        eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

        from src.graca.pseudo_label import compute_soft_pseudo_labels
        q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
            teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
        )

        scoring_cfg = config.get("scoring", {})
        use_multi_checkpoint = scoring_cfg.get("use_multi_checkpoint", False)

        if use_multi_checkpoint and len(saved_checkpoints) > 1:
            logger.info(f"Using {len(saved_checkpoints)} checkpoints for gradient averaging")
            grad_result = collect_multi_checkpoint_gradients(
                model=model, x=x, edge_index=edge_index, y=y,
                teacher_probs=teacher_probs, rho_train=rho_train,
                train_mask=train_mask, unlabeled_mask=unlabeled_mask,
                lambda_s=scoring_cfg.get("lambda_s", 1.0),
                checkpoints=saved_checkpoints,
            )
        else:
            grad_result = collect_hidden_gradients(
                model=model, x=x, edge_index=edge_index, y=y,
                teacher_probs=teacher_probs, rho_train=rho_train,
                train_mask=train_mask, unlabeled_mask=unlabeled_mask,
                lambda_s=scoring_cfg.get("lambda_s", 1.0),
                collect_layer=scoring_cfg.get("collect_layer", "all"),
            )

        grad = grad_result["grad"]
        num_nodes = x.shape[0]

        # 4. Compute rho_score
        rho_score = compute_rho_score(
            teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
        )

        # 5. Edge scoring
        logger.info("Computing edge scores...")
        edge_scores = compute_edge_scores(
            grad=grad, edge_index=edge_index, rho_score=rho_score,
            num_nodes=num_nodes, eta=scoring_cfg.get("eta", 1.0), epsilon_rho=eps_rho,
        )

        P = edge_scores["P"]
        if config.get("dataset", {}).get("undirected", True):
            P = average_undirected_scores(edge_index, P)

        D = edge_scores["D"]
        logger.info(f"Edge score stats: D mean={D.mean():.4f}, frac_neg={(D < 0).float().mean():.4f}, "
                     f"P mean={P.mean():.4f}, H mean={edge_scores['H'].mean():.4f}, R mean={edge_scores['R'].mean():.4f}")

        # 6. Prune with optional bridge protection
        pruning_cfg = config.get("pruning", {})
        pruned_edge_index, prune_mask, graph_stats = prune_graph(
            edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
            beta=pruning_cfg.get("beta", 0.2),
            min_degree=pruning_cfg.get("min_degree", 1),
            lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
            undirected=config.get("dataset", {}).get("undirected", True),
            protect_self_loops=pruning_cfg.get("protect_self_loops", True),
            protect_bridges=pruning_cfg.get("protect_bridges", False),
        )

        logger.info(f"Pruning: {graph_stats['num_edges_before']} -> {graph_stats['num_edges_after']} edges "
                     f"(ratio={graph_stats['prune_ratio']:.3f}), isolated={graph_stats['isolated_nodes']}")

        # 7. Save sanitized graph
        run_id = config.get("experiment", {}).get("run_id", f"{method}_{ds_name}_seed{seed}")
        graph_path = save_sanitized_graph(
            pruned_edge_index, prune_mask, graph_stats, graph_dir, f"{ds_name}_seed{seed}"
        )

        # 8. Downstream retraining
        logger.info("Training downstream models...")
        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])

        for ds_model_name in downstream_names:
            set_seed(seed)
            ds_result = train_downstream(
                model_name=ds_model_name, data=data,
                edge_index=pruned_edge_index, config=config,
                num_features=num_features, num_classes=num_classes,
                device=device, seed=seed,
            )

            write_result_row({
                "run_id": run_id, "seed": seed, "dataset": ds_name,
                "method": method, "oracle_only": False,
                "proxy_model": proxy_name, "downstream_model": ds_model_name,
                "prune_ratio": graph_stats["prune_ratio"],
                "num_edges_before": graph_stats["num_edges_before"],
                "num_edges_after": graph_stats["num_edges_after"],
                "isolated_nodes": graph_stats["isolated_nodes"],
                "min_degree": graph_stats.get("min_degree", 0),
                "mean_degree": graph_stats.get("mean_degree", 0),
                "largest_connected_component_ratio": graph_stats.get("largest_connected_component_ratio", 0),
                "val_acc": ds_result["val_acc"],
                "test_acc": ds_result["test_acc"],
                "test_f1": ds_result["test_f1"],
                "best_epoch": ds_result["best_epoch"],
                "runtime": ds_result["runtime"],
                "config_path": args.config, "graph_path": graph_path, "checkpoint_path": "",
            }, f"{result_dir}/results.csv")

    logger.info(f"{method} pipeline complete!")


if __name__ == "__main__":
    main()
