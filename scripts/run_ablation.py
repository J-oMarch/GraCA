"""
Run ablation experiments for GraCA-lite.
"""
import sys
import os
import argparse
import copy
import torch

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
from src.graca.save_graph import save_sanitized_graph
from src.training.train_downstream import train_downstream
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


ABLATION_CONFIGS = {
    "no_ema": {"teacher": {"use_ema": False}},
    "hard_pseudo": {"pseudo": {"hard_pseudo": True, "use_soft_label": False}},
    "no_reliability": {"pseudo": {"no_reliability": True}},
    "harmful_only": {"scoring": {"harmful_only": True}},
    "helpful_only": {"scoring": {"helpful_only": True}},
    "global_threshold": {"pruning": {"use_local_threshold": False}},
    "train_only": {"scoring": {"train_only": True}},
}


def merge_config(base, override):
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key].update(val)
        else:
            result[key] = val
    return result


def run_single_ablation(config, ablation_name, ablation_overrides, device, seed, logger):
    """Run a single ablation variant."""
    config = merge_config(config, ablation_overrides)

    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    ds_name = config["dataset"]["name"]

    # Train proxy
    model, teacher, train_log, _ = train_proxy(
        config, data, num_features, num_classes, device, seed
    )

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    unlabeled_mask = ~train_mask

    # Teacher predictions
    if config.get("teacher", {}).get("use_ema", True) and teacher is not None:
        teacher_probs = teacher.predict(x, edge_index)
    else:
        model.eval()
        with torch.no_grad():
            teacher_probs = torch.softmax(model(x, edge_index), dim=-1)

    pseudo_cfg = config.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.8)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )

    # Handle ablation-specific rho modifications
    if pseudo_cfg.get("no_reliability", False):
        rho_train = torch.ones_like(rho_train)

    # Collect gradients
    grad_result = collect_hidden_gradients(
        model=model, x=x, edge_index=edge_index, y=y,
        teacher_probs=teacher_probs, rho_train=rho_train,
        train_mask=train_mask, unlabeled_mask=unlabeled_mask,
        lambda_s=config.get("scoring", {}).get("lambda_s", 1.0),
    )

    grad = grad_result["grad"]
    num_nodes = x.shape[0]

    # Compute rho_score
    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)
    if pseudo_cfg.get("no_reliability", False):
        rho_score = torch.ones_like(rho_score)

    # Edge scoring
    scoring_cfg = config.get("scoring", {})
    edge_scores = compute_edge_scores(
        grad=grad, edge_index=edge_index, rho_score=rho_score,
        num_nodes=num_nodes, eta=scoring_cfg.get("eta", 1.0), epsilon_rho=eps_rho,
    )

    # Apply ablation modifications to P
    if scoring_cfg.get("harmful_only", False):
        P = edge_scores["R"]  # Only harmful, no helpful offset
    elif scoring_cfg.get("helpful_only", False):
        P = -edge_scores["H"]  # Prune low-helpful edges
    else:
        P = edge_scores["P"]

    if config.get("dataset", {}).get("undirected", True):
        P = average_undirected_scores(edge_index, P)

    # Pruning
    pruning_cfg = config.get("pruning", {})
    if not pruning_cfg.get("use_local_threshold", True):
        lambda_theta = 999.0  # Effectively global: only prune above global mean
    else:
        lambda_theta = pruning_cfg.get("lambda_theta", 0.0)

    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
        beta=pruning_cfg.get("beta", 0.2),
        min_degree=pruning_cfg.get("min_degree", 1),
        lambda_theta=lambda_theta,
        undirected=config.get("dataset", {}).get("undirected", True),
        protect_self_loops=pruning_cfg.get("protect_self_loops", True),
    )

    # Save graph
    graph_dir = f"sanitized_graphs/ablation/"
    graph_path = save_sanitized_graph(
        pruned_edge_index, prune_mask, graph_stats, graph_dir,
        f"{ds_name}_{ablation_name}_seed{seed}"
    )

    # Downstream
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
            "run_id": f"ablation_{ablation_name}_{ds_name}_{ds_model_name}_seed{seed}",
            "seed": seed, "dataset": ds_name,
            "method": f"GraCA-ablation-{ablation_name}",
            "oracle_only": False,
            "proxy_model": config["proxy_model"]["name"],
            "downstream_model": ds_model_name,
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
            "config_path": "",
            "graph_path": graph_path,
            "checkpoint_path": "",
        }, "results/ablation/ablation_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all"] + list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_ablation")
    device = get_device(config)

    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    ablation_names = [args.ablation] if args.ablation != "all" else list(ABLATION_CONFIGS.keys())

    for seed in seeds:
        for abl_name in ablation_names:
            logger.info(f"=== Ablation: {abl_name}, seed={seed} ===")
            set_seed(seed)
            run_single_ablation(
                config, abl_name, ABLATION_CONFIGS[abl_name], device, seed, logger
            )

    logger.info("All ablations complete!")


if __name__ == "__main__":
    main()
