"""
Robustness experiments: inject noisy edges and test GraCA-lite's ability to remove them.
Usage:
    python scripts/run_robustness.py --config configs/graca_lite_cora.yaml --seed 0
"""
import sys
import os
import argparse
import torch
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
from src.baselines.random_pruning import run_random_pruning
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


def add_noisy_edges(edge_index, y, num_nodes, noise_ratio, train_mask, seed=42):
    """Add random cross-class edges as noise.

    Args:
        edge_index: [2, E] original edges
        y: [N] node labels
        num_nodes: total nodes
        noise_ratio: fraction of original edges to add as noise
        train_mask: boolean mask for labeled nodes
        seed: random seed

    Returns:
        noisy_edge_index: [2, E + E_noise]
        num_added: number of noise edges added
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    edge_index = edge_index.cpu()
    E = edge_index.shape[1]
    num_noise = int(E * noise_ratio)

    # Generate random edges
    src = torch.randint(0, num_nodes, (num_noise,), generator=rng)
    dst = torch.randint(0, num_nodes, (num_noise,), generator=rng)

    # Move to CPU for filtering
    y_cpu = y.cpu()
    train_mask_cpu = train_mask.cpu()

    # Filter: keep only cross-class edges (where both endpoints have labels)
    both_labeled = train_mask_cpu[src] & train_mask_cpu[dst]
    cross_class = y_cpu[src] != y_cpu[dst]
    valid = both_labeled & cross_class & (src != dst)

    noise_src = src[valid]
    noise_dst = dst[valid]

    # Combine with original edges
    noisy_edge_index = torch.cat([edge_index, torch.stack([noise_src, noise_dst])], dim=1)

    return noisy_edge_index, len(noise_src)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_robustness")
    device = get_device(config)
    ds_name = config["dataset"]["name"]

    noise_ratios = [0.05, 0.10, 0.20, 0.30]
    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    for seed in seeds:
        set_seed(seed)
        data, num_features, num_classes = load_dataset(config)
        data = data.to(device)

        train_mask = data.train_mask.to(device)
        y = data.y.to(device)
        num_nodes = data.num_nodes

        for noise_ratio in noise_ratios:
            logger.info(f"=== {ds_name} seed={seed} noise={noise_ratio} ===")

            # Add noisy edges
            noisy_edge_index, num_added = add_noisy_edges(
                data.edge_index, y, num_nodes, noise_ratio, train_mask, seed
            )
            logger.info(f"Added {num_added} noise edges (ratio={noise_ratio})")

            # Create noisy data object
            noisy_data = copy.deepcopy(data)
            noisy_data.edge_index = noisy_edge_index

            # --- Baseline: Original on noisy graph ---
            set_seed(seed)
            orig_result = train_downstream(
                model_name="GCN", data=noisy_data, edge_index=noisy_edge_index,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=seed,
            )
            write_result_row({
                "run_id": f"robust_orig_{ds_name}_noise{noise_ratio}_seed{seed}",
                "seed": seed, "dataset": ds_name,
                "method": f"Original+Noise{noise_ratio}",
                "oracle_only": False, "proxy_model": "-",
                "downstream_model": "GCN",
                "prune_ratio": 0,
                "num_edges_before": noisy_edge_index.shape[1],
                "num_edges_after": noisy_edge_index.shape[1],
                "isolated_nodes": 0,
                "val_acc": orig_result["val_acc"],
                "test_acc": orig_result["test_acc"],
                "test_f1": orig_result["test_f1"],
                "best_epoch": orig_result["best_epoch"],
                "runtime": orig_result["runtime"],
                "config_path": args.config, "graph_path": "", "checkpoint_path": "",
            }, "results/robustness/robustness_results.csv")

            # --- GraCA-lite on noisy graph ---
            set_seed(seed)
            model, teacher, _, _ = train_proxy(
                config, noisy_data, num_features, num_classes, device, seed
            )

            x = noisy_data.x.to(device)
            ei = noisy_data.edge_index.to(device)
            unlabeled_mask = ~train_mask

            teacher_probs = teacher.predict(x, ei)
            tau = config.get("pseudo", {}).get("tau", 0.6)
            alpha = config.get("pseudo", {}).get("alpha", 1.0)
            eps_rho = config.get("pseudo", {}).get("epsilon_rho", 0.05)

            q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
                teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
            )

            grad_result = collect_hidden_gradients(
                model=model, x=x, edge_index=ei, y=y,
                teacher_probs=teacher_probs, rho_train=rho_train,
                train_mask=train_mask, unlabeled_mask=unlabeled_mask,
                lambda_s=config.get("scoring", {}).get("lambda_s", 1.0),
            )

            grad = grad_result["grad"]
            rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

            edge_scores = compute_edge_scores(
                grad=grad, edge_index=ei, rho_score=rho_score,
                num_nodes=num_nodes, eta=config.get("scoring", {}).get("eta", 1.0),
                epsilon_rho=eps_rho,
            )

            P = edge_scores["P"]
            if config.get("dataset", {}).get("undirected", True):
                P = average_undirected_scores(ei, P)

            pruning_cfg = config.get("pruning", {})
            pruned_ei, prune_mask, graph_stats = prune_graph(
                edge_index=ei, risk_score=P, num_nodes=num_nodes,
                beta=pruning_cfg.get("beta", 0.2),
                min_degree=pruning_cfg.get("min_degree", 1),
                lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
                undirected=config.get("dataset", {}).get("undirected", True),
                protect_self_loops=pruning_cfg.get("protect_self_loops", True),
            )

            set_seed(seed)
            graca_result = train_downstream(
                model_name="GCN", data=noisy_data, edge_index=pruned_ei,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=seed,
            )

            write_result_row({
                "run_id": f"robust_graca_{ds_name}_noise{noise_ratio}_seed{seed}",
                "seed": seed, "dataset": ds_name,
                "method": f"GraCA-lite+Noise{noise_ratio}",
                "oracle_only": False, "proxy_model": "GCN",
                "downstream_model": "GCN",
                "prune_ratio": graph_stats["prune_ratio"],
                "num_edges_before": graph_stats["num_edges_before"],
                "num_edges_after": graph_stats["num_edges_after"],
                "isolated_nodes": graph_stats["isolated_nodes"],
                "val_acc": graca_result["val_acc"],
                "test_acc": graca_result["test_acc"],
                "test_f1": graca_result["test_f1"],
                "best_epoch": graca_result["best_epoch"],
                "runtime": graca_result["runtime"],
                "config_path": args.config, "graph_path": "", "checkpoint_path": "",
            }, "results/robustness/robustness_results.csv")

            # --- Random pruning on noisy graph ---
            set_seed(seed)
            random_results, random_stats = run_random_pruning(
                noisy_data, config, num_features, num_classes, device, seed,
                prune_ratio=graph_stats["prune_ratio"],
            )

            write_result_row({
                "run_id": f"robust_random_{ds_name}_noise{noise_ratio}_seed{seed}",
                "seed": seed, "dataset": ds_name,
                "method": f"Random+Noise{noise_ratio}",
                "oracle_only": False, "proxy_model": "-",
                "downstream_model": "GCN",
                "prune_ratio": random_stats["prune_ratio"],
                "num_edges_before": random_stats["num_edges_before"],
                "num_edges_after": random_stats["num_edges_after"],
                "isolated_nodes": 0,
                "val_acc": random_results["GCN"]["val_acc"],
                "test_acc": random_results["GCN"]["test_acc"],
                "test_f1": random_results["GCN"]["test_f1"],
                "best_epoch": random_results["GCN"]["best_epoch"],
                "runtime": random_results["GCN"]["runtime"],
                "config_path": args.config, "graph_path": "", "checkpoint_path": "",
            }, "results/robustness/robustness_results.csv")

            logger.info(
                f"noise={noise_ratio}: "
                f"Orig={orig_result['test_acc']:.4f} "
                f"GraCA={graca_result['test_acc']:.4f} "
                f"Random={random_results['GCN']['test_acc']:.4f}"
            )

    logger.info("Robustness experiments complete!")


if __name__ == "__main__":
    main()
