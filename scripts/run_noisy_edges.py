"""
Noisy-edge robustness experiments.
Inject task-harmful edges and evaluate GraCA's ability to detect and remove them.

Metrics:
- downstream test_acc
- bad-edge removal precision
- bad-edge removal recall
- bad-edge removal F1
- clean-edge mistakenly removed ratio

Usage:
    python scripts/run_noisy_edges.py --config configs/graca_lite_cora.yaml --seed 0
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
from src.baselines.random_pruning import run_random_pruning, run_degree_aware_random
from src.baselines.similarity_pruning import run_cosine_pruning
from src.baselines.homophily_pruning import run_homophily_pruning
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


def inject_cross_class_noise(edge_index, y, num_nodes, noise_ratio, train_mask, seed=42):
    """Inject random cross-class edges as noise.

    Returns:
        noisy_edge_index: [2, E + E_noise]
        bad_edge_mask: [E + E_noise] True for injected noise edges
        num_added: number of noise edges added
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    E = edge_index.shape[1]
    num_noise = int(E * noise_ratio)

    src = torch.randint(0, num_nodes, (num_noise * 3,), generator=rng)
    dst = torch.randint(0, num_nodes, (num_noise * 3,), generator=rng)

    y_cpu = y.cpu()
    train_mask_cpu = train_mask.cpu()

    both_labeled = train_mask_cpu[src] & train_mask_cpu[dst]
    cross_class = y_cpu[src] != y_cpu[dst]
    valid = both_labeled & cross_class & (src != dst)

    noise_src = src[valid][:num_noise]
    noise_dst = dst[valid][:num_noise]

    actual_added = len(noise_src)
    noisy_edge_index = torch.cat([edge_index.cpu(), torch.stack([noise_src, noise_dst])], dim=1)

    bad_edge_mask = torch.zeros(noisy_edge_index.shape[1], dtype=torch.bool)
    bad_edge_mask[E:] = True

    return noisy_edge_index, bad_edge_mask, actual_added


def evaluate_edge_detection(prune_mask, bad_edge_mask, keep_mask):
    """Evaluate bad-edge detection performance.

    Args:
        prune_mask: [E] True = edges removed by method
        bad_edge_mask: [E] True = injected noise edges
        keep_mask: [E] True = edges kept by method

    Returns:
        dict with precision, recall, f1, clean_mistaken_ratio
    """
    # True positive: correctly removed bad edges
    tp = (prune_mask & bad_edge_mask).sum().item()
    # False positive: mistakenly removed good edges
    fp = (prune_mask & ~bad_edge_mask).sum().item()
    # False negative: bad edges not removed
    fn = (~prune_mask & bad_edge_mask).sum().item()
    # True negative: good edges correctly kept
    tn = (~prune_mask & ~bad_edge_mask).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Clean edges mistakenly removed
    total_clean = (~bad_edge_mask).sum().item()
    clean_mistaken = fp / max(total_clean, 1)

    return {
        "bad_edge_precision": precision,
        "bad_edge_recall": recall,
        "bad_edge_f1": f1,
        "clean_edge_mistakenly_removed_ratio": clean_mistaken,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise_ratio", type=float, default=None,
                        help="Override noise ratio (default: test all)")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_noisy_edges")
    device = get_device(config)
    ds_name = config["dataset"]["name"]

    noise_ratios = [0.05, 0.10, 0.20, 0.30]
    if args.noise_ratio is not None:
        noise_ratios = [args.noise_ratio]

    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    result_dir = "results/noisy_edges/"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = f"{result_dir}/noisy_edge_results.csv"

    for seed in seeds:
        for noise_ratio in noise_ratios:
            logger.info(f"=== {ds_name} seed={seed} noise={noise_ratio} ===")
            set_seed(seed)

            # Load clean data
            data, num_features, num_classes = load_dataset(config)
            data = data.to(device)
            train_mask = data.train_mask.to(device)
            y = data.y.to(device)
            num_nodes = data.num_nodes

            # Inject noise
            noisy_edge_index, bad_edge_mask, num_added = inject_cross_class_noise(
                data.edge_index, y, num_nodes, noise_ratio, train_mask, seed
            )
            logger.info(f"Injected {num_added} cross-class noise edges (ratio={noise_ratio})")

            noisy_data = copy.deepcopy(data)
            noisy_data.edge_index = noisy_edge_index

            # --- 1. Original + Noise ---
            set_seed(seed)
            orig_result = train_downstream(
                model_name="GCN", data=noisy_data, edge_index=noisy_edge_index,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=seed,
            )
            write_result_row({
                "run_id": f"noisy_orig_{ds_name}_n{noise_ratio}_s{seed}",
                "seed": seed, "dataset": ds_name,
                "method": "Original+Noise", "oracle_only": False,
                "proxy_model": "-", "downstream_model": "GCN",
                "actual_prune_ratio": 0.0,
                "num_edges_before": noisy_edge_index.shape[1],
                "num_edges_after": noisy_edge_index.shape[1],
                "isolated_nodes": 0, "min_degree": 0, "mean_degree": 0,
                "val_acc": orig_result["val_acc"],
                "test_acc": orig_result["test_acc"],
                "test_f1": orig_result["test_f1"],
                "best_epoch": orig_result["best_epoch"],
                "runtime": orig_result["runtime"],
                "config_path": args.config, "graph_path": "", "checkpoint_path": "",
            }, csv_path)

            # --- 2. GraCA-lite on noisy graph ---
            set_seed(seed)
            model, teacher, train_log, saved_checkpoints = train_proxy(
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
                deterministic=True,
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

            # Evaluate edge detection
            detection = evaluate_edge_detection(prune_mask, bad_edge_mask.to(prune_mask.device), ~prune_mask)
            logger.info(f"GraCA detection: P={detection['bad_edge_precision']:.3f}, "
                        f"R={detection['bad_edge_recall']:.3f}, F1={detection['bad_edge_f1']:.3f}")

            set_seed(seed)
            graca_result = train_downstream(
                model_name="GCN", data=noisy_data, edge_index=pruned_ei,
                config=config, num_features=num_features, num_classes=num_classes,
                device=device, seed=seed,
            )

            write_result_row({
                "run_id": f"noisy_graca_{ds_name}_n{noise_ratio}_s{seed}",
                "seed": seed, "dataset": ds_name,
                "method": "GraCA-lite+Noise", "oracle_only": False,
                "proxy_model": "GCN", "downstream_model": "GCN",
                "actual_prune_ratio": graph_stats["prune_ratio"],
                "num_edges_before": graph_stats["num_edges_before"],
                "num_edges_after": graph_stats["num_edges_after"],
                "isolated_nodes": graph_stats["isolated_nodes"],
                "min_degree": graph_stats.get("min_degree", 0),
                "mean_degree": graph_stats.get("mean_degree", 0),
                "val_acc": graca_result["val_acc"],
                "test_acc": graca_result["test_acc"],
                "test_f1": graca_result["test_f1"],
                "best_epoch": graca_result["best_epoch"],
                "runtime": graca_result["runtime"],
                "config_path": args.config, "graph_path": "", "checkpoint_path": "",
                "bad_edge_precision": detection["bad_edge_precision"],
                "bad_edge_recall": detection["bad_edge_recall"],
                "bad_edge_f1": detection["bad_edge_f1"],
                "clean_edge_mistakenly_removed_ratio": detection["clean_edge_mistakenly_removed_ratio"],
            }, csv_path)

            # --- 3. Random Pruning (matched ratio) ---
            set_seed(seed)
            random_results, random_stats = run_random_pruning(
                noisy_data, config, num_features, num_classes, device, seed,
                match_graca_ratio=graph_stats["prune_ratio"],
            )

            # Random detection eval
            random_prune_mask = torch.zeros(noisy_edge_index.shape[1], dtype=torch.bool)
            E_before = noisy_edge_index.shape[1]
            E_after = random_stats["num_edges_after"]
            perm = torch.randperm(E_before)[:E_before - E_after]
            random_prune_mask[perm] = True
            random_detection = evaluate_edge_detection(
                random_prune_mask, bad_edge_mask, ~random_prune_mask
            )

            write_result_row({
                "run_id": f"noisy_random_{ds_name}_n{noise_ratio}_s{seed}",
                "seed": seed, "dataset": ds_name,
                "method": "Random+Noise", "oracle_only": False,
                "proxy_model": "-", "downstream_model": "GCN",
                "actual_prune_ratio": random_stats["prune_ratio"],
                "num_edges_before": random_stats["num_edges_before"],
                "num_edges_after": random_stats["num_edges_after"],
                "isolated_nodes": 0, "min_degree": 0, "mean_degree": 0,
                "val_acc": random_results["GCN"]["val_acc"],
                "test_acc": random_results["GCN"]["test_acc"],
                "test_f1": random_results["GCN"]["test_f1"],
                "best_epoch": random_results["GCN"]["best_epoch"],
                "runtime": random_results["GCN"]["runtime"],
                "config_path": args.config, "graph_path": "", "checkpoint_path": "",
                "bad_edge_precision": random_detection["bad_edge_precision"],
                "bad_edge_recall": random_detection["bad_edge_recall"],
                "bad_edge_f1": random_detection["bad_edge_f1"],
                "clean_edge_mistakenly_removed_ratio": random_detection["clean_edge_mistakenly_removed_ratio"],
            }, csv_path)

            logger.info(
                f"noise={noise_ratio}: "
                f"Orig={orig_result['test_acc']:.4f} "
                f"GraCA={graca_result['test_acc']:.4f} "
                f"Random={random_results['GCN']['test_acc']:.4f} "
                f"GraCA_F1={detection['bad_edge_f1']:.3f} "
                f"Random_F1={random_detection['bad_edge_f1']:.3f}"
            )

    logger.info("Noisy-edge experiments complete!")


if __name__ == "__main__":
    main()
