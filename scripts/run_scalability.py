"""
Scalability experiments: measure runtime, memory, edge count, pruning time.
Usage:
    python scripts/run_scalability.py --seed 0
"""
import sys
import os
import argparse
import torch
import time
import gc

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
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


def measure_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def run_scalability(config_path: str, seed: int, device):
    """Run scalability measurement for a single config."""
    config = load_config(config_path)
    logger = get_logger("scalability")
    ds_name = config["dataset"]["name"]

    set_seed(seed)
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --- GraCA-lite pipeline timing ---
    t0 = time.time()

    # Train proxy
    model, teacher, train_log, saved_checkpoints = train_proxy(
        config, data, num_features, num_classes, device, seed
    )
    t_proxy = time.time() - t0

    # Gradient collection
    t_grad_start = time.time()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    unlabeled_mask = ~train_mask

    teacher_probs = teacher.predict(x, edge_index)
    tau = config.get("pseudo", {}).get("tau", 0.6)
    alpha = config.get("pseudo", {}).get("alpha", 1.0)
    eps_rho = config.get("pseudo", {}).get("epsilon_rho", 0.05)

    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )

    grad_result = collect_hidden_gradients(
        model=model, x=x, edge_index=edge_index, y=y,
        teacher_probs=teacher_probs, rho_train=rho_train,
        train_mask=train_mask, unlabeled_mask=unlabeled_mask,
        lambda_s=config.get("scoring", {}).get("lambda_s", 1.0),
    )
    t_grad = time.time() - t_grad_start

    # Edge scoring
    t_score_start = time.time()
    grad = grad_result["grad"]
    num_nodes = x.shape[0]
    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)
    edge_scores = compute_edge_scores(
        grad=grad, edge_index=edge_index, rho_score=rho_score,
        num_nodes=num_nodes, eta=config.get("scoring", {}).get("eta", 1.0), epsilon_rho=eps_rho,
    )
    P = edge_scores["P"]
    if config.get("dataset", {}).get("undirected", True):
        P = average_undirected_scores(edge_index, P)
    t_score = time.time() - t_score_start

    # Pruning
    t_prune_start = time.time()
    pruning_cfg = config.get("pruning", {})
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
        beta=pruning_cfg.get("beta", 0.2),
        min_degree=pruning_cfg.get("min_degree", 1),
        lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
        undirected=config.get("dataset", {}).get("undirected", True),
        protect_self_loops=pruning_cfg.get("protect_self_loops", True),
    )
    t_prune = time.time() - t_prune_start

    # Downstream
    t_down_start = time.time()
    set_seed(seed)
    ds_result = train_downstream(
        model_name="GCN", data=data, edge_index=pruned_edge_index,
        config=config, num_features=num_features, num_classes=num_classes,
        device=device, seed=seed,
    )
    t_down = time.time() - t_down_start

    t_total = time.time() - t0
    peak_mem = measure_memory()

    result = {
        "dataset": ds_name,
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "num_features": num_features,
        "num_classes": num_classes,
        "prune_ratio": graph_stats["prune_ratio"],
        "edges_after": graph_stats["num_edges_after"],
        "t_proxy": t_proxy,
        "t_gradient": t_grad,
        "t_scoring": t_score,
        "t_pruning": t_prune,
        "t_downstream": t_down,
        "t_total": t_total,
        "peak_memory_mb": peak_mem,
        "test_acc": ds_result["test_acc"],
    }

    logger.info(
        f"{ds_name}: nodes={data.num_nodes}, edges={data.edge_index.shape[1]}, "
        f"prune={graph_stats['prune_ratio']:.3f}, "
        f"t_proxy={t_proxy:.1f}s, t_grad={t_grad:.1f}s, t_prune={t_prune:.1f}s, "
        f"t_total={t_total:.1f}s, mem={peak_mem:.0f}MB, acc={ds_result['test_acc']:.4f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device({"training": {"device": "cuda"}})
    logger = get_logger("scalability")

    datasets = ["cora", "citeseer", "pubmed", "actor", "texas", "cornell", "wisconsin"]
    results = []

    for ds in datasets:
        config_path = f"configs/graca_lite_{ds}.yaml"
        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path}")
            continue

        logger.info(f"=== Scalability: {ds} ===")
        try:
            result = run_scalability(config_path, args.seed, device)
            results.append(result)

            # Write to CSV
            write_result_row({
                "run_id": f"scalability_{ds}_seed{args.seed}",
                "seed": args.seed, "dataset": ds, "method": "GraCA-lite",
                "oracle_only": False, "proxy_model": "GCN", "downstream_model": "GCN",
                "prune_ratio": result["prune_ratio"],
                "num_edges_before": result["num_edges"],
                "num_edges_after": result["edges_after"],
                "isolated_nodes": 0,
                "val_acc": 0, "test_acc": result["test_acc"],
                "test_f1": 0, "best_epoch": 0, "runtime": result["t_total"],
                "config_path": config_path, "graph_path": "", "checkpoint_path": "",
            }, "results/scalability/scalability_results.csv")
        except Exception as e:
            logger.error(f"Failed on {ds}: {e}")

    # Print summary
    print("\n" + "=" * 100)
    print("SCALABILITY RESULTS")
    print("=" * 100)
    print(f"{'Dataset':12s} {'Nodes':>8s} {'Edges':>8s} {'Prune%':>8s} {'T_proxy':>8s} {'T_grad':>8s} {'T_prune':>8s} {'T_total':>8s} {'Mem(MB)':>8s} {'Acc':>8s}")
    print("-" * 100)
    for r in results:
        print(f"{r['dataset']:12s} {r['num_nodes']:8d} {r['num_edges']:8d} {r['prune_ratio']*100:7.1f}% {r['t_proxy']:7.1f}s {r['t_gradient']:7.1f}s {r['t_pruning']:7.1f}s {r['t_total']:7.1f}s {r['peak_memory_mb']:7.0f} {r['test_acc']:8.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
