"""
Smoke test: run a minimal experiment on Cora seed=0 to verify the pipeline works.
Tests: Original, GraCA-lite, Random Pruning (matched ratio).
Outputs: results/smoke/smoke_results.csv
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.training.train_downstream import train_downstream
from src.training.train_proxy import train_proxy
from src.graca.gradient_collector import collect_hidden_gradients
from src.graca.edge_scoring import compute_edge_scores, average_undirected_scores, compute_rho_score
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.pruning import prune_graph
from src.baselines.random_pruning import run_random_pruning
from src.eval.result_writer import write_result_row
from src.data.leakage_check import ensure_no_test_label_leakage


def run_smoke():
    print("=" * 60)
    print("GraCA Smoke Test - Cora seed=0")
    print("=" * 60)

    config = load_config("configs/graca_lite_cora.yaml")
    device = get_device(config)
    seed = 0
    set_seed(seed)

    # Load data
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    print(f"Data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, {num_features} features, {num_classes} classes")
    print(f"Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

    # Leakage check
    ensure_no_test_label_leakage(config, data.train_mask, data.test_mask, data.train_mask, mode="practical")
    print("✓ No test label leakage")

    result_dir = "results/smoke/"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = f"{result_dir}/smoke_results.csv"

    # --- 1. Original ---
    print("\n--- Original ---")
    set_seed(seed)
    orig_result = train_downstream(
        model_name="GCN", data=data, edge_index=data.edge_index,
        config=config, num_features=num_features, num_classes=num_classes,
        device=device, seed=seed,
    )
    print(f"  Test Acc: {orig_result['test_acc']:.4f}")
    write_result_row({
        "run_id": "smoke_original", "seed": seed, "dataset": "Cora",
        "method": "Original", "oracle_only": False,
        "proxy_model": "-", "downstream_model": "GCN",
        "actual_prune_ratio": 0.0,
        "num_edges_before": data.edge_index.shape[1],
        "num_edges_after": data.edge_index.shape[1],
        "isolated_nodes": 0, "min_degree": 0, "mean_degree": 0,
        "val_acc": orig_result["val_acc"],
        "test_acc": orig_result["test_acc"],
        "test_f1": orig_result["test_f1"],
        "best_epoch": orig_result["best_epoch"],
        "runtime": orig_result["runtime"],
        "config_path": "configs/graca_lite_cora.yaml",
        "graph_path": "", "checkpoint_path": "",
    }, csv_path)

    # --- 2. GraCA-lite ---
    print("\n--- GraCA-lite ---")
    set_seed(seed)
    model, teacher, train_log, saved_checkpoints = train_proxy(
        config, data, num_features, num_classes, device, seed
    )

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
        collect_layer="all",
    )

    grad = grad_result["grad"]
    num_nodes = x.shape[0]
    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

    edge_scores = compute_edge_scores(
        grad=grad, edge_index=edge_index, rho_score=rho_score,
        num_nodes=num_nodes, eta=1.0, epsilon_rho=eps_rho,
    )
    P = edge_scores["P"]
    if config.get("dataset", {}).get("undirected", True):
        P = average_undirected_scores(edge_index, P)

    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
        beta=0.2, min_degree=1, lambda_theta=0.0,
        undirected=True, protect_self_loops=True,
    )

    print(f"  Pruned: {graph_stats['num_edges_before']} -> {graph_stats['num_edges_after']} (ratio={graph_stats['prune_ratio']:.4f})")

    set_seed(seed)
    graca_result = train_downstream(
        model_name="GCN", data=data, edge_index=pruned_edge_index,
        config=config, num_features=num_features, num_classes=num_classes,
        device=device, seed=seed,
    )
    print(f"  Test Acc: {graca_result['test_acc']:.4f}")

    graca_prune_ratio = graph_stats["prune_ratio"]
    write_result_row({
        "run_id": "smoke_graca", "seed": seed, "dataset": "Cora",
        "method": "GraCA-lite", "oracle_only": False,
        "proxy_model": "GCN", "downstream_model": "GCN",
        "actual_prune_ratio": graca_prune_ratio,
        "num_edges_before": graph_stats["num_edges_before"],
        "num_edges_after": graph_stats["num_edges_after"],
        "isolated_nodes": graph_stats["isolated_nodes"],
        "min_degree": graph_stats["min_degree"],
        "mean_degree": graph_stats["mean_degree"],
        "val_acc": graca_result["val_acc"],
        "test_acc": graca_result["test_acc"],
        "test_f1": graca_result["test_f1"],
        "best_epoch": graca_result["best_epoch"],
        "runtime": graca_result["runtime"],
        "config_path": "configs/graca_lite_cora.yaml",
        "graph_path": "", "checkpoint_path": "",
    }, csv_path)

    # --- 3. Random Pruning (matched ratio) ---
    print("\n--- Random Pruning (matched ratio) ---")
    set_seed(seed)
    random_results, random_stats = run_random_pruning(
        data, config, num_features, num_classes, device, seed,
        prune_ratio=graca_prune_ratio,
    )
    print(f"  Test Acc: {random_results['GCN']['test_acc']:.4f}")
    write_result_row({
        "run_id": "smoke_random", "seed": seed, "dataset": "Cora",
        "method": "Random Pruning", "oracle_only": False,
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
        "config_path": "configs/graca_lite_cora.yaml",
        "graph_path": "", "checkpoint_path": "",
    }, csv_path)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"{'Method':20s} {'Test Acc':>10s} {'Prune%':>10s}")
    print("-" * 40)
    print(f"{'Original':20s} {orig_result['test_acc']*100:10.2f} {0:10.1f}%")
    print(f"{'GraCA-lite':20s} {graca_result['test_acc']*100:10.2f} {graca_prune_ratio*100:10.1f}%")
    print(f"{'Random (matched)':20s} {random_results['GCN']['test_acc']*100:10.2f} {random_stats['prune_ratio']*100:10.1f}%")
    print("=" * 60)
    print(f"\nResults saved to: {csv_path}")
    print("✓ Smoke test passed!")


if __name__ == "__main__":
    run_smoke()
