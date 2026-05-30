"""
Hyperparameter sweep for GraCA-lite.
Usage:
    python scripts/run_sweep.py --config configs/graca_lite_cora.yaml --seed 0
"""
import sys
import os
import argparse
import copy
import itertools
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
from src.training.train_downstream import train_downstream
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


# Sweep parameters
SWEEP_SPACE = {
    "pseudo.tau": [0.6, 0.7, 0.8],
    "pruning.beta": [0.05, 0.10, 0.20, 0.30],
    "scoring.eta": [0.5, 1.0, 2.0],
}


def set_nested(d, key, value):
    """Set a nested dict value using dot notation key."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_nested(d, key, default=None):
    """Get a nested dict value using dot notation key."""
    keys = key.split(".")
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def run_single_config(config, device, seed, logger):
    """Run a single configuration and return downstream results."""
    set_seed(seed)

    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)

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
    teacher_probs = teacher.predict(x, edge_index)
    pseudo_cfg = config.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.6)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )

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

    # Edge scoring
    scoring_cfg = config.get("scoring", {})
    edge_scores = compute_edge_scores(
        grad=grad, edge_index=edge_index, rho_score=rho_score,
        num_nodes=num_nodes, eta=scoring_cfg.get("eta", 1.0), epsilon_rho=eps_rho,
    )

    P = edge_scores["P"]
    if config.get("dataset", {}).get("undirected", True):
        P = average_undirected_scores(edge_index, P)

    # Pruning
    pruning_cfg = config.get("pruning", {})
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=edge_index, risk_score=P, num_nodes=num_nodes,
        beta=pruning_cfg.get("beta", 0.2),
        min_degree=pruning_cfg.get("min_degree", 1),
        lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
        undirected=config.get("dataset", {}).get("undirected", True),
        protect_self_loops=pruning_cfg.get("protect_self_loops", True),
    )

    # Downstream (only GCN for sweep speed)
    ds_result = train_downstream(
        model_name="GCN", data=data, edge_index=pruned_edge_index,
        config=config, num_features=num_features, num_classes=num_classes,
        device=device, seed=seed,
    )

    return {
        "val_acc": ds_result["val_acc"],
        "test_acc": ds_result["test_acc"],
        "prune_ratio": graph_stats["prune_ratio"],
        "num_edges_after": graph_stats["num_edges_after"],
        "isolated_nodes": graph_stats["isolated_nodes"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep_keys", nargs="+", default=None,
                        help="Keys to sweep (default: all in SWEEP_SPACE)")
    args = parser.parse_args()

    base_config = load_config(args.config)
    logger = get_logger("run_sweep")
    device = get_device(base_config)
    ds_name = base_config["dataset"]["name"]

    # Determine which keys to sweep
    sweep_keys = args.sweep_keys if args.sweep_keys else list(SWEEP_SPACE.keys())
    sweep_values = {k: SWEEP_SPACE[k] for k in sweep_keys if k in SWEEP_SPACE}

    # Generate all combinations
    all_keys = list(sweep_values.keys())
    all_combos = list(itertools.product(*[sweep_values[k] for k in all_keys]))

    logger.info(f"Sweeping {len(all_combos)} configurations on {ds_name}")

    best_val_acc = 0
    best_config = None

    for i, combo in enumerate(all_combos):
        config = copy.deepcopy(base_config)
        param_str = []
        for key, val in zip(all_keys, combo):
            set_nested(config, key, val)
            param_str.append(f"{key}={val}")

        logger.info(f"[{i+1}/{len(all_combos)}] {', '.join(param_str)}")

        try:
            result = run_single_config(config, device, args.seed, logger)

            # Write result
            write_result_row({
                "run_id": f"sweep_{ds_name}_seed{args.seed}",
                "seed": args.seed,
                "dataset": ds_name,
                "method": "GraCA-lite-sweep",
                "oracle_only": False,
                "proxy_model": config["proxy_model"]["name"],
                "downstream_model": "GCN",
                "prune_ratio": result["prune_ratio"],
                "num_edges_before": 0,
                "num_edges_after": result["num_edges_after"],
                "isolated_nodes": result["isolated_nodes"],
                "val_acc": result["val_acc"],
                "test_acc": result["test_acc"],
                "test_f1": 0,
                "best_epoch": 0,
                "runtime": 0,
                "config_path": ",".join(param_str),
                "graph_path": "",
                "checkpoint_path": "",
            }, "results/sweeps/sweep_results.csv")

            if result["val_acc"] > best_val_acc:
                best_val_acc = result["val_acc"]
                best_config = dict(zip(all_keys, combo))
                logger.info(f"  New best: val_acc={result['val_acc']:.4f}, test_acc={result['test_acc']:.4f}")

        except Exception as e:
            logger.error(f"  Failed: {e}")

    logger.info(f"\nBest config: {best_config}")
    logger.info(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
