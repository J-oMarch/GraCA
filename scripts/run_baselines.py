"""
Run all baselines: Original, DropEdge, Random Pruning, Homophily Pruning.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.baselines.original import run_original
from src.baselines.dropedge import run_dropedge
from src.baselines.random_pruning import run_random_pruning
from src.baselines.homophily_pruning import run_homophily_pruning
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline", type=str, default="all",
                        choices=["all", "original", "dropedge", "random", "homophily"])
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_baselines")

    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    result_dir = "results/baselines/"
    baseline_type = args.baseline

    for seed in seeds:
        set_seed(seed)
        data, num_features, num_classes = load_dataset(config)
        data = data.to(device)
        logger.info(f"=== Baselines on {ds_name}, seed={seed} ===")

        # Original
        if baseline_type in ("all", "original"):
            logger.info("Running Original baseline...")
            original_results = run_original(data, config, num_features, num_classes, device, seed)
            for model_name, res in original_results.items():
                write_result_row({
                    "run_id": f"original_{ds_name}_{model_name}_seed{seed}",
                    "seed": seed,
                    "dataset": ds_name,
                    "method": "Original",
                    "oracle_only": False,
                    "proxy_model": "-",
                    "downstream_model": model_name,
                    "prune_ratio": 0.0,
                    "num_edges_before": data.edge_index.shape[1],
                    "num_edges_after": data.edge_index.shape[1],
                    "isolated_nodes": 0,
                    "val_acc": res["val_acc"],
                    "test_acc": res["test_acc"],
                    "test_f1": res["test_f1"],
                    "best_epoch": res["best_epoch"],
                    "runtime": res["runtime"],
                    "config_path": args.config,
                    "graph_path": "",
                    "checkpoint_path": "",
                }, f"{result_dir}/baseline_results.csv")

        # DropEdge
        if baseline_type in ("all", "dropedge"):
            logger.info("Running DropEdge baseline...")
            dropedge_results = run_dropedge(data, config, num_features, num_classes, device, seed)
            for model_name, res in dropedge_results.items():
                write_result_row({
                    "run_id": f"dropedge_{ds_name}_{model_name}_seed{seed}",
                    "seed": seed,
                    "dataset": ds_name,
                    "method": "DropEdge",
                    "oracle_only": False,
                    "proxy_model": "-",
                    "downstream_model": model_name,
                    "prune_ratio": config.get("baselines", {}).get("dropedge_rate", 0.2),
                    "num_edges_before": data.edge_index.shape[1],
                    "num_edges_after": data.edge_index.shape[1],
                    "isolated_nodes": 0,
                    "val_acc": res["val_acc"],
                    "test_acc": res["test_acc"],
                    "test_f1": res["test_f1"],
                    "best_epoch": res["best_epoch"],
                    "runtime": res["runtime"],
                    "config_path": args.config,
                    "graph_path": "",
                    "checkpoint_path": "",
                }, f"{result_dir}/baseline_results.csv")

        # Random Pruning
        if baseline_type in ("all", "random"):
            logger.info("Running Random Pruning baseline...")
            random_results, random_stats = run_random_pruning(
                data, config, num_features, num_classes, device, seed
            )
            for model_name, res in random_results.items():
                write_result_row({
                    "run_id": f"random_{ds_name}_{model_name}_seed{seed}",
                    "seed": seed,
                    "dataset": ds_name,
                    "method": "Random Pruning",
                    "oracle_only": False,
                    "proxy_model": "-",
                    "downstream_model": model_name,
                    "prune_ratio": random_stats["prune_ratio"],
                    "num_edges_before": random_stats["num_edges_before"],
                    "num_edges_after": random_stats["num_edges_after"],
                    "isolated_nodes": 0,
                    "val_acc": res["val_acc"],
                    "test_acc": res["test_acc"],
                    "test_f1": res["test_f1"],
                    "best_epoch": res["best_epoch"],
                    "runtime": res["runtime"],
                    "config_path": args.config,
                    "graph_path": "",
                    "checkpoint_path": "",
                }, f"{result_dir}/baseline_results.csv")

        # Homophily Pruning
        if baseline_type in ("all", "homophily"):
            logger.info("Running Homophily Pruning baseline...")
            homo_results, homo_stats = run_homophily_pruning(
                data, config, num_features, num_classes, device, seed
            )
            for model_name, res in homo_results.items():
                write_result_row({
                    "run_id": f"homophily_{ds_name}_{model_name}_seed{seed}",
                    "seed": seed,
                    "dataset": ds_name,
                    "method": "Homophily Pruning",
                    "oracle_only": False,
                    "proxy_model": "-",
                    "downstream_model": model_name,
                    "prune_ratio": homo_stats["prune_ratio"],
                    "num_edges_before": homo_stats["num_edges_before"],
                    "num_edges_after": homo_stats["num_edges_after"],
                    "isolated_nodes": 0,
                    "val_acc": res["val_acc"],
                    "test_acc": res["test_acc"],
                    "test_f1": res["test_f1"],
                    "best_epoch": res["best_epoch"],
                    "runtime": res["runtime"],
                    "config_path": args.config,
                    "graph_path": "",
                    "checkpoint_path": "",
                }, f"{result_dir}/baseline_results.csv")

    logger.info("All baselines complete!")


if __name__ == "__main__":
    main()
