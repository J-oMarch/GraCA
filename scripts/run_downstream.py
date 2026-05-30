"""
Run downstream GNN training on a saved sanitized graph.
Usage:
    python scripts/run_downstream.py --config configs/graca_lite_cora.yaml \
        --graph sanitized_graphs/graca_lite/Cora_seed0.pt --model GCN --seed 0
"""
import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.graca.save_graph import load_sanitized_graph
from src.training.train_downstream import train_downstream
from src.eval.result_writer import write_result_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--graph", type=str, required=True, help="Path to sanitized graph .pt file")
    parser.add_argument("--model", type=str, default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    set_seed(args.seed)

    # Load data
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)

    # Load sanitized graph
    graph_data = load_sanitized_graph(args.graph)
    pruned_edge_index = graph_data["edge_index"]
    graph_stats = graph_data["graph_stats"]

    print(f"Loaded sanitized graph: {graph_stats}")

    # Train downstream
    result = train_downstream(
        model_name=args.model,
        data=data,
        edge_index=pruned_edge_index,
        config=config,
        num_features=num_features,
        num_classes=num_classes,
        device=device,
        seed=args.seed,
    )

    print(f"\nResults:")
    print(f"  Val Acc:  {result['val_acc']:.4f}")
    print(f"  Test Acc: {result['test_acc']:.4f}")
    print(f"  Test F1:  {result['test_f1']:.4f}")
    print(f"  Best Epoch: {result['best_epoch']}")
    print(f"  Runtime:  {result['runtime']:.1f}s")

    # Write result
    ds_name = config["dataset"]["name"]
    write_result_row({
        "run_id": f"downstream_{ds_name}_{args.model}_seed{args.seed}",
        "seed": args.seed,
        "dataset": ds_name,
        "method": "GraCA-lite",
        "oracle_only": False,
        "proxy_model": config["proxy_model"]["name"],
        "downstream_model": args.model,
        "prune_ratio": graph_stats.get("prune_ratio", 0),
        "num_edges_before": graph_stats.get("num_edges_before", 0),
        "num_edges_after": graph_stats.get("num_edges_after", 0),
        "isolated_nodes": graph_stats.get("isolated_nodes", 0),
        "min_degree": graph_stats.get("min_degree", 0),
        "mean_degree": graph_stats.get("mean_degree", 0),
        "largest_connected_component_ratio": graph_stats.get("largest_connected_component_ratio", 0),
        "val_acc": result["val_acc"],
        "test_acc": result["test_acc"],
        "test_f1": result["test_f1"],
        "best_epoch": result["best_epoch"],
        "runtime": result["runtime"],
        "config_path": args.config,
        "graph_path": args.graph,
        "checkpoint_path": "",
    }, "results/main/downstream_results.csv")


if __name__ == "__main__":
    main()
