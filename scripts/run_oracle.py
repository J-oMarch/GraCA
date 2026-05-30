"""
Run Oracle GraCA: full-label gradient scoring for diagnostic upper-bound.
ORACLE ONLY: uses full labels. Do not include in main semi-supervised results.
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
from src.models.model_factory import build_model
from src.graca.oracle import run_oracle_graca
from src.training.train_downstream import train_downstream
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("run_oracle")

    # Verify oracle config
    assert config["experiment"]["oracle_only"] is True, "Must set oracle_only=true"
    assert config["experiment"]["method"] == "oracle", "Must set method=oracle"

    seeds = config.get("experiment", {}).get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    device = get_device(config)
    ds_name = config["dataset"]["name"]

    for seed in seeds:
        set_seed(seed)
        logger.info(f"=== Oracle GraCA on {ds_name}, seed={seed} ===")

        # Load data
        data, num_features, num_classes = load_dataset(config)
        data = data.to(device)

        # Build and train proxy with oracle labels (for gradient collection)
        proxy_cfg = config["proxy_model"]
        train_cfg = config["training"]

        model = build_model(
            name=proxy_cfg["name"],
            in_dim=num_features,
            hidden_dim=proxy_cfg["hidden_dim"],
            out_dim=num_classes,
            num_layers=proxy_cfg.get("num_layers", 2),
            dropout=proxy_cfg.get("dropout", 0.5),
        ).to(device)

        # Quick train proxy (oracle uses full labels for training too)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("lr", 0.01),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
        )

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)

        # Train with train labels (same as practical)
        import torch.nn.functional as F
        from src.training.evaluator import evaluate
        from src.training.early_stopping import EarlyStopping

        early_stopping = EarlyStopping(patience=train_cfg.get("patience", 100))
        for epoch in range(1, train_cfg.get("epochs", 300) + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_eval = model(x, edge_index)
                val_metrics = evaluate(logits_eval, y, data.val_mask.to(device))

            if early_stopping.step(val_metrics["accuracy"], model, epoch):
                break

        early_stopping.load_best_model(model)

        # Run oracle GraCA
        oracle_result = run_oracle_graca(config, model, data, device)

        # Downstream retraining on oracle graph
        logger.info("Training downstream on oracle graph...")
        downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])

        for ds_model_name in downstream_names:
            set_seed(seed)
            ds_result = train_downstream(
                model_name=ds_model_name,
                data=data,
                edge_index=oracle_result["pruned_edge_index"],
                config=config,
                num_features=num_features,
                num_classes=num_classes,
                device=device,
                seed=seed,
            )

            result_row = {
                "run_id": f"oracle_{ds_name}_seed{seed}",
                "seed": seed,
                "dataset": ds_name,
                "method": "Oracle GraCA",
                "oracle_only": True,
                "proxy_model": proxy_cfg["name"],
                "downstream_model": ds_model_name,
                "prune_ratio": oracle_result["graph_stats"]["prune_ratio"],
                "num_edges_before": oracle_result["graph_stats"]["num_edges_before"],
                "num_edges_after": oracle_result["graph_stats"]["num_edges_after"],
                "isolated_nodes": oracle_result["graph_stats"]["isolated_nodes"],
                "min_degree": oracle_result["graph_stats"]["min_degree"],
                "mean_degree": oracle_result["graph_stats"]["mean_degree"],
                "largest_connected_component_ratio": oracle_result["graph_stats"]["largest_connected_component_ratio"],
                "val_acc": ds_result["val_acc"],
                "test_acc": ds_result["test_acc"],
                "test_f1": ds_result["test_f1"],
                "best_epoch": ds_result["best_epoch"],
                "runtime": ds_result["runtime"],
                "config_path": args.config,
                "graph_path": oracle_result["graph_path"],
                "checkpoint_path": "",
            }
            write_result_row(result_row, "results/oracle/oracle_results.csv")

    logger.info("Oracle GraCA complete!")


if __name__ == "__main__":
    main()
