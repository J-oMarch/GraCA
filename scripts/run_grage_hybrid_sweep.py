#!/usr/bin/env python3
"""
GraGE-Hybrid Sweep: Evaluate hybrid scoring variants across datasets and noise types.

This script runs the hybrid scoring variants that combine static feature smoothness
with training-dynamics calibration.

Usage:
    python scripts/run_grage_hybrid_sweep.py --stage signal
    python scripts/run_grage_hybrid_sweep.py --stage confirmation
"""
import os
import sys
import argparse
import time
import json
import logging
import itertools
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_dataset
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE
from src.training.train_downstream import train_downstream
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.data.load_data import compute_edge_homophily
from src.graca.pruning import prune_graph, compute_graph_stats
from src.graca.edge_bench import compute_edge_bench_scores_simple
from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order
from src.grage.unrolled_hypergradient import compute_edge_gate_influence_unrolled
from src.grage.hybrid_score import compute_grage_hybrid_score
from src.baselines.similarity_pruning import run_jaccard_pruning, run_cosine_pruning
from src.baselines.random_pruning import run_random_pruning, run_degree_aware_random
from src.utils.mask_split import split_train_support_score
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Default config ───
DEFAULT_CONFIG = {
    "dataset": {"undirected": True},
    "pruning": {"beta": 0.2, "min_degree": 1, "lambda_theta": 0.0},
    "training": {"lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 50},
    "downstream_model": {"names": ["GCN"]},
}

# ─── Hyperparameter sweep ───
LAMBDA_POS_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0]
LAMBDA_NEG_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5]
SCORE_RATIO_VALUES = [0.2, 0.3, 0.5]
DEGREE_NORM_VALUES = [False, True]


def get_method_configs():
    """Return all method configurations to evaluate."""
    methods = []

    # 1. Feature-only (baseline)
    methods.append({
        "name": "Feature-only",
        "type": "feature_only",
        "lambda_pos": 0.0, "lambda_neg": 0.0,
        "score_ratio": 0.3, "degree_norm": False,
    })

    # 2. GraGE-FO-grad (gradient only)
    methods.append({
        "name": "GraGE-FO-grad",
        "type": "grad_only",
        "lambda_pos": 1.0, "lambda_neg": 0.0,
        "score_ratio": 0.3, "degree_norm": False,
    })

    # 3. GraGE-FO-neggrad (negative gradient only)
    methods.append({
        "name": "GraGE-FO-neggrad",
        "type": "neg_grad",
        "lambda_pos": 0.0, "lambda_neg": 1.0,
        "score_ratio": 0.3, "degree_norm": False,
    })

    # 4. GraGE-FO-absgrad (absolute gradient)
    methods.append({
        "name": "GraGE-FO-absgrad",
        "type": "abs_grad",
        "lambda_pos": 1.0, "lambda_neg": 0.0,
        "score_ratio": 0.3, "degree_norm": False,
    })

    # 5. GraGE-Hybrid-FO-pos (feature + positive gradient)
    for lam in LAMBDA_POS_VALUES:
        methods.append({
            "name": f"GraGE-Hybrid-FO-pos-lam{lam}",
            "type": "feature_plus_pos",
            "lambda_pos": lam, "lambda_neg": 0.0,
            "score_ratio": 0.3, "degree_norm": False,
        })

    # 6. GraGE-Hybrid-FO-posneg (feature + positive - negative gradient)
    for lam_pos, lam_neg in itertools.product(LAMBDA_POS_VALUES, LAMBDA_NEG_VALUES):
        methods.append({
            "name": f"GraGE-Hybrid-FO-posneg-lp{lam_pos}-ln{lam_neg}",
            "type": "feature_pos_neg",
            "lambda_pos": lam_pos, "lambda_neg": lam_neg,
            "score_ratio": 0.3, "degree_norm": False,
        })

    # 7. GraGE-Hybrid-FO-posneg-degree
    for lam_pos, lam_neg in itertools.product(LAMBDA_POS_VALUES[:3], LAMBDA_NEG_VALUES[:3]):
        methods.append({
            "name": f"GraGE-Hybrid-FO-posneg-degree-lp{lam_pos}-ln{lam_neg}",
            "type": "feature_pos_neg_degree",
            "lambda_pos": lam_pos, "lambda_neg": lam_neg,
            "score_ratio": 0.3, "degree_norm": True,
        })

    # 8. GraGE-Hybrid-UnrolledK1/K3/K5
    for K in [1, 3, 5]:
        for lam_pos, lam_neg in [(0.25, 0.25), (0.5, 0.1), (0.1, 0.5)]:
            methods.append({
                "name": f"GraGE-Hybrid-UnrolledK{K}-posneg-lp{lam_pos}-ln{lam_neg}",
                "type": "feature_pos_neg",
                "lambda_pos": lam_pos, "lambda_neg": lam_neg,
                "score_ratio": 0.3, "degree_norm": False,
                "inner_steps": K,
            })

    return methods


def compute_feature_risk(x, edge_index, device):
    """Compute feature-based risk score: 1 - cosine similarity."""
    src = edge_index[0]
    dst = edge_index[1]
    cosine_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
    feature_risk = 1.0 - cosine_sim
    return feature_risk


def train_model_for_grage(model, x, edge_index, y, train_mask, val_mask,
                         lr=0.01, weight_decay=5e-4, epochs=200, patience=50, seed=42):
    """Train a model and return its state_dict for GraGE computation."""
    set_seed(seed)
    device = x.device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(x, edge_index)
            val_pred = logits_val[val_mask].argmax(dim=1)
            val_acc = (val_pred == y[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state_dict


def run_single_experiment(
    dataset_name, noise_type, noise_ratio, seed, method_config,
    downstream_model_name, prune_ratio, data, noisy_edge_index, bad_edge_mask,
    device, config
):
    """Run a single experiment with the given method configuration."""
    set_seed(seed)
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    noisy_edge_index = noisy_edge_index.to(device)
    E_noisy = noisy_edge_index.shape[1]

    start_time = time.time()

    method_name = method_config["name"]
    method_type = method_config["type"]
    lambda_pos = method_config.get("lambda_pos", 0.0)
    lambda_neg = method_config.get("lambda_neg", 0.0)
    score_ratio = method_config.get("score_ratio", 0.3)
    degree_norm = method_config.get("degree_norm", False)
    inner_steps = method_config.get("inner_steps", 0)

    # Step 1: Compute edge scores based on method type
    if method_type == "feature_only":
        # Feature-only: just use feature risk
        edge_scores = compute_feature_risk(x, noisy_edge_index, device)

    elif method_type in ["grad_only", "neg_grad", "abs_grad", "feature_plus_pos",
                         "feature_pos_neg", "feature_pos_neg_degree"]:
        # Need dynamic gradient from GraGE
        num_classes = int(y.max().item()) + 1

        if inner_steps > 0:
            # Unrolled hypergradient
            def model_ctor():
                return GCN(
                    in_dim=x.shape[1], hidden_dim=64,
                    out_dim=num_classes, num_layers=2, dropout=0.5
                )

            # Train initial model to get init_state_dict
            model_init = model_ctor().to(device)
            init_state_dict = train_model_for_grage(
                model_init, x, noisy_edge_index, y, train_mask, val_mask,
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
                epochs=100, patience=50, seed=seed,
            )

            # Split train into support/score
            support_mask, score_mask = split_train_support_score(
                train_mask, y, score_ratio=score_ratio, seed=seed
            )

            result = compute_edge_gate_influence_unrolled(
                model_ctor=model_ctor,
                init_state_dict=init_state_dict,
                x=x, edge_index=noisy_edge_index, y=y,
                support_mask=support_mask, score_mask=score_mask,
                inner_steps=inner_steps, inner_lr=0.01,
                undirected=True, bad_edge_mask=bad_edge_mask,
            )
            dynamic_grad = result["raw_grad"]
        else:
            # First-order approximation
            model = GCN(
                in_dim=x.shape[1], hidden_dim=64,
                out_dim=num_classes, num_layers=2, dropout=0.5
            ).to(device)

            state_dict = train_model_for_grage(
                model, x, noisy_edge_index, y, train_mask, val_mask,
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
                epochs=200, patience=50, seed=seed,
            )
            model.load_state_dict(state_dict)

            # Split train into support/score
            support_mask, score_mask = split_train_support_score(
                train_mask, y, score_ratio=score_ratio, seed=seed
            )

            result = compute_edge_gate_influence_first_order(
                model=model, x=x, edge_index=noisy_edge_index, y=y,
                score_mask=score_mask, normalize=False, undirected=True,
                bad_edge_mask=bad_edge_mask,
            )
            dynamic_grad = result["raw_grad"]

        # Compute feature risk
        feature_risk = compute_feature_risk(x, noisy_edge_index, device)

        # Compute hybrid score
        degree = torch.zeros(x.shape[0], device=device)
        degree.scatter_add_(0, noisy_edge_index[1].cpu(), torch.ones(E_noisy))
        degree = degree.to(device)

        hybrid_result = compute_grage_hybrid_score(
            feature_risk=feature_risk,
            dynamic_grad=dynamic_grad,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            degree=degree if degree_norm else None,
            degree_norm=degree_norm,
            mode=method_type,
            undirected=True,
            edge_index=noisy_edge_index,
            bad_edge_mask=bad_edge_mask,
        )
        edge_scores = hybrid_result["hybrid_score"]

    else:
        raise ValueError(f"Unknown method type: {method_type}")

    # Step 2: Prune edges by score
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=noisy_edge_index,
        risk_score=edge_scores,
        num_nodes=x.shape[0],
        beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"],
        target_prune_ratio=prune_ratio,
    )

    # Step 3: Evaluate bad edge detection
    detection = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)

    # Step 4: Compute homophily
    homo_before = compute_edge_homophily(noisy_edge_index, y)
    homo_after = compute_edge_homophily(pruned_edge_index, y)

    # Step 5: Train downstream model
    downstream_results = train_downstream(
        model_name=downstream_model_name, data=data, edge_index=pruned_edge_index,
        config=config, num_features=x.shape[1], num_classes=int(y.max().item()) + 1,
        device=device, seed=seed,
    )

    runtime = time.time() - start_time

    # Build result dict
    result = {
        "dataset": dataset_name,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "method": method_name,
        "method_type": method_type,
        "lambda_pos": lambda_pos,
        "lambda_neg": lambda_neg,
        "score_ratio": score_ratio,
        "degree_norm": degree_norm,
        "inner_steps": inner_steps,
        "downstream_model": downstream_model_name,
        "test_acc": downstream_results["test_acc"],
        "test_f1": downstream_results["test_f1"],
        "val_acc": downstream_results["val_acc"],
        "bad_edge_precision": detection["bad_edge_precision"],
        "bad_edge_recall": detection["bad_edge_recall"],
        "bad_edge_f1": detection["bad_edge_f1"],
        "edge_score_auc": 0.0,
        "actual_prune_ratio": graph_stats["prune_ratio"],
        "edge_homophily_before": homo_before,
        "edge_homophily_after": homo_after,
        "num_edges_before": graph_stats["num_edges_before"],
        "num_edges_after": graph_stats["num_edges_after"],
        "runtime": runtime,
        "notes": "",
    }

    return result


def run_signal_diagnostic(device, output_dir):
    """Stage A: Signal diagnostic on 3 datasets × 3 noise types × 3 seeds."""
    datasets = ["Cora", "CiteSeer", "PubMed"]
    noise_types = [
        "cross_class_oracle",
        "feature_similar_cross_class",
        "degree_aligned_random",
        "random_inter_community",
        "low_feature_similarity",
    ]
    noise_ratio = 0.3
    seeds = [0, 1, 2]
    downstream_model = "GCN"
    prune_ratio = 0.2

    method_configs = get_method_configs()

    # Filter to a manageable subset for signal diagnostic
    # Keep key methods: feature_only, grad_only, feature_pos_neg with key lambdas
    key_methods = [m for m in method_configs if m["name"] in [
        "Feature-only",
        "GraGE-FO-grad",
        "GraGE-FO-neggrad",
        "GraGE-FO-absgrad",
        "GraGE-Hybrid-FO-pos-lam0.25",
        "GraGE-Hybrid-FO-pos-lam0.5",
        "GraGE-Hybrid-FO-posneg-lp0.25-ln0.25",
        "GraGE-Hybrid-FO-posneg-lp0.5-ln0.1",
        "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5",
        "GraGE-Hybrid-FO-posneg-degree-lp0.25-ln0.25",
        "GraGE-Hybrid-UnrolledK1-posneg-lp0.25-ln0.25",
        "GraGE-Hybrid-UnrolledK3-posneg-lp0.25-ln0.25",
    ]]

    return _run_experiment_matrix(
        datasets, noise_types, noise_ratio, seeds, downstream_model,
        prune_ratio, key_methods, device, output_dir, stage="signal"
    )


def run_confirmation(device, output_dir):
    """Stage B: Confirmation on best methods from signal diagnostic."""
    # These would be filled in after signal diagnostic analysis
    datasets = ["Cora", "CiteSeer", "PubMed", "Actor", "Texas", "Wisconsin"]
    noise_types = [
        "feature_similar_cross_class",
        "degree_aligned_random",
        "random_inter_community",
        "cross_class_oracle",
        "low_feature_similarity",
    ]
    noise_ratio = 0.3
    seeds = [0, 1, 2, 3, 4]
    downstream_models = ["GCN", "GAT", "GraphSAGE"]
    prune_ratio = 0.2

    # Placeholder: would be filled with best methods from signal diagnostic
    method_configs = [
        {"name": "Feature-only", "type": "feature_only",
         "lambda_pos": 0.0, "lambda_neg": 0.0, "score_ratio": 0.3, "degree_norm": False},
        {"name": "GraGE-Hybrid-FO-posneg-lp0.25-ln0.25", "type": "feature_pos_neg",
         "lambda_pos": 0.25, "lambda_neg": 0.25, "score_ratio": 0.3, "degree_norm": False},
    ]

    return _run_experiment_matrix(
        datasets, noise_types, noise_ratio, seeds, downstream_models,
        prune_ratio, method_configs, device, output_dir, stage="confirmation"
    )


def _run_experiment_matrix(
    datasets, noise_types, noise_ratio, seeds, downstream_models,
    prune_ratio, method_configs, device, output_dir, stage="signal"
):
    """Run the full experiment matrix."""
    all_results = []
    total = len(datasets) * len(noise_types) * len(seeds) * len(method_configs)
    if isinstance(downstream_models, list):
        total *= len(downstream_models)

    config = DEFAULT_CONFIG.copy()
    completed = 0

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        try:
            # Create config for this dataset
            dataset_config = config.copy()
            dataset_config["dataset"]["name"] = dataset_name
            data, num_features, num_classes = load_dataset(dataset_config)
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        data = data.to(device)

        for noise_type in noise_types:
            logger.info(f"\n--- Noise: {noise_type} ---")

            for seed in seeds:
                set_seed(seed)

                # Inject noise
                noise_result = inject_noise(
                    edge_index=data.edge_index,
                    num_nodes=data.num_nodes,
                    noise_type=noise_type,
                    noise_ratio=noise_ratio,
                    x=data.x,
                    y=data.y,
                    train_mask=data.train_mask,
                    seed=seed,
                )

                noisy_edge_index = noise_result["noisy_edge_index"]
                bad_edge_mask = noise_result["bad_edge_mask"]

                # Create a copy of data with noisy edge_index
                data_noisy = data.clone()
                data_noisy.edge_index = noisy_edge_index

                for method_config in method_configs:
                    if isinstance(downstream_models, str):
                        ds_models = [downstream_models]
                    else:
                        ds_models = downstream_models

                    for ds_model in ds_models:
                        try:
                            result = run_single_experiment(
                                dataset_name=dataset_name,
                                noise_type=noise_type,
                                noise_ratio=noise_ratio,
                                seed=seed,
                                method_config=method_config,
                                downstream_model_name=ds_model,
                                prune_ratio=prune_ratio,
                                data=data_noisy,
                                noisy_edge_index=noisy_edge_index,
                                bad_edge_mask=bad_edge_mask,
                                device=device,
                                config=config,
                            )
                            all_results.append(result)
                            completed += 1

                            if completed % 10 == 0:
                                logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

                        except Exception as e:
                            logger.error(f"Failed: {dataset_name}/{noise_type}/seed{seed}/{method_config['name']}: {e}")
                            import traceback
                            traceback.print_exc()

    # Save results
    df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, "results.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total experiments: {len(all_results)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="GraGE-Hybrid Sweep Experiments")
    parser.add_argument("--stage", choices=["signal", "confirmation"], default="signal",
                        help="Experiment stage: signal (small) or confirmation (full)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results_clean/grage_hybrid_sweep/{args.stage}"

    # Run experiments
    if args.stage == "signal":
        df = run_signal_diagnostic(device, output_dir)
    else:
        df = run_confirmation(device, output_dir)

    # Print summary
    if df is not None and len(df) > 0:
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)

        # Group by method and compute mean test_acc
        summary = df.groupby("method")["test_acc"].agg(["mean", "std", "count"])
        summary = summary.sort_values("mean", ascending=False)
        logger.info("\nMean test accuracy by method:")
        logger.info(summary.to_string())


if __name__ == "__main__":
    main()
