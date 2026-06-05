#!/usr/bin/env python3
"""
Ambiguity and Stability Evidence: P0/P1 diagnostic experiment for StabilityResidual-GraGE.

P0: Ambiguity Contribution Analysis
    - Feature-defined Low/Medium/High ambiguity buckets
    - Bucket-gated residual variants
    - Changed-prune attribution

P1: Stability Validation and Alignment Destruction
    - Feature+Confidence control
    - Feature+Random Stability control
    - Feature+Shuffled Stability control
    - Feature+Permuted Stability control

Usage:
    python scripts/run_ambiguity_stability_evidence.py --mode smoke \
        --output_dir experiments/2026-06-05-ambiguity-stability-evidence/logs/smoke

    python scripts/run_ambiguity_stability_evidence.py --mode full \
        --output_dir experiments/2026-06-05-ambiguity-stability-evidence/logs/full
"""
import os
import sys
import argparse
import time
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_dataset, compute_edge_homophily
from src.models.gcn import GCN
from src.training.train_downstream import train_downstream
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.graca.pruning import prune_graph, compute_graph_stats
from src.grage.adaptive_score import (
    compute_stability_residual_score,
    compute_ambiguity_buckets,
    compute_confidence_edge_score,
    compute_random_stability_residual,
    compute_shuffled_stability_residual,
    compute_permuted_stability_residual,
    collect_multi_view_predictions,
    compute_node_stability,
    stability_to_edge_score,
    residualize_stability_score,
    rank_normalize,
    collect_multi_checkpoint_grads,
)
from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order
from src.grage.hybrid_score import compute_grage_hybrid_score
from src.baselines.random_pruning import run_degree_aware_random
from src.baselines.similarity_pruning import run_jaccard_pruning
from src.utils.mask_split import split_train_support_score
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Default config ───
DEFAULT_CONFIG = {
    "dataset": {"undirected": True},
    "pruning": {"beta": 0.2, "min_degree": 1},
    "training": {"lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 50},
    "downstream_model": {"names": ["GCN"]},
}


def compute_feature_risk(x, edge_index, device):
    """Compute feature-based risk score: 1 - cosine similarity."""
    src = edge_index[0]
    dst = edge_index[1]
    cosine_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
    return 1.0 - cosine_sim


def compute_feature_similarity(x, edge_index, device):
    """Compute cosine similarity for each edge."""
    src = edge_index[0]
    dst = edge_index[1]
    return F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)


def train_model_for_grage(model, x, edge_index, y, train_mask, val_mask,
                          lr=0.01, weight_decay=5e-4, epochs=200, patience=50, seed=42):
    """Train a model and return its state_dict for GraGE computation."""
    set_seed(seed)
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
        model.eval()
        with torch.no_grad():
            val_pred = model(x, edge_index)[val_mask].argmax(dim=1)
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


# ═══════════════════════════════════════════════════════════════════════════════
# P0: Ambiguity bucket diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_bucket_diagnostics(
    feature_risk, feature_similarity, edge_index, bad_edge_mask,
    feature_prune_mask, stability_prune_mask, stability_residual_score,
    raw_stability_score, residual, bucket_labels, num_buckets=3, device=None,
):
    """Compute per-bucket diagnostics for P0 analysis.

    Args:
        feature_risk: [E] 1 - cosine similarity.
        feature_similarity: [E] cosine similarity.
        edge_index: [2, E] edge indices.
        bad_edge_mask: [E] boolean, True = injected bad edge.
        feature_prune_mask: [E] boolean, True = pruned by Feature-only.
        stability_prune_mask: [E] boolean, True = pruned by StabilityResidual.
        stability_residual_score: [E] final StabilityResidual edge score.
        raw_stability_score: [E] raw stability edge score.
        residual: [E] stability residual after residualization.
        bucket_labels: [E] long tensor, bucket assignment.
        num_buckets: number of buckets.
        device: torch device.

    Returns:
        dict with per-bucket metrics and overall diagnostics.
    """
    from sklearn.metrics import roc_auc_score

    results = {"buckets": {}}

    for b in range(num_buckets):
        mask_b = bucket_labels == b
        if mask_b.sum() < 2:
            results["buckets"][str(b)] = {"count": int(mask_b.sum()), "skip": True}
            continue

        fr_b = feature_risk[mask_b].cpu().numpy()
        bad_b = bad_edge_mask[mask_b].cpu().numpy()
        fp_b = feature_prune_mask[mask_b].cpu().numpy()
        sp_b = stability_prune_mask[mask_b].cpu().numpy()
        stab_b = stability_residual_score[mask_b].cpu().numpy()
        raw_b = raw_stability_score[mask_b].cpu().numpy()
        resid_b = residual[mask_b].cpu().numpy()

        # Bad-edge precision/recall/F1 for Feature-only pruning
        tp_fo = (fp_b & bad_b).sum()
        fp_fo = (fp_b & ~bad_b).sum()
        fn_fo = (~fp_b & bad_b).sum()
        prec_fo = tp_fo / max(tp_fo + fp_fo, 1)
        rec_fo = tp_fo / max(tp_fo + fn_fo, 1)
        f1_fo = 2 * prec_fo * rec_fo / max(prec_fo + rec_fo, 1e-8)

        # Bad-edge precision/recall/F1 for StabilityResidual pruning
        tp_sr = (sp_b & bad_b).sum()
        fp_sr = (sp_b & ~bad_b).sum()
        fn_sr = (~sp_b & bad_b).sum()
        prec_sr = tp_sr / max(tp_sr + fp_sr, 1)
        rec_sr = tp_sr / max(tp_sr + fn_sr, 1)
        f1_sr = 2 * prec_sr * rec_sr / max(prec_sr + rec_sr, 1e-8)

        # AUCs (diagnostic only)
        feature_auc = 0.5
        raw_auc = 0.5
        resid_auc = 0.5
        if len(np.unique(bad_b)) > 1:
            try:
                feature_auc = float(roc_auc_score(bad_b, fr_b))
            except ValueError:
                pass
            try:
                raw_auc = float(roc_auc_score(bad_b, raw_b))
            except ValueError:
                pass
            try:
                resid_auc = float(roc_auc_score(bad_b, resid_b))
            except ValueError:
                pass

        # Overlap: edges pruned by both, only one, or neither
        both_pruned = (fp_b & sp_b).sum()
        fo_only = (fp_b & ~sp_b).sum()
        sr_only = (~fp_b & sp_b).sum()

        # Changed-prune bad-edge enrichment
        fo_only_bad = ((fp_b & ~sp_b) & bad_b).sum()
        sr_only_bad = ((~fp_b & sp_b) & bad_b).sum()
        fo_only_total = max(fo_only, 1)
        sr_only_total = max(sr_only, 1)

        results["buckets"][str(b)] = {
            "count": int(mask_b.sum()),
            "bad_count": int(bad_b.sum()),
            # Feature-only pruning diagnostics
            "fo_prune_count": int(fp_b.sum()),
            "fo_bad_prune_count": int(tp_fo),
            "fo_precision": float(prec_fo),
            "fo_recall": float(rec_fo),
            "fo_f1": float(f1_fo),
            # StabilityResidual pruning diagnostics
            "sr_prune_count": int(sp_b.sum()),
            "sr_bad_prune_count": int(tp_sr),
            "sr_precision": float(prec_sr),
            "sr_recall": float(rec_sr),
            "sr_f1": float(f1_sr),
            # AUCs
            "feature_risk_auc": feature_auc,
            "raw_stability_auc": raw_auc,
            "residual_auc": resid_auc,
            # Overlap
            "both_pruned": int(both_pruned),
            "fo_only_pruned": int(fo_only),
            "sr_only_pruned": int(sr_only),
            # Changed-prune enrichment
            "fo_only_bad_count": int(fo_only_bad),
            "sr_only_bad_count": int(sr_only_bad),
            "fo_only_bad_rate": float(fo_only_bad / fo_only_total),
            "sr_only_bad_rate": float(sr_only_bad / sr_only_total),
        }

    # Overall diagnostics
    results["overall"] = {
        "num_edges": int(feature_risk.shape[0]),
        "num_bad_edges": int(bad_edge_mask.sum()),
        "feature_prune_count": int(feature_prune_mask.sum()),
        "stability_prune_count": int(stability_prune_mask.sum()),
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Single experiment runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_p0_experiment(
    dataset_name, noise_type, noise_ratio, seed, method_config,
    prune_ratio, data, noisy_edge_index, bad_edge_mask, device, config,
    feature_risk, feature_similarity, stability_result, feature_prune_mask,
    bucket_info,
):
    """Run a single P0 bucket-gated experiment.

    For bucket-gated variants, the stability residual is applied only to edges
    in the specified bucket; other edges fall back to Feature-only.
    """
    set_seed(seed)
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    noisy_edge_index = noisy_edge_index.to(device)
    E_noisy = noisy_edge_index.shape[1]

    start_time = time.time()
    method_name = method_config["name"]
    method_type = method_config["type"]
    bucket_labels = bucket_info["bucket_labels"]

    if method_type == "feature_only":
        edge_scores = feature_risk.clone()

    elif method_type == "stability_residual_full":
        edge_scores = stability_result["edge_score"].clone()

    elif method_type == "feature_plus_stability":
        # Feature + real stability residual (same as full StabilityResidual)
        edge_scores = stability_result["edge_score"].clone()

    elif method_type == "bucket_gated_residual":
        # Apply residual only to edges in the target bucket
        target_bucket = method_config["target_bucket"]
        R_feature = rank_normalize(feature_risk)
        residual = stability_result["residual"]
        alpha = 0.5

        # Default: feature-only
        edge_scores = R_feature.clone()
        # Apply residual only to target bucket
        bucket_mask = bucket_labels == target_bucket
        edge_scores[bucket_mask] = (R_feature[bucket_mask] + alpha * residual[bucket_mask]).clamp(-2, 5)

    else:
        raise ValueError(f"Unknown P0 method type: {method_type}")

    # Prune
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=noisy_edge_index, risk_score=edge_scores,
        num_nodes=x.shape[0], beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"], target_prune_ratio=prune_ratio,
    )

    # Evaluate
    detection = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)
    homo_before = compute_edge_homophily(noisy_edge_index, y)
    homo_after = compute_edge_homophily(pruned_edge_index, y)

    downstream_results = train_downstream(
        model_name="GCN", data=data, edge_index=pruned_edge_index,
        config=config, num_features=x.shape[1], num_classes=int(y.max().item()) + 1,
        device=device, seed=seed,
    )

    runtime = time.time() - start_time

    # Bucket diagnostics
    bucket_diag = compute_bucket_diagnostics(
        feature_risk=feature_risk, feature_similarity=feature_similarity,
        edge_index=noisy_edge_index, bad_edge_mask=bad_edge_mask,
        feature_prune_mask=feature_prune_mask, stability_prune_mask=prune_mask,
        stability_residual_score=edge_scores,
        raw_stability_score=stability_result.get("edge_score", edge_scores),
        residual=stability_result.get("residual", edge_scores),
        bucket_labels=bucket_labels, num_buckets=3, device=device,
    )

    result = {
        "dataset": dataset_name,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "method": method_name,
        "method_type": method_type,
        "downstream_model": "GCN",
        "test_acc": downstream_results["test_acc"],
        "test_f1": downstream_results["test_f1"],
        "val_acc": downstream_results["val_acc"],
        "bad_edge_precision": detection["bad_edge_precision"],
        "bad_edge_recall": detection["bad_edge_recall"],
        "bad_edge_f1": detection["bad_edge_f1"],
        "actual_prune_ratio": graph_stats["prune_ratio"],
        "edge_homophily_before": homo_before,
        "edge_homophily_after": homo_after,
        "num_edges_before": graph_stats["num_edges_before"],
        "num_edges_after": graph_stats["num_edges_after"],
        "runtime": runtime,
    }

    # Add bucket diagnostics as flat columns
    for b_str, b_diag in bucket_diag["buckets"].items():
        if b_diag.get("skip"):
            continue
        for k, v in b_diag.items():
            result[f"b{b_str}_{k}"] = v

    return result


def run_single_p1_experiment(
    dataset_name, noise_type, noise_ratio, seed, method_config,
    prune_ratio, data, noisy_edge_index, bad_edge_mask, device, config,
    feature_risk, feature_similarity, predictions, node_stability,
    stability_result,
):
    """Run a single P1 alignment-destruction experiment."""
    set_seed(seed)
    x = data.x.to(device)
    y = data.y.to(device)
    noisy_edge_index = noisy_edge_index.to(device)

    start_time = time.time()
    method_name = method_config["name"]
    method_type = method_config["type"]

    if method_type == "feature_only":
        edge_scores = feature_risk.clone()

    elif method_type == "feature_plus_confidence":
        conf_result = compute_confidence_edge_score(
            predictions=predictions, feature_risk=feature_risk,
            edge_index=noisy_edge_index, feature_similarity=feature_similarity,
            undirected=True, device=device,
        )
        edge_scores = conf_result["edge_score"]

    elif method_type == "feature_plus_stability":
        edge_scores = stability_result["edge_score"].clone()

    elif method_type == "feature_plus_random_stability":
        rand_result = compute_random_stability_residual(
            feature_risk=feature_risk, edge_index=noisy_edge_index,
            feature_similarity=feature_similarity, seed=seed + 7919,
            undirected=True, device=device,
        )
        edge_scores = rand_result["edge_score"]

    elif method_type == "feature_plus_shuffled_stability":
        shuf_result = compute_shuffled_stability_residual(
            real_residual=stability_result["residual"],
            feature_risk=feature_risk, seed=seed + 7919, device=device,
        )
        edge_scores = shuf_result["edge_score"]

    elif method_type == "feature_plus_permuted_stability":
        perm_result = compute_permuted_stability_residual(
            node_instability=node_stability["node_instability"],
            feature_risk=feature_risk, edge_index=noisy_edge_index,
            feature_similarity=feature_similarity, seed=seed + 7919,
            undirected=True, device=device,
        )
        edge_scores = perm_result["edge_score"]

    else:
        raise ValueError(f"Unknown P1 method type: {method_type}")

    # Prune
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=noisy_edge_index, risk_score=edge_scores,
        num_nodes=x.shape[0], beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"], target_prune_ratio=prune_ratio,
    )

    # Evaluate
    detection = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)
    homo_before = compute_edge_homophily(noisy_edge_index, y)
    homo_after = compute_edge_homophily(pruned_edge_index, y)

    downstream_results = train_downstream(
        model_name="GCN", data=data, edge_index=pruned_edge_index,
        config=config, num_features=x.shape[1], num_classes=int(y.max().item()) + 1,
        device=device, seed=seed,
    )

    runtime = time.time() - start_time

    result = {
        "dataset": dataset_name,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "method": method_name,
        "method_type": method_type,
        "downstream_model": "GCN",
        "test_acc": downstream_results["test_acc"],
        "test_f1": downstream_results["test_f1"],
        "val_acc": downstream_results["val_acc"],
        "bad_edge_precision": detection["bad_edge_precision"],
        "bad_edge_recall": detection["bad_edge_recall"],
        "bad_edge_f1": detection["bad_edge_f1"],
        "actual_prune_ratio": graph_stats["prune_ratio"],
        "edge_homophily_before": homo_before,
        "edge_homophily_after": homo_after,
        "num_edges_before": graph_stats["num_edges_before"],
        "num_edges_after": graph_stats["num_edges_after"],
        "runtime": runtime,
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Method configurations
# ═══════════════════════════════════════════════════════════════════════════════


def get_p0_methods():
    """Return P0 bucket-gated method configurations."""
    return [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "Feature+Residual-LowOnly", "type": "bucket_gated_residual", "target_bucket": 0},
        {"name": "Feature+Residual-MediumOnly", "type": "bucket_gated_residual", "target_bucket": 1},
        {"name": "Feature+Residual-HighOnly", "type": "bucket_gated_residual", "target_bucket": 2},
        {"name": "Feature+Stability", "type": "feature_plus_stability"},
        {"name": "StabilityResidual-v5-dp0.15-grad-frozen", "type": "stability_residual_full"},
    ]


def get_p1_methods():
    """Return P1 alignment-destruction method configurations."""
    return [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "Feature+Confidence", "type": "feature_plus_confidence"},
        {"name": "Feature+Stability", "type": "feature_plus_stability"},
        {"name": "Feature+Random-Stability", "type": "feature_plus_random_stability"},
        {"name": "Feature+Shuffled-Stability", "type": "feature_plus_shuffled_stability"},
        {"name": "Feature+Permuted-Stability", "type": "feature_plus_permuted_stability"},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment_matrix(
    datasets, noise_types, noise_ratio, seeds, prune_ratio, device, output_dir,
):
    """Run the full P0+P1 experiment matrix."""
    all_results = []
    config = DEFAULT_CONFIG.copy()

    p0_methods = get_p0_methods()
    p1_methods = get_p1_methods()

    total = len(datasets) * len(noise_types) * len(seeds)
    completed = 0

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        try:
            dataset_config = config.copy()
            dataset_config["dataset"]["name"] = dataset_name
            data, num_features, num_classes = load_dataset(dataset_config)
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        for noise_type in noise_types:
            logger.info(f"\n--- Noise: {noise_type} ---")

            for seed in seeds:
                set_seed(seed)

                # Inject noise
                noise_result = inject_noise(
                    edge_index=data.edge_index, num_nodes=data.num_nodes,
                    noise_type=noise_type, noise_ratio=noise_ratio,
                    x=data.x, y=data.y, train_mask=data.train_mask, seed=seed,
                )
                noisy_edge_index = noise_result["noisy_edge_index"]
                bad_edge_mask = noise_result["bad_edge_mask"]

                data_noisy = data.clone()
                data_noisy.edge_index = noisy_edge_index

                x = data.x.to(device)
                y = data.y.to(device)
                train_mask = data.train_mask.to(device)
                val_mask = data.val_mask.to(device)
                noisy_edge_index = noisy_edge_index.to(device)
                bad_edge_mask = bad_edge_mask.to(device)

                # ─── Compute shared resources ───
                feature_risk = compute_feature_risk(x, noisy_edge_index, device)
                feature_similarity = compute_feature_similarity(x, noisy_edge_index, device)

                # Train model for StabilityResidual
                num_classes = int(y.max().item()) + 1

                def model_ctor():
                    return GCN(in_dim=num_features, hidden_dim=64,
                               out_dim=num_classes, num_layers=2, dropout=0.5)

                model = model_ctor().to(device)
                state_dict = train_model_for_grage(
                    model, x, noisy_edge_index, y, train_mask, val_mask,
                    lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"],
                    epochs=200, patience=50, seed=seed,
                )
                model.load_state_dict(state_dict)

                support_mask, score_mask = split_train_support_score(
                    train_mask, y, score_ratio=0.3, seed=seed,
                )

                # Collect frozen gradients
                checkpoint_grads = collect_multi_checkpoint_grads(
                    model_ctor=model_ctor, init_state_dict=state_dict,
                    x=x, edge_index=noisy_edge_index, y=y,
                    train_mask=train_mask, score_mask=score_mask,
                    checkpoint_fractions=[0.3, 0.5, 0.7, 0.9],
                    total_epochs=200, lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"], undirected=True,
                )
                checkpoint_grads = [checkpoint_grads[0].clone() for _ in checkpoint_grads]

                # Compute StabilityResidual (the shared signal)
                stability_result = compute_stability_residual_score(
                    model_ctor=model_ctor, init_state_dict=state_dict,
                    x=x, edge_index=noisy_edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask,
                    feature_risk=feature_risk, feature_similarity=feature_similarity,
                    checkpoint_grads=checkpoint_grads,
                    num_views=5, edge_dropout_rates=[0.0, 0.10, 0.15, 0.20, 0.30],
                    total_epochs=200, lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"], patience=50,
                    use_gradient_confidence=True, gradient_abstention_threshold=0.1,
                    undirected=True, bad_edge_mask=bad_edge_mask,
                    skip_residualization=False,
                )

                # Collect multi-view predictions for P1 Confidence control
                predictions = collect_multi_view_predictions(
                    model_ctor=model_ctor, init_state_dict=state_dict,
                    x=x, edge_index=noisy_edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask,
                    num_views=5, edge_dropout_rates=[0.0, 0.10, 0.15, 0.20, 0.30],
                    total_epochs=200, lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"], patience=50,
                )

                # Compute Feature-only pruning mask for bucket definition
                fo_scores = feature_risk.clone()
                _, fo_prune_mask, _ = prune_graph(
                    edge_index=noisy_edge_index, risk_score=fo_scores,
                    num_nodes=x.shape[0], beta=config["pruning"]["beta"],
                    min_degree=config["pruning"]["min_degree"], target_prune_ratio=prune_ratio,
                )

                # Compute ambiguity buckets
                bucket_info = compute_ambiguity_buckets(
                    feature_risk=feature_risk, prune_mask=fo_prune_mask, num_buckets=3,
                )

                # Get node stability for P1 permuted control
                node_stab = stability_result["node_stability"]

                # ─── Run P0 methods ───
                for method_config in p0_methods:
                    try:
                        result = run_single_p0_experiment(
                            dataset_name=dataset_name, noise_type=noise_type,
                            noise_ratio=noise_ratio, seed=seed,
                            method_config=method_config, prune_ratio=prune_ratio,
                            data=data_noisy, noisy_edge_index=noisy_edge_index,
                            bad_edge_mask=bad_edge_mask, device=device, config=config,
                            feature_risk=feature_risk, feature_similarity=feature_similarity,
                            stability_result=stability_result,
                            feature_prune_mask=fo_prune_mask,
                            bucket_info=bucket_info,
                        )
                        result["phase"] = "P0"
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"P0 failed: {dataset_name}/{noise_type}/seed{seed}/{method_config['name']}: {e}")
                        import traceback
                        traceback.print_exc()

                # ─── Run P1 methods ───
                for method_config in p1_methods:
                    try:
                        result = run_single_p1_experiment(
                            dataset_name=dataset_name, noise_type=noise_type,
                            noise_ratio=noise_ratio, seed=seed,
                            method_config=method_config, prune_ratio=prune_ratio,
                            data=data_noisy, noisy_edge_index=noisy_edge_index,
                            bad_edge_mask=bad_edge_mask, device=device, config=config,
                            feature_risk=feature_risk, feature_similarity=feature_similarity,
                            predictions=predictions, node_stability=node_stab,
                            stability_result=stability_result,
                        )
                        result["phase"] = "P1"
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"P1 failed: {dataset_name}/{noise_type}/seed{seed}/{method_config['name']}: {e}")
                        import traceback
                        traceback.print_exc()

                completed += 1
                if completed % 5 == 0:
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total experiments: {len(all_results)}")

    return df


def run_smoke(device, output_dir):
    """Smoke test: tiny matrix for CI/debug."""
    return run_experiment_matrix(
        datasets=["Cora"],
        noise_types=["feature_similar_cross_class"],
        noise_ratio=0.3,
        seeds=[0],
        prune_ratio=0.2,
        device=device,
        output_dir=output_dir,
    )


def run_full(device, output_dir):
    """Full experiment: 3 datasets, 3 noise types, 20 seeds."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        noise_types=["feature_similar_cross_class", "low_feature_similarity", "degree_aligned_random"],
        noise_ratio=0.3,
        seeds=list(range(20)),
        prune_ratio=0.2,
        device=device,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Ambiguity and Stability Evidence")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"experiments/2026-06-05-ambiguity-stability-evidence/logs/{args.mode}"

    if args.mode == "smoke":
        df = run_smoke(device, output_dir)
    elif args.mode == "full":
        df = run_full(device, output_dir)

    # Print summary
    if df is not None and len(df) > 0:
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)

        for phase in ["P0", "P1"]:
            phase_df = df[df["phase"] == phase]
            if len(phase_df) == 0:
                continue
            logger.info(f"\n--- {phase} Methods ---")
            summary = phase_df.groupby("method")["test_acc"].agg(["mean", "std", "count"])
            summary = summary.sort_values("mean", ascending=False)
            logger.info(summary.to_string())

        summary_path = os.path.join(output_dir, "summary.csv")
        df.groupby(["phase", "method"])["test_acc"].agg(["mean", "std", "count"]).to_csv(summary_path)
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
