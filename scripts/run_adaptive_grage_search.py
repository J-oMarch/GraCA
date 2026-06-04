#!/usr/bin/env python3
"""
Adaptive GraGE Method Search: Evaluate candidate methods for adaptive graph evolution.

Modes:
    smoke   — tiny matrix for CI/debug (1 dataset, 1 noise, 1 seed, 1 method)
    search  — candidate search matrix (Cora, CiteSeer × 2 noise types × 3 seeds)
    validate — validation matrix for best candidate (3 datasets × 3 noise × 5 seeds)
    selective_smoke — smoke test for Selective MCGC
    selective_search — no-leak Selective MCGC search matrix
    selective_validate — validation matrix for a selected Selective MCGC config

Usage:
    python scripts/run_adaptive_grage_search.py --mode smoke \
        --output_dir experiments/2026-06-04-adaptive-grage-search/logs/smoke

    python scripts/run_adaptive_grage_search.py --mode search \
        --output_dir experiments/2026-06-04-adaptive-grage-search/logs/search

    python scripts/run_adaptive_grage_search.py --mode validate \
        --output_dir experiments/2026-06-04-adaptive-grage-search/logs/validate
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
from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order
from src.grage.hybrid_score import compute_grage_hybrid_score
from src.grage.adaptive_score import (
    compute_faa_hybrid_score,
    compute_mcgc_score,
    compute_selective_mcgc_score,
    collect_multi_checkpoint_grads,
)
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
    feature_risk = 1.0 - cosine_sim
    return feature_risk


def compute_feature_similarity(x, edge_index, device):
    """Compute cosine similarity for each edge."""
    src = edge_index[0]
    dst = edge_index[1]
    cosine_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
    return cosine_sim


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
    noisy_edge_index = noisy_edge_index.to(device)
    E_noisy = noisy_edge_index.shape[1]

    start_time = time.time()

    method_name = method_config["name"]
    method_type = method_config["type"]
    method_diagnostics = {}

    # ─── Compute edge scores ───
    if method_type == "feature_only":
        edge_scores = compute_feature_risk(x, noisy_edge_index, device)

    elif method_type == "hybrid_baseline":
        # Current best GraGE-Hybrid-FO-posneg
        lambda_pos = method_config.get("lambda_pos", 0.1)
        lambda_neg = method_config.get("lambda_neg", 0.5)
        score_ratio = method_config.get("score_ratio", 0.3)

        num_classes = int(y.max().item()) + 1
        model = GCN(
            in_dim=x.shape[1], hidden_dim=64,
            out_dim=num_classes, num_layers=2, dropout=0.5,
        ).to(device)

        state_dict = train_model_for_grage(
            model, x, noisy_edge_index, y, train_mask, val_mask,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=200, patience=50, seed=seed,
        )
        model.load_state_dict(state_dict)

        support_mask, score_mask = split_train_support_score(
            train_mask, y, score_ratio=score_ratio, seed=seed
        )

        result = compute_edge_gate_influence_first_order(
            model=model, x=x, edge_index=noisy_edge_index, y=y,
            score_mask=score_mask, normalize=False, undirected=True,
            bad_edge_mask=bad_edge_mask,
        )
        dynamic_grad = result["raw_grad"]
        feature_risk = compute_feature_risk(x, noisy_edge_index, device)

        hybrid_result = compute_grage_hybrid_score(
            feature_risk=feature_risk,
            dynamic_grad=dynamic_grad,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            mode="feature_pos_neg",
            undirected=True,
            edge_index=noisy_edge_index,
            bad_edge_mask=bad_edge_mask,
        )
        edge_scores = hybrid_result["hybrid_score"]
        method_diagnostics = hybrid_result.get("diagnostics", {})

    elif method_type == "faa_hybrid":
        # Feature-Ambiguity-Adaptive Hybrid
        lambda_pos = method_config.get("lambda_pos", 0.1)
        lambda_neg = method_config.get("lambda_neg", 0.5)
        ambig_scale = method_config.get("ambig_scale", 1.0)
        base_alpha = method_config.get("base_alpha", 0.0)
        base_beta = method_config.get("base_beta", 0.0)
        score_ratio = method_config.get("score_ratio", 0.3)

        num_classes = int(y.max().item()) + 1
        model = GCN(
            in_dim=x.shape[1], hidden_dim=64,
            out_dim=num_classes, num_layers=2, dropout=0.5,
        ).to(device)

        state_dict = train_model_for_grage(
            model, x, noisy_edge_index, y, train_mask, val_mask,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=200, patience=50, seed=seed,
        )
        model.load_state_dict(state_dict)

        support_mask, score_mask = split_train_support_score(
            train_mask, y, score_ratio=score_ratio, seed=seed
        )

        result = compute_edge_gate_influence_first_order(
            model=model, x=x, edge_index=noisy_edge_index, y=y,
            score_mask=score_mask, normalize=False, undirected=True,
            bad_edge_mask=bad_edge_mask,
        )
        dynamic_grad = result["raw_grad"]
        feature_risk = compute_feature_risk(x, noisy_edge_index, device)
        feature_similarity = compute_feature_similarity(x, noisy_edge_index, device)

        faa_result = compute_faa_hybrid_score(
            feature_risk=feature_risk,
            dynamic_grad=dynamic_grad,
            feature_similarity=feature_similarity,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            ambig_scale=ambig_scale,
            base_alpha=base_alpha,
            base_beta=base_beta,
            undirected=True,
            edge_index=noisy_edge_index,
            bad_edge_mask=bad_edge_mask,
        )
        edge_scores = faa_result["hybrid_score"]
        method_diagnostics = faa_result.get("diagnostics", {})

    elif method_type == "mcgc":
        # Multi-Checkpoint Gradient Consistency
        lambda_pos = method_config.get("lambda_pos", 0.1)
        lambda_neg = method_config.get("lambda_neg", 0.5)
        consistency_weight = method_config.get("consistency_weight", 1.0)
        score_ratio = method_config.get("score_ratio", 0.3)
        checkpoint_fractions = method_config.get("checkpoint_fractions", [0.3, 0.5, 0.7, 0.9])
        total_epochs = method_config.get("total_epochs", 200)

        num_classes = int(y.max().item()) + 1

        def model_ctor():
            return GCN(
                in_dim=x.shape[1], hidden_dim=64,
                out_dim=num_classes, num_layers=2, dropout=0.5,
            )

        model = model_ctor().to(device)
        init_state_dict = train_model_for_grage(
            model, x, noisy_edge_index, y, train_mask, val_mask,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=200, patience=50, seed=seed,
        )

        support_mask, score_mask = split_train_support_score(
            train_mask, y, score_ratio=score_ratio, seed=seed
        )

        # Collect multi-checkpoint gradients
        checkpoint_grads = collect_multi_checkpoint_grads(
            model_ctor=model_ctor,
            init_state_dict=init_state_dict,
            x=x, edge_index=noisy_edge_index, y=y,
            train_mask=train_mask, score_mask=score_mask,
            checkpoint_fractions=checkpoint_fractions,
            total_epochs=total_epochs,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            undirected=True,
        )

        feature_risk = compute_feature_risk(x, noisy_edge_index, device)

        mcgc_result = compute_mcgc_score(
            feature_risk=feature_risk,
            checkpoint_grads=checkpoint_grads,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            consistency_weight=consistency_weight,
            undirected=True,
            edge_index=noisy_edge_index,
            bad_edge_mask=bad_edge_mask,
        )
        edge_scores = mcgc_result["hybrid_score"]
        method_diagnostics = mcgc_result.get("diagnostics", {})

    elif method_type == "selective_mcgc":
        # Selective Multi-Checkpoint Gradient Consistency with a no-leak
        # feature-regime gate.
        lambda_pos = method_config.get("lambda_pos", 0.1)
        lambda_neg = method_config.get("lambda_neg", 0.5)
        consistency_weight = method_config.get("consistency_weight", 1.0)
        score_ratio = method_config.get("score_ratio", 0.3)
        checkpoint_fractions = method_config.get("checkpoint_fractions", [0.3, 0.5, 0.7, 0.9])
        total_epochs = method_config.get("total_epochs", 200)
        tau = method_config.get("tau", None)
        tau_quantile = method_config.get("tau_quantile", 0.75)
        gate_type = method_config.get("gate_type", "hard")
        soft_k = method_config.get("soft_k", 20.0)
        checkpoint_control = method_config.get("checkpoint_control", "real")

        num_classes = int(y.max().item()) + 1

        def model_ctor():
            return GCN(
                in_dim=x.shape[1], hidden_dim=64,
                out_dim=num_classes, num_layers=2, dropout=0.5,
            )

        model = model_ctor().to(device)
        init_state_dict = train_model_for_grage(
            model, x, noisy_edge_index, y, train_mask, val_mask,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=200, patience=50, seed=seed,
        )

        support_mask, score_mask = split_train_support_score(
            train_mask, y, score_ratio=score_ratio, seed=seed
        )

        checkpoint_grads = collect_multi_checkpoint_grads(
            model_ctor=model_ctor,
            init_state_dict=init_state_dict,
            x=x, edge_index=noisy_edge_index, y=y,
            train_mask=train_mask, score_mask=score_mask,
            checkpoint_fractions=checkpoint_fractions,
            total_epochs=total_epochs,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            undirected=True,
        )

        if checkpoint_control == "shuffled":
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 7919)
            checkpoint_grads = [
                grad[torch.randperm(grad.numel(), device=device, generator=generator)]
                for grad in checkpoint_grads
            ]
        elif checkpoint_control == "frozen":
            checkpoint_grads = [checkpoint_grads[0].clone() for _ in checkpoint_grads]
        elif checkpoint_control != "real":
            raise ValueError(f"Unknown checkpoint_control: {checkpoint_control}")

        feature_risk = compute_feature_risk(x, noisy_edge_index, device)
        feature_similarity = compute_feature_similarity(x, noisy_edge_index, device)

        selective_result = compute_selective_mcgc_score(
            feature_risk=feature_risk,
            feature_similarity=feature_similarity,
            checkpoint_grads=checkpoint_grads,
            tau=tau,
            tau_quantile=tau_quantile,
            gate_type=gate_type,
            soft_k=soft_k,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            consistency_weight=consistency_weight,
            undirected=True,
            edge_index=noisy_edge_index,
            bad_edge_mask=bad_edge_mask,
        )
        edge_scores = selective_result["hybrid_score"]
        method_diagnostics = selective_result.get("diagnostics", {})

    else:
        raise ValueError(f"Unknown method type: {method_type}")

    # ─── Prune edges by score ───
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=noisy_edge_index,
        risk_score=edge_scores,
        num_nodes=x.shape[0],
        beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"],
        target_prune_ratio=prune_ratio,
    )

    # ─── Evaluate bad edge detection ───
    detection = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)

    # ─── Compute homophily ───
    homo_before = compute_edge_homophily(noisy_edge_index, y)
    homo_after = compute_edge_homophily(pruned_edge_index, y)

    # ─── Train downstream model ───
    downstream_results = train_downstream(
        model_name=downstream_model_name, data=data, edge_index=pruned_edge_index,
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
        "downstream_model": downstream_model_name,
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

    # Add method-specific hyperparameters
    for k, v in method_config.items():
        if k not in ("name", "type") and not k.startswith("_"):
            result[f"hp_{k}"] = v

    for k, v in method_diagnostics.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            result[f"diag_{k}"] = v

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Method configurations
# ═══════════════════════════════════════════════════════════════════════════════


def get_search_methods():
    """Return method configurations for the search phase."""
    methods = []

    # 1. Feature-only baseline
    methods.append({
        "name": "Feature-only",
        "type": "feature_only",
    })

    # 2. Current best hybrid baseline
    methods.append({
        "name": "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5",
        "type": "hybrid_baseline",
        "lambda_pos": 0.1, "lambda_neg": 0.5,
        "score_ratio": 0.3,
    })

    # 3. Random-Matched baseline (computed separately, placeholder here)
    # We'll handle Random-Matched inline in the experiment loop

    # ─── FAA-Hybrid variants ───
    for ambig_scale in [0.5, 1.0, 2.0, 3.0]:
        for lp, ln in [(0.1, 0.5), (0.25, 0.25)]:
            methods.append({
                "name": f"FAA-Hybrid-as{ambig_scale}-lp{lp}-ln{ln}",
                "type": "faa_hybrid",
                "lambda_pos": lp, "lambda_neg": ln,
                "ambig_scale": ambig_scale,
                "base_alpha": 0.0, "base_beta": 0.0,
                "score_ratio": 0.3,
            })

    # ─── MCGC variants ───
    for cw in [0.5, 1.0, 2.0, 3.0]:
        for lp, ln in [(0.1, 0.5), (0.25, 0.25)]:
            methods.append({
                "name": f"MCGC-cw{cw}-lp{lp}-ln{ln}",
                "type": "mcgc",
                "lambda_pos": lp, "lambda_neg": ln,
                "consistency_weight": cw,
                "score_ratio": 0.3,
                "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
                "total_epochs": 200,
            })

    return methods


def get_smoke_methods():
    """Return a tiny method set for smoke testing."""
    return [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "FAA-Hybrid-as1.0-lp0.1-ln0.5", "type": "faa_hybrid",
         "lambda_pos": 0.1, "lambda_neg": 0.5, "ambig_scale": 1.0,
         "base_alpha": 0.0, "base_beta": 0.0, "score_ratio": 0.3},
    ]


def get_selective_search_methods():
    """Return Selective MCGC method configurations for regime-gate search."""
    methods = [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "type": "hybrid_baseline",
         "lambda_pos": 0.1, "lambda_neg": 0.5, "score_ratio": 0.3},
        {"name": "MCGC-cw3.0-lp0.1-ln0.5", "type": "mcgc",
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0,
         "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 200},
    ]

    for gate_type in ["hard", "soft"]:
        for tau_quantile in [0.5, 0.75, 0.9]:
            for lp, ln in [(0.1, 0.5), (0.25, 0.25)]:
                methods.append({
                    "name": f"Selective-MCGC-{gate_type}-q{tau_quantile}-lp{lp}-ln{ln}",
                    "type": "selective_mcgc",
                    "gate_type": gate_type,
                    "tau_quantile": tau_quantile,
                    "soft_k": 20.0,
                    "lambda_pos": lp,
                    "lambda_neg": ln,
                    "consistency_weight": 3.0,
                    "score_ratio": 0.3,
                    "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
                    "total_epochs": 200,
                    "checkpoint_control": "real",
                })

    # Controls for attribution: if these match the real-gradient variant, the
    # paper cannot claim training-dynamics signal.
    methods.extend([
        {"name": "Selective-MCGC-hard-q0.75-shuffled", "type": "selective_mcgc",
         "gate_type": "hard", "tau_quantile": 0.75,
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0, "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 200, "checkpoint_control": "shuffled"},
        {"name": "Selective-MCGC-hard-q0.75-frozen", "type": "selective_mcgc",
         "gate_type": "hard", "tau_quantile": 0.75,
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0, "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 200, "checkpoint_control": "frozen"},
        {"name": "Selective-MCGC-zero-gate", "type": "selective_mcgc",
         "gate_type": "hard", "tau": 2.0,
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0, "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 200, "checkpoint_control": "real"},
    ])

    return methods


def get_selective_smoke_methods():
    """Return a tiny Selective MCGC method set for smoke testing."""
    return [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "MCGC-cw3.0-lp0.1-ln0.5", "type": "mcgc",
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0, "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 40},
        {"name": "Selective-MCGC-hard-q0.75-lp0.1-ln0.5", "type": "selective_mcgc",
         "gate_type": "hard", "tau_quantile": 0.75,
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0, "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 40, "checkpoint_control": "real"},
        {"name": "Selective-MCGC-soft-q0.75-lp0.1-ln0.5", "type": "selective_mcgc",
         "gate_type": "soft", "tau_quantile": 0.75, "soft_k": 20.0,
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0, "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 40, "checkpoint_control": "real"},
    ]


def get_validation_methods(best_candidate_config):
    """Return methods for validation: best candidate + baselines."""
    methods = [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "type": "hybrid_baseline",
         "lambda_pos": 0.1, "lambda_neg": 0.5, "score_ratio": 0.3},
    ]
    if best_candidate_config is not None:
        methods.append(best_candidate_config)
    return methods


def get_selective_validation_methods(best_candidate_config):
    """Return validation methods for Selective MCGC."""
    methods = [
        {"name": "Feature-only", "type": "feature_only"},
        {"name": "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "type": "hybrid_baseline",
         "lambda_pos": 0.1, "lambda_neg": 0.5, "score_ratio": 0.3},
        {"name": "MCGC-cw3.0-lp0.1-ln0.5", "type": "mcgc",
         "lambda_pos": 0.1, "lambda_neg": 0.5,
         "consistency_weight": 3.0,
         "score_ratio": 0.3,
         "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
         "total_epochs": 200},
    ]
    if best_candidate_config is not None:
        methods.append(best_candidate_config)
    else:
        methods.append({
            "name": "Selective-MCGC-hard-q0.75-lp0.1-ln0.5",
            "type": "selective_mcgc",
            "gate_type": "hard",
            "tau_quantile": 0.75,
            "lambda_pos": 0.1,
            "lambda_neg": 0.5,
            "consistency_weight": 3.0,
            "score_ratio": 0.3,
            "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
            "total_epochs": 200,
            "checkpoint_control": "real",
        })
    return methods


def run_random_matched_baseline(
    dataset_name, noise_type, noise_ratio, seed, downstream_model_name,
    prune_ratio, data, noisy_edge_index, bad_edge_mask, device, config
):
    """Run Random-Matched baseline: random pruning with matched budget."""
    set_seed(seed)
    x = data.x.to(device)
    y = data.y.to(device)
    noisy_edge_index = noisy_edge_index.to(device)
    E_noisy = noisy_edge_index.shape[1]

    start_time = time.time()

    # Random scores
    torch.manual_seed(seed)
    random_scores = torch.rand(E_noisy, device=device)

    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=noisy_edge_index,
        risk_score=random_scores,
        num_nodes=x.shape[0],
        beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"],
        target_prune_ratio=prune_ratio,
    )

    detection = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)
    homo_before = compute_edge_homophily(noisy_edge_index, y)
    homo_after = compute_edge_homophily(pruned_edge_index, y)

    downstream_results = train_downstream(
        model_name=downstream_model_name, data=data, edge_index=pruned_edge_index,
        config=config, num_features=x.shape[1], num_classes=int(y.max().item()) + 1,
        device=device, seed=seed,
    )

    runtime = time.time() - start_time

    return {
        "dataset": dataset_name,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "method": "Random-Matched",
        "method_type": "random_matched",
        "downstream_model": downstream_model_name,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment_matrix(
    datasets, noise_types, noise_ratio, seeds, downstream_model,
    prune_ratio, method_configs, device, output_dir, include_random_matched=True,
):
    """Run the full experiment matrix."""
    all_results = []
    config = DEFAULT_CONFIG.copy()

    total = len(datasets) * len(noise_types) * len(seeds) * (len(method_configs) + (1 if include_random_matched else 0))
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

        # Keep data on CPU for noise injection; move to device inside run_single_experiment
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

                data_noisy = data.clone()
                data_noisy.edge_index = noisy_edge_index

                # Run each method
                for method_config in method_configs:
                    try:
                        result = run_single_experiment(
                            dataset_name=dataset_name,
                            noise_type=noise_type,
                            noise_ratio=noise_ratio,
                            seed=seed,
                            method_config=method_config,
                            downstream_model_name=downstream_model,
                            prune_ratio=prune_ratio,
                            data=data_noisy,
                            noisy_edge_index=noisy_edge_index,
                            bad_edge_mask=bad_edge_mask,
                            device=device,
                            config=config,
                        )
                        all_results.append(result)
                        completed += 1

                        if completed % 5 == 0:
                            logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

                    except Exception as e:
                        logger.error(f"Failed: {dataset_name}/{noise_type}/seed{seed}/{method_config['name']}: {e}")
                        import traceback
                        traceback.print_exc()

                # Random-Matched baseline
                if include_random_matched:
                    try:
                        result = run_random_matched_baseline(
                            dataset_name=dataset_name,
                            noise_type=noise_type,
                            noise_ratio=noise_ratio,
                            seed=seed,
                            downstream_model_name=downstream_model,
                            prune_ratio=prune_ratio,
                            data=data_noisy,
                            noisy_edge_index=noisy_edge_index,
                            bad_edge_mask=bad_edge_mask,
                            device=device,
                            config=config,
                        )
                        all_results.append(result)
                        completed += 1
                    except Exception as e:
                        logger.error(f"Random-Matched failed: {dataset_name}/{noise_type}/seed{seed}: {e}")

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
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=get_smoke_methods(),
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_search(device, output_dir):
    """Search phase: evaluate all candidate methods."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer"],
        noise_types=["feature_similar_cross_class", "cross_class_oracle"],
        noise_ratio=0.3,
        seeds=[0, 1, 2],
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=get_search_methods(),
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_validate(device, output_dir, best_candidate_config):
    """Validation phase: evaluate best candidate on expanded matrix."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        noise_types=["feature_similar_cross_class", "low_feature_similarity", "degree_aligned_random"],
        noise_ratio=0.3,
        seeds=[0, 1, 2, 3, 4],
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=get_validation_methods(best_candidate_config),
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_selective_smoke(device, output_dir):
    """Selective MCGC smoke test."""
    return run_experiment_matrix(
        datasets=["Cora"],
        noise_types=["feature_similar_cross_class"],
        noise_ratio=0.3,
        seeds=[0],
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=get_selective_smoke_methods(),
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_selective_search(device, output_dir):
    """Selective MCGC no-leak regime-gate search."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer"],
        noise_types=["feature_similar_cross_class", "low_feature_similarity"],
        noise_ratio=0.3,
        seeds=[0, 1, 2],
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=get_selective_search_methods(),
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_selective_validate(device, output_dir, best_candidate_config):
    """Selective MCGC validation matrix."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        noise_types=["feature_similar_cross_class", "low_feature_similarity", "degree_aligned_random"],
        noise_ratio=0.3,
        seeds=[0, 1, 2, 3, 4],
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=get_selective_validation_methods(best_candidate_config),
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Adaptive GraGE Method Search")
    parser.add_argument("--mode", choices=[
        "smoke", "search", "validate",
        "selective_smoke", "selective_search", "selective_validate",
    ], default="smoke",
                        help="Experiment mode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--best_candidate", type=str, default=None,
                        help="JSON string of best candidate config (for validate mode)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"experiments/2026-06-04-adaptive-grage-search/logs/{args.mode}"

    if args.mode == "smoke":
        df = run_smoke(device, output_dir)
    elif args.mode == "search":
        df = run_search(device, output_dir)
    elif args.mode == "validate":
        best_config = None
        if args.best_candidate:
            best_config = json.loads(args.best_candidate)
        df = run_validate(device, output_dir, best_config)
    elif args.mode == "selective_smoke":
        df = run_selective_smoke(device, output_dir)
    elif args.mode == "selective_search":
        df = run_selective_search(device, output_dir)
    elif args.mode == "selective_validate":
        best_config = None
        if args.best_candidate:
            best_config = json.loads(args.best_candidate)
        df = run_selective_validate(device, output_dir, best_config)

    # Print summary
    if df is not None and len(df) > 0:
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)

        summary = df.groupby("method")["test_acc"].agg(["mean", "std", "count"])
        summary = summary.sort_values("mean", ascending=False)
        logger.info("\nMean test accuracy by method:")
        logger.info(summary.to_string())

        # Save summary
        summary_path = os.path.join(output_dir, "summary.csv")
        summary.to_csv(summary_path)
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
