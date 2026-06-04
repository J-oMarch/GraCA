#!/usr/bin/env python3
"""
GraGE Mechanism Diagnostics: Test whether training-dynamics signals contain
residual information beyond static feature similarity.

This script computes edge-level score tables and evaluates whether dynamic
edge-gate gradients improve bad-edge detection beyond what feature cosine
similarity alone can achieve.

Key diagnostics:
- Global ROC-AUC for each score against bad_edge_mask
- Precision/recall/F1 at fixed prune ratio
- Spearman correlation between feature risk and dynamic scores
- AUC within feature-similarity bins (especially most feature-similar quartile)
- Residual diagnostic: does dynamic signal detect bad edges after regressing out feature risk?
- Score shuffling ablation: does shuffled gradient lose the advantage?

Usage:
    python scripts/analyze_grage_mechanisms.py \
        --output_dir experiments/2026-06-04-dynamics-mechanism-diagnostics/logs \
        --datasets Cora CiteSeer PubMed \
        --noise_types feature_similar_cross_class cross_class_oracle low_feature_similarity \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --noise_ratio 0.3 \
        --prune_ratio 0.2
"""
import os
import sys
import argparse
import time
import json
import logging
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_dataset
from src.models.gcn import GCN
from src.eval.noise_injection import inject_noise
from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order
from src.grage.hybrid_score import compute_grage_hybrid_score, rank_normalize
from src.utils.mask_split import split_train_support_score
from src.utils.seed import set_seed

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Default training config ───────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "dataset": {"undirected": True, "root": "data/"},
    "training": {"lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 50},
}


def train_model_for_grage(model, x, edge_index, y, train_mask, val_mask,
                          lr=0.01, weight_decay=5e-4, epochs=200, patience=50, seed=42):
    """Train a GCN and return best state_dict."""
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


def compute_feature_risk(x, edge_index):
    """Compute 1 - cosine similarity for each edge."""
    src = edge_index[0]
    dst = edge_index[1]
    cosine_sim = F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)
    return 1.0 - cosine_sim


def compute_prune_metrics(score, bad_edge_mask, prune_ratio):
    """Compute precision/recall/F1 at a fixed prune ratio."""
    E = score.shape[0]
    n_prune = max(1, int(E * prune_ratio))
    # Higher score = more harmful => prune top-scoring edges
    sorted_idx = torch.argsort(score, descending=True)
    prune_mask = torch.zeros(E, dtype=torch.bool)
    prune_mask[sorted_idx[:n_prune]] = True

    bad = bad_edge_mask.bool()
    tp = (prune_mask & bad).sum().item()
    fp = (prune_mask & ~bad).sum().item()
    fn = (~prune_mask & bad).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_auc(y_true, y_score):
    """Compute AUC using sklearn if available, else manual."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


def compute_feature_bin_auc(feature_risk, bad_edge_mask, n_bins=4):
    """Compute AUC within feature-similarity bins.

    Bin edges are quantiles of feature_risk. The *lowest* feature_risk bin
    corresponds to the *most feature-similar* edges (hardest to detect).
    """
    fr = feature_risk.cpu().numpy()
    bad = bad_edge_mask.cpu().numpy().astype(int)

    # Quantile bins: bin 0 = lowest risk (most similar), bin n_bins-1 = highest risk
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(fr, quantiles)

    results = {}
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (fr >= lo) & (fr <= hi)
        else:
            mask = (fr >= lo) & (fr < hi)

        if mask.sum() < 5:
            results[f"bin_{i}"] = {"auc": 0.5, "n_edges": int(mask.sum()), "n_bad": int(bad[mask].sum())}
            continue

        bin_bad = bad[mask]
        if bin_bad.sum() == 0 or bin_bad.sum() == mask.sum():
            results[f"bin_{i}"] = {"auc": 0.5, "n_edges": int(mask.sum()), "n_bad": int(bin_bad.sum())}
            continue

        # AUC within this bin (using raw dynamic_grad, not hybrid)
        results[f"bin_{i}"] = {
            "auc": None,  # filled by caller with specific score
            "n_edges": int(mask.sum()),
            "n_bad": int(bin_bad.sum()),
            "feature_risk_lo": float(lo),
            "feature_risk_hi": float(hi),
        }

    return results


def compute_residual_auc(feature_risk, dynamic_score, bad_edge_mask):
    """Residual diagnostic: regress dynamic on feature risk, test residual."""
    fr = feature_risk.cpu().numpy()
    ds = dynamic_score.cpu().numpy()
    bad = bad_edge_mask.cpu().numpy().astype(int)

    # Rank-residual approach: compute rank of dynamic, regress on rank of feature
    rank_fr = sp_stats.rankdata(fr)
    rank_ds = sp_stats.rankdata(ds)

    # Linear regression: rank_ds = a * rank_fr + b + residual
    from numpy.linalg import lstsq
    A = np.column_stack([rank_fr, np.ones(len(rank_fr))])
    coef, _, _, _ = lstsq(A, rank_ds, rcond=None)
    residual = rank_ds - A @ coef

    # AUC of residual against bad_edge_mask
    try:
        from sklearn.metrics import roc_auc_score
        # Check direction: positive residual should indicate bad edge
        auc_pos = roc_auc_score(bad, residual)
        auc_neg = roc_auc_score(bad, -residual)
        residual_auc = max(auc_pos, auc_neg)
        direction = "positive" if auc_pos >= auc_neg else "negative"
    except ValueError:
        residual_auc = 0.5
        direction = "none"

    # Also compute Spearman correlation of residual with bad_edge_mask
    spearman_corr, spearman_p = sp_stats.spearmanr(residual, bad)

    return {
        "residual_auc": residual_auc,
        "residual_direction": direction,
        "residual_spearman": spearman_corr,
        "residual_spearman_p": spearman_p,
    }


def compute_frozen_channel_auc(feature_risk, dynamic_grad, bad_edge_mask, seed):
    """Frozen/inner-channel diagnostic: shuffle dynamic gradient to remove
    graph-channel information while preserving marginal distribution.

    This tests whether the advantage comes from the graph-channel edge-gradient
    or from a generic training effect. Both real and shuffled use the same
    undirected=False setting for a fair comparison.
    """
    rng = np.random.RandomState(seed + 1000)
    perm = rng.permutation(len(dynamic_grad))
    shuffled_grad = dynamic_grad[perm]

    bad = bad_edge_mask.cpu().numpy().astype(int)

    # Both use undirected=False for consistent comparison
    real_hybrid_result = compute_grage_hybrid_score(
        feature_risk=feature_risk,
        dynamic_grad=dynamic_grad,
        lambda_pos=0.1,
        lambda_neg=0.5,
        mode="feature_pos_neg",
        undirected=False,
    )
    real_hybrid = real_hybrid_result["hybrid_score"]

    shuffled_hybrid_result = compute_grage_hybrid_score(
        feature_risk=feature_risk,
        dynamic_grad=shuffled_grad,
        lambda_pos=0.1,
        lambda_neg=0.5,
        mode="feature_pos_neg",
        undirected=False,
    )
    shuffled_hybrid = shuffled_hybrid_result["hybrid_score"]

    from sklearn.metrics import roc_auc_score
    try:
        real_auc = roc_auc_score(bad, real_hybrid.cpu().numpy())
    except ValueError:
        real_auc = 0.5
    try:
        shuffled_auc = roc_auc_score(bad, shuffled_hybrid.cpu().numpy())
    except ValueError:
        shuffled_auc = 0.5

    return {
        "real_hybrid_auc": real_auc,
        "shuffled_hybrid_auc": shuffled_auc,
        "delta": real_auc - shuffled_auc,
    }


def run_single_case(dataset_name, noise_type, noise_ratio, seed, device, prune_ratio):
    """Run mechanism diagnostics for one dataset/noise/seed combination."""
    set_seed(seed)
    config = DEFAULT_CONFIG.copy()
    config["dataset"]["name"] = dataset_name

    # Load data
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    x = data.x
    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask

    # Inject noise
    noise_result = inject_noise(
        edge_index=data.edge_index.cpu(),
        num_nodes=data.num_nodes,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        x=data.x.cpu(),
        y=data.y.cpu(),
        train_mask=data.train_mask.cpu(),
        seed=seed,
    )
    noisy_edge_index = noise_result["noisy_edge_index"].to(device)
    bad_edge_mask = noise_result["bad_edge_mask"]
    E = noisy_edge_index.shape[1]

    # Train model
    model = GCN(
        in_dim=num_features, hidden_dim=64,
        out_dim=num_classes, num_layers=2, dropout=0.5,
    ).to(device)
    state_dict = train_model_for_grage(
        model, x, noisy_edge_index, y, train_mask, val_mask,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        seed=seed,
    )
    model.load_state_dict(state_dict)

    # Split train into support/score
    support_mask, score_mask = split_train_support_score(
        train_mask, y, score_ratio=0.3, seed=seed,
    )

    # Compute dynamic gradient (first-order)
    result = compute_edge_gate_influence_first_order(
        model=model, x=x, edge_index=noisy_edge_index, y=y,
        score_mask=score_mask, normalize=False, undirected=True,
        bad_edge_mask=bad_edge_mask,
    )
    raw_grad = result["raw_grad"]

    # Compute feature risk
    feature_risk = compute_feature_risk(x, noisy_edge_index)

    # ─── Score variants ─────────────────────────────────────────────────────
    pos_grad = F.relu(raw_grad)
    neg_grad = F.relu(-raw_grad)

    # Hybrid: best method from prior experiments (GraGE-Hybrid-FO-posneg-lp0.1-ln0.5)
    hybrid_result = compute_grage_hybrid_score(
        feature_risk=feature_risk, dynamic_grad=raw_grad,
        lambda_pos=0.1, lambda_neg=0.5,
        mode="feature_pos_neg", undirected=True,
        edge_index=noisy_edge_index, bad_edge_mask=bad_edge_mask,
    )
    hybrid_score = hybrid_result["hybrid_score"]

    # Shuffled-gradient hybrid
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(raw_grad))
    shuffled_grad = raw_grad[perm]
    shuffled_result = compute_grage_hybrid_score(
        feature_risk=feature_risk, dynamic_grad=shuffled_grad,
        lambda_pos=0.1, lambda_neg=0.5,
        mode="feature_pos_neg", undirected=True,
        edge_index=noisy_edge_index, bad_edge_mask=bad_edge_mask,
    )
    shuffled_hybrid = shuffled_result["hybrid_score"]

    # ─── Global AUC ─────────────────────────────────────────────────────────
    bad_np = bad_edge_mask.cpu().numpy().astype(int)
    fr_np = feature_risk.cpu().numpy()
    raw_np = raw_grad.cpu().numpy()
    pos_np = pos_grad.cpu().numpy()
    neg_np = neg_grad.cpu().numpy()
    hybrid_np = hybrid_score.cpu().numpy()
    shuffled_np = shuffled_hybrid.cpu().numpy()

    global_auc = {
        "feature_risk": compute_auc(bad_np, fr_np),
        "raw_grad": compute_auc(bad_np, raw_np),
        "pos_grad": compute_auc(bad_np, pos_np),
        "neg_grad": compute_auc(bad_np, neg_np),
        "hybrid": compute_auc(bad_np, hybrid_np),
        "shuffled_hybrid": compute_auc(bad_np, shuffled_np),
    }

    # ─── Prune metrics ──────────────────────────────────────────────────────
    prune_metrics = {}
    for name, score_tensor in [
        ("feature_risk", feature_risk),
        ("raw_grad", raw_grad),
        ("pos_grad", pos_grad),
        ("neg_grad", neg_grad),
        ("hybrid", hybrid_score),
        ("shuffled_hybrid", shuffled_hybrid),
    ]:
        prune_metrics[name] = compute_prune_metrics(score_tensor, bad_edge_mask, prune_ratio)

    # ─── Spearman correlation ───────────────────────────────────────────────
    corr_raw, p_raw = sp_stats.spearmanr(fr_np, raw_np)
    corr_pos, p_pos = sp_stats.spearmanr(fr_np, pos_np)
    corr_neg, p_neg = sp_stats.spearmanr(fr_np, neg_np)
    corr_hybrid, p_hybrid = sp_stats.spearmanr(fr_np, hybrid_np)

    correlations = {
        "feature_vs_raw_grad": {"corr": corr_raw, "p": p_raw},
        "feature_vs_pos_grad": {"corr": corr_pos, "p": p_pos},
        "feature_vs_neg_grad": {"corr": corr_neg, "p": p_neg},
        "feature_vs_hybrid": {"corr": corr_hybrid, "p": p_hybrid},
    }

    # ─── Feature-bin AUC ────────────────────────────────────────────────────
    # 4 bins: bin 0 = most feature-similar (lowest risk), bin 3 = least similar
    bin_auc = {}
    n_bins = 4
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(fr_np, quantiles)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (fr_np >= lo) & (fr_np <= hi)
        else:
            mask = (fr_np >= lo) & (fr_np < hi)

        bin_label = f"bin_{i}"
        n_edges = int(mask.sum())
        n_bad = int(bad_np[mask].sum())

        if n_edges < 5 or n_bad == 0 or n_bad == n_edges:
            bin_auc[bin_label] = {
                "feature_risk_auc": 0.5,
                "raw_grad_auc": 0.5,
                "hybrid_auc": 0.5,
                "shuffled_hybrid_auc": 0.5,
                "n_edges": n_edges,
                "n_bad": n_bad,
                "feature_risk_lo": float(lo),
                "feature_risk_hi": float(hi),
            }
            continue

        bin_auc[bin_label] = {
            "feature_risk_auc": compute_auc(bad_np[mask], fr_np[mask]),
            "raw_grad_auc": compute_auc(bad_np[mask], raw_np[mask]),
            "hybrid_auc": compute_auc(bad_np[mask], hybrid_np[mask]),
            "shuffled_hybrid_auc": compute_auc(bad_np[mask], shuffled_np[mask]),
            "n_edges": n_edges,
            "n_bad": n_bad,
            "feature_risk_lo": float(lo),
            "feature_risk_hi": float(hi),
        }

    # ─── Residual diagnostic ────────────────────────────────────────────────
    residual_raw = compute_residual_auc(feature_risk, raw_grad, bad_edge_mask)
    residual_pos = compute_residual_auc(feature_risk, pos_grad, bad_edge_mask)
    residual_hybrid = compute_residual_auc(feature_risk, hybrid_score, bad_edge_mask)

    # ─── Frozen/inner-channel diagnostic ────────────────────────────────────
    frozen_diag = compute_frozen_channel_auc(feature_risk, raw_grad, bad_edge_mask, seed)

    return {
        "dataset": dataset_name,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "E_noisy": E,
        "n_bad": int(bad_np.sum()),
        "global_auc": global_auc,
        "prune_metrics": prune_metrics,
        "correlations": correlations,
        "bin_auc": bin_auc,
        "residual": {
            "raw_grad": residual_raw,
            "pos_grad": residual_pos,
            "hybrid": residual_hybrid,
        },
        "frozen_channel": frozen_diag,
        "diagnostics": result["diagnostics"],
    }


def build_tables(all_results, output_dir):
    """Build paper-friendly CSV tables from all results."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # ─── 1. global_signal_auc.csv ───────────────────────────────────────────
    rows = []
    for r in all_results:
        for score_name, auc_val in r["global_auc"].items():
            rows.append({
                "dataset": r["dataset"],
                "noise_type": r["noise_type"],
                "seed": r["seed"],
                "score": score_name,
                "auc": auc_val,
            })
    df_global = pd.DataFrame(rows)
    df_global.to_csv(os.path.join(tables_dir, "global_signal_auc.csv"), index=False)

    # ─── 2. feature_bin_auc.csv ─────────────────────────────────────────────
    rows = []
    for r in all_results:
        for bin_label, bin_data in r["bin_auc"].items():
            for score_name in ["feature_risk_auc", "raw_grad_auc", "hybrid_auc", "shuffled_hybrid_auc"]:
                rows.append({
                    "dataset": r["dataset"],
                    "noise_type": r["noise_type"],
                    "seed": r["seed"],
                    "bin": bin_label,
                    "score": score_name.replace("_auc", ""),
                    "auc": bin_data[score_name],
                    "n_edges": bin_data["n_edges"],
                    "n_bad": bin_data["n_bad"],
                })
    df_bin = pd.DataFrame(rows)
    df_bin.to_csv(os.path.join(tables_dir, "feature_bin_auc.csv"), index=False)

    # ─── 3. shuffle_ablation.csv ────────────────────────────────────────────
    rows = []
    for r in all_results:
        rows.append({
            "dataset": r["dataset"],
            "noise_type": r["noise_type"],
            "seed": r["seed"],
            "real_hybrid_auc": r["frozen_channel"]["real_hybrid_auc"],
            "shuffled_hybrid_auc": r["frozen_channel"]["shuffled_hybrid_auc"],
            "delta": r["frozen_channel"]["delta"],
        })
    df_shuffle = pd.DataFrame(rows)
    df_shuffle.to_csv(os.path.join(tables_dir, "shuffle_ablation.csv"), index=False)

    # ─── 4. residual_signal.csv ─────────────────────────────────────────────
    rows = []
    for r in all_results:
        for score_name, resid_data in r["residual"].items():
            rows.append({
                "dataset": r["dataset"],
                "noise_type": r["noise_type"],
                "seed": r["seed"],
                "score": score_name,
                "residual_auc": resid_data["residual_auc"],
                "residual_direction": resid_data["residual_direction"],
                "residual_spearman": resid_data["residual_spearman"],
                "residual_spearman_p": resid_data["residual_spearman_p"],
            })
    df_resid = pd.DataFrame(rows)
    df_resid.to_csv(os.path.join(tables_dir, "residual_signal.csv"), index=False)

    # ─── 5. correlation_summary.csv ─────────────────────────────────────────
    rows = []
    for r in all_results:
        for pair_name, corr_data in r["correlations"].items():
            rows.append({
                "dataset": r["dataset"],
                "noise_type": r["noise_type"],
                "seed": r["seed"],
                "pair": pair_name,
                "spearman_corr": corr_data["corr"],
                "spearman_p": corr_data["p"],
            })
    df_corr = pd.DataFrame(rows)
    df_corr.to_csv(os.path.join(tables_dir, "correlation_summary.csv"), index=False)

    # ─── 6. prune_metrics.csv (bonus) ──────────────────────────────────────
    rows = []
    for r in all_results:
        for score_name, pm in r["prune_metrics"].items():
            rows.append({
                "dataset": r["dataset"],
                "noise_type": r["noise_type"],
                "seed": r["seed"],
                "score": score_name,
                **pm,
            })
    df_prune = pd.DataFrame(rows)
    df_prune.to_csv(os.path.join(tables_dir, "prune_metrics.csv"), index=False)

    return {
        "global_signal_auc": df_global,
        "feature_bin_auc": df_bin,
        "shuffle_ablation": df_shuffle,
        "residual_signal": df_resid,
        "correlation_summary": df_corr,
        "prune_metrics": df_prune,
    }


def build_figures(tables, output_dir):
    """Build paper-friendly plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figures")
        return []

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_paths = []

    # ─── Figure 1: Feature-bin AUC by noise type (feature_similar_cross_class) ──
    df_bin = tables["feature_bin_auc"]
    fscc = df_bin[df_bin["noise_type"] == "feature_similar_cross_class"]

    if len(fscc) > 0:
        # Average across seeds
        agg = fscc.groupby(["dataset", "bin", "score"])["auc"].mean().reset_index()

        datasets = sorted(agg["dataset"].unique())
        scores = ["feature_risk", "raw_grad", "hybrid", "shuffled_hybrid"]
        score_labels = ["Feature Risk", "Raw Gradient", "Hybrid (real)", "Hybrid (shuffled)"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
        if len(datasets) == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            ds_data = agg[agg["dataset"] == ds]
            x_pos = np.arange(4)
            width = 0.2

            for j, (score, label, color) in enumerate(zip(scores, score_labels, colors)):
                vals = []
                for bin_label in ["bin_0", "bin_1", "bin_2", "bin_3"]:
                    subset = ds_data[(ds_data["score"] == score) & (ds_data["bin"] == bin_label)]
                    vals.append(subset["auc"].values[0] if len(subset) > 0 else 0.5)
                ax.bar(x_pos + j * width, vals, width, label=label, color=color, alpha=0.8)

            ax.set_xlabel("Feature Similarity Bin")
            ax.set_ylabel("AUC")
            ax.set_title(f"{ds} — feature_similar_cross_class")
            ax.set_xticks(x_pos + 1.5 * width)
            ax.set_xticklabels(["Most\nSimilar", "Q2", "Q3", "Least\nSimilar"])
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylim(0.3, 1.0)
            ax.legend(fontsize=7)

        plt.tight_layout()
        path = os.path.join(figures_dir, "feature_bin_auc_fscc.png")
        plt.savefig(path, dpi=150)
        plt.close()
        fig_paths.append(path)

    # ─── Figure 2: Score distribution for feature_similar_cross_class ────────
    # (requires raw edge-level data, skip if not available)

    # ─── Figure 3: Shuffle ablation delta by noise type ─────────────────────
    df_shuffle = tables["shuffle_ablation"]
    if len(df_shuffle) > 0:
        agg = df_shuffle.groupby(["dataset", "noise_type"])["delta"].agg(["mean", "std"]).reset_index()

        noise_types = sorted(agg["noise_type"].unique())
        datasets = sorted(agg["dataset"].unique())

        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(noise_types))
        width = 0.8 / len(datasets)

        for i, ds in enumerate(datasets):
            ds_data = agg[agg["dataset"] == ds]
            vals = [ds_data[ds_data["noise_type"] == nt]["mean"].values[0]
                    if len(ds_data[ds_data["noise_type"] == nt]) > 0 else 0
                    for nt in noise_types]
            errs = [ds_data[ds_data["noise_type"] == nt]["std"].values[0]
                    if len(ds_data[ds_data["noise_type"] == nt]) > 0 else 0
                    for nt in noise_types]
            ax.bar(x_pos + i * width, vals, width, label=ds, yerr=errs, alpha=0.8, capsize=3)

        ax.set_xlabel("Noise Type")
        ax.set_ylabel("AUC Delta (real - shuffled)")
        ax.set_title("Shuffle Ablation: Real vs Shuffled Dynamic Gradient")
        ax.set_xticks(x_pos + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(noise_types, rotation=30, ha="right")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        path = os.path.join(figures_dir, "shuffle_ablation_delta.png")
        plt.savefig(path, dpi=150)
        plt.close()
        fig_paths.append(path)

    return fig_paths


def build_metrics_json(all_results, tables, fig_paths, output_dir):
    """Build the metrics.json summary."""
    # Compute summary statistics
    df_global = tables["global_signal_auc"]
    df_bin = tables["feature_bin_auc"]
    df_shuffle = tables["shuffle_ablation"]
    df_resid = tables["residual_signal"]

    # Mean dynamic-feature Spearman correlation
    df_corr = tables["correlation_summary"]
    raw_grad_corr = df_corr[df_corr["pair"] == "feature_vs_raw_grad"]["spearman_corr"].mean()

    # Mean residual AUC for hybrid
    hybrid_resid = df_resid[df_resid["score"] == "hybrid"]["residual_auc"].mean()

    # Best signal in feature-similar bin (bin_0 = most similar)
    fscc_bin0 = df_bin[(df_bin["noise_type"] == "feature_similar_cross_class") & (df_bin["bin"] == "bin_0")]
    if len(fscc_bin0) > 0:
        best_bin0 = fscc_bin0.groupby("score")["auc"].mean().idxmax()
        best_bin0_auc = fscc_bin0.groupby("score")["auc"].mean().max()
    else:
        best_bin0 = "unknown"
        best_bin0_auc = 0.5

    # Feature-similar bin real vs shuffled delta
    fscc_shuffle = df_shuffle[df_shuffle["noise_type"] == "feature_similar_cross_class"]
    fscc_delta = fscc_shuffle["delta"].mean() if len(fscc_shuffle) > 0 else 0.0

    # Inner channel diagnostic
    inner_diag = "shuffled gradient loses advantage" if fscc_delta > 0.01 else "no clear difference"

    # Failure modes
    failure_modes = []
    if hybrid_resid < 0.52:
        failure_modes.append("weak residual signal after feature risk regression")
    if raw_grad_corr > 0.8:
        failure_modes.append("dynamic gradient highly correlated with feature risk")
    if fscc_delta < 0.005:
        failure_modes.append("shuffle ablation shows no clear advantage in feature_similar_cross_class")

    metrics = {
        "exp_id": "2026-06-04-dynamics-mechanism-diagnostics",
        "status": "completed",
        "residual_dynamic_signal_supported": bool(hybrid_resid > 0.52 and fscc_delta > 0.005),
        "feature_similar_bin_real_vs_shuffled_delta": round(float(fscc_delta), 4),
        "best_signal_in_feature_similar_bin": best_bin0,
        "mean_residual_auc": round(float(hybrid_resid), 4),
        "mean_dynamic_feature_spearman": round(float(raw_grad_corr), 4),
        "inner_channel_diagnostic": inner_diag,
        "failure_modes": failure_modes,
        "num_cases": len(all_results),
        "tables": list(tables.keys()),
        "figures": [os.path.basename(p) for p in fig_paths],
        "notes": "",
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GraGE Mechanism Diagnostics")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/2026-06-04-dynamics-mechanism-diagnostics/logs")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--noise_types", nargs="+", default=[
        "feature_similar_cross_class", "cross_class_oracle", "low_feature_similarity",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--prune_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    total = len(args.datasets) * len(args.noise_types) * len(args.seeds)
    completed = 0
    start_time = time.time()

    for dataset_name in args.datasets:
        for noise_type in args.noise_types:
            for seed in args.seeds:
                case_start = time.time()
                try:
                    result = run_single_case(
                        dataset_name=dataset_name,
                        noise_type=noise_type,
                        noise_ratio=args.noise_ratio,
                        seed=seed,
                        device=device,
                        prune_ratio=args.prune_ratio,
                    )
                    all_results.append(result)
                    elapsed = time.time() - case_start
                    completed += 1
                    logger.info(
                        f"[{completed}/{total}] {dataset_name}/{noise_type}/seed{seed} "
                        f"done in {elapsed:.1f}s — "
                        f"global_auc: feature={result['global_auc']['feature_risk']:.3f}, "
                        f"hybrid={result['global_auc']['hybrid']:.3f}, "
                        f"bin0_hybrid={result['bin_auc'].get('bin_0', {}).get('hybrid_auc', 0):.3f}"
                    )
                except Exception as e:
                    completed += 1
                    logger.error(f"[{completed}/{total}] FAILED {dataset_name}/{noise_type}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()

    total_time = time.time() - start_time
    logger.info(f"\nAll cases done in {total_time:.1f}s. {len(all_results)}/{total} succeeded.")

    # Build tables
    logger.info("Building tables...")
    tables = build_tables(all_results, args.output_dir)

    # Build figures
    logger.info("Building figures...")
    fig_paths = build_figures(tables, args.output_dir)

    # Build metrics.json
    logger.info("Building metrics.json...")
    metrics = build_metrics_json(all_results, tables, fig_paths, args.output_dir)

    # Write metrics.json
    exp_dir = os.path.dirname(args.output_dir)  # experiments/2026-06-04-.../
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Write summary to stdout
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total cases: {len(all_results)}")
    logger.info(f"Mean residual AUC (hybrid): {metrics['mean_residual_auc']}")
    logger.info(f"Mean dynamic-feature Spearman: {metrics['mean_dynamic_feature_spearman']}")
    logger.info(f"Feature-similar bin real vs shuffled delta: {metrics['feature_similar_bin_real_vs_shuffled_delta']}")
    logger.info(f"Best signal in feature-similar bin: {metrics['best_signal_in_feature_similar_bin']}")
    logger.info(f"Inner channel diagnostic: {metrics['inner_channel_diagnostic']}")
    logger.info(f"Residual dynamic signal supported: {metrics['residual_dynamic_signal_supported']}")
    logger.info(f"Failure modes: {metrics['failure_modes']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
