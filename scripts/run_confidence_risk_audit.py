#!/usr/bin/env python3
"""
Confidence Risk Audit for StabilityResidual-GraGE.

Diagnoses whether prediction-stability-derived edge evidence remains useful after
controlling for confidence/uncertainty. Exports edge-level diagnostics and runs
confidence-controlled analyses.

Usage:
    python scripts/run_confidence_risk_audit.py --mode smoke \
        --output_dir experiments/2026-06-05-confidence-risk-audit/logs/smoke

    python scripts/run_confidence_risk_audit.py --mode full \
        --output_dir experiments/2026-06-05-confidence-risk-audit/logs/full
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
    collect_multi_view_predictions,
    compute_node_stability,
    stability_to_edge_score,
    residualize_stability_score,
    rank_normalize,
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

EXP_ID = "2026-06-05-confidence-risk-audit"


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
# Edge-level diagnostics export
# ═══════════════════════════════════════════════════════════════════════════════


def compute_edge_level_diagnostics(
    dataset_name, seed, x, y, noisy_edge_index, bad_edge_mask,
    feature_risk, feature_similarity, stability_result,
    confidence_edge_result, fo_prune_mask, sr_prune_mask, conf_prune_mask,
    bucket_labels, device,
):
    """Export per-edge diagnostics for the confidence risk audit.

    Returns a list of dicts, one per undirected edge pair.
    """
    E = noisy_edge_index.shape[1]
    src = noisy_edge_index[0].cpu()
    dst = noisy_edge_index[1].cpu()

    # Group by undirected pair
    edge_key_to_indices = defaultdict(list)
    for i in range(E):
        u, v = src[i].item(), dst[i].item()
        key = (min(u, v), max(u, v))
        edge_key_to_indices[key].append(i)

    # Extract scores
    raw_stability = stability_result["raw_edge_score"]
    residual = stability_result["residual"]
    final_score = stability_result["edge_score"]
    conf_edge_score = confidence_edge_result["edge_score"]

    rows = []
    for key, indices in edge_key_to_indices.items():
        u, v = key
        idx = indices[0]  # use first direction for scores (averaged undirected)

        row = {
            "dataset": dataset_name,
            "seed": seed,
            "edge_src": u,
            "edge_dst": v,
            "feature_risk": float(feature_risk[idx].cpu()),
            "feature_similarity": float(feature_similarity[idx].cpu()),
            "confidence_edge_score": float(conf_edge_score[idx].cpu()),
            "raw_stability_score": float(raw_stability[idx].cpu()),
            "residualized_stability": float(residual[idx].cpu()),
            "stability_residual_final": float(final_score[idx].cpu()),
            "fo_prune": bool(fo_prune_mask[idx].cpu()),
            "sr_prune": bool(sr_prune_mask[idx].cpu()),
            "conf_prune": bool(conf_prune_mask[idx].cpu()),
            "bad_edge": bool(bad_edge_mask[idx].cpu()),
            "ambiguity_bucket": int(bucket_labels[idx].cpu()),
        }
        rows.append(row)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence-controlled analyses
# ═══════════════════════════════════════════════════════════════════════════════


def run_confidence_bucket_analysis(edge_df, num_buckets=5):
    """Split edges into confidence quantiles and report AUC within each bucket.

    Confidence bucket is defined by the confidence_edge_score quantile.
    Within each bucket, we report AUC for feature_risk, raw_stability,
    residualized_stability, and stability_residual_final.
    """
    from sklearn.metrics import roc_auc_score

    df = edge_df.copy()
    df["conf_bucket"] = pd.qcut(
        df["confidence_edge_score"], q=num_buckets, labels=False, duplicates="drop"
    )

    results = []
    score_cols = [
        "feature_risk", "raw_stability_score", "residualized_stability",
        "stability_residual_final", "confidence_edge_score",
    ]

    for bucket_id in sorted(df["conf_bucket"].unique()):
        bucket_df = df[df["conf_bucket"] == bucket_id]
        bad = bucket_df["bad_edge"].values
        if len(np.unique(bad)) < 2:
            continue

        row = {
            "conf_bucket": int(bucket_id),
            "count": len(bucket_df),
            "bad_count": int(bad.sum()),
            "conf_score_range_min": float(bucket_df["confidence_edge_score"].min()),
            "conf_score_range_max": float(bucket_df["confidence_edge_score"].max()),
        }

        for col in score_cols:
            scores = bucket_df[col].values
            try:
                auc = float(roc_auc_score(bad, scores))
            except ValueError:
                auc = 0.5
            row[f"{col}_auc"] = auc

        # Residual stability AUC delta over confidence AUC within this bucket
        row["resid_auc_minus_conf_auc"] = (
            row["stability_residual_final_auc"] - row["confidence_edge_score_auc"]
        )

        results.append(row)

    return pd.DataFrame(results)


def run_high_ambiguity_confidence_bucket_analysis(edge_df, num_buckets=5):
    """Same as confidence bucket analysis but restricted to High-Ambiguity edges."""
    high_df = edge_df[edge_df["ambiguity_bucket"] == 2].copy()
    if high_df.empty:
        return pd.DataFrame()
    return run_confidence_bucket_analysis(high_df, num_buckets=num_buckets)


def run_matched_analysis(edge_df, num_strata=10):
    """Confidence-matched analysis: compare stability vs confidence pruning.

    Stratify edges by confidence score, then within each stratum compare
    the bad-edge rate among edges pruned by stability vs confidence.
    """
    from sklearn.metrics import roc_auc_score

    df = edge_df.copy()
    df["conf_stratum"] = pd.qcut(
        df["confidence_edge_score"], q=num_strata, labels=False, duplicates="drop"
    )

    stratum_results = []
    for stratum_id in sorted(df["conf_stratum"].unique()):
        stratum_df = df[df["conf_stratum"] == stratum_id]
        if stratum_df.empty:
            continue

        # Bad-edge rate among edges pruned by each method
        sr_pruned = stratum_df[stratum_df["sr_prune"]]
        conf_pruned = stratum_df[stratum_df["conf_prune"]]

        sr_bad_rate = float(sr_pruned["bad_edge"].mean()) if len(sr_pruned) > 0 else np.nan
        conf_bad_rate = float(conf_pruned["bad_edge"].mean()) if len(conf_pruned) > 0 else np.nan

        stratum_results.append({
            "conf_stratum": int(stratum_id),
            "count": len(stratum_df),
            "bad_count": int(stratum_df["bad_edge"].sum()),
            "sr_pruned_count": len(sr_pruned),
            "conf_pruned_count": len(conf_pruned),
            "sr_bad_edge_rate": sr_bad_rate,
            "conf_bad_edge_rate": conf_bad_rate,
            "sr_minus_conf_bad_rate": (
                sr_bad_rate - conf_bad_rate
                if not (np.isnan(sr_bad_rate) or np.isnan(conf_bad_rate))
                else np.nan
            ),
        })

    return pd.DataFrame(stratum_results)


def run_partial_correlation_diagnostic(edge_df):
    """Estimate whether residual stability predicts bad edges after controlling
    for feature_risk and confidence.

    Uses logistic regression: bad_edge ~ feature_risk + confidence + residual_stability.
    Reports coefficient and significance for residual_stability.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = edge_df.dropna(subset=[
        "bad_edge", "feature_risk", "confidence_edge_score", "residualized_stability"
    ]).copy()

    if df.empty or df["bad_edge"].nunique() < 2:
        return {"status": "insufficient_data"}

    X = df[["feature_risk", "confidence_edge_score", "residualized_stability"]].values
    y = df["bad_edge"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full model: feature_risk + confidence + residual
    model_full = LogisticRegression(max_iter=1000, solver="lbfgs")
    model_full.fit(X_scaled, y)
    full_auc = float(roc_auc_score_safe(y, model_full.predict_proba(X_scaled)[:, 1]))

    # Reduced model: feature_risk + confidence only
    X_reduced = X_scaled[:, :2]
    model_reduced = LogisticRegression(max_iter=1000, solver="lbfgs")
    model_reduced.fit(X_reduced, y)
    reduced_auc = float(roc_auc_score_safe(y, model_reduced.predict_proba(X_reduced)[:, 1]))

    # Coefficient for residual_stability in full model
    resid_coef = float(model_full.coef_[0][2])
    resid_coef_per_unit = resid_coef / max(scaler.scale_[2], 1e-8)

    return {
        "status": "ok",
        "full_model_auc": full_auc,
        "reduced_model_auc": reduced_auc,
        "auc_improvement_from_residual": full_auc - reduced_auc,
        "residual_coef_standardized": resid_coef,
        "residual_coef_per_unit": resid_coef_per_unit,
        "n_edges": len(df),
        "n_bad": int(y.sum()),
    }


def run_global_auc_analysis(edge_df):
    """Report global AUC for each score type."""
    from sklearn.metrics import roc_auc_score

    bad = edge_df["bad_edge"].values
    if len(np.unique(bad)) < 2:
        return {}

    score_cols = [
        "feature_risk", "confidence_edge_score", "raw_stability_score",
        "residualized_stability", "stability_residual_final",
    ]

    results = {}
    for col in score_cols:
        scores = edge_df[col].values
        try:
            auc = float(roc_auc_score(bad, scores))
        except ValueError:
            auc = 0.5
        results[f"{col}_auc"] = auc

    return results


def roc_auc_score_safe(y_true, y_score):
    """Safe AUC computation."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Single seed runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_seed(
    dataset_name, seed, data, noisy_edge_index, bad_edge_mask,
    feature_risk, feature_similarity, x, y, train_mask, val_mask,
    device, config, prune_ratio,
):
    """Run all diagnostics for a single dataset/seed combination."""
    set_seed(seed)
    num_features = x.shape[1]
    num_classes = int(y.max().item()) + 1

    def model_ctor():
        return GCN(in_dim=num_features, hidden_dim=64,
                   out_dim=num_classes, num_layers=2, dropout=0.5)

    # Train model
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

    # Compute StabilityResidual
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

    # Collect multi-view predictions for Confidence control
    predictions = collect_multi_view_predictions(
        model_ctor=model_ctor, init_state_dict=state_dict,
        x=x, edge_index=noisy_edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask,
        num_views=5, edge_dropout_rates=[0.0, 0.10, 0.15, 0.20, 0.30],
        total_epochs=200, lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"], patience=50,
    )

    # Compute confidence edge score
    confidence_edge_result = compute_confidence_edge_score(
        predictions=predictions, feature_risk=feature_risk,
        edge_index=noisy_edge_index, feature_similarity=feature_similarity,
        undirected=True, device=device,
    )

    # Compute pruning masks
    # Feature-only
    fo_scores = feature_risk.clone()
    _, fo_prune_mask, _ = prune_graph(
        edge_index=noisy_edge_index, risk_score=fo_scores,
        num_nodes=x.shape[0], beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"], target_prune_ratio=prune_ratio,
    )

    # StabilityResidual
    _, sr_prune_mask, _ = prune_graph(
        edge_index=noisy_edge_index, risk_score=stability_result["edge_score"],
        num_nodes=x.shape[0], beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"], target_prune_ratio=prune_ratio,
    )

    # Confidence-control
    _, conf_prune_mask, _ = prune_graph(
        edge_index=noisy_edge_index, risk_score=confidence_edge_result["edge_score"],
        num_nodes=x.shape[0], beta=config["pruning"]["beta"],
        min_degree=config["pruning"]["min_degree"], target_prune_ratio=prune_ratio,
    )

    # Compute ambiguity buckets (feature-only, no label leakage)
    bucket_info = compute_ambiguity_buckets(
        feature_risk=feature_risk, prune_mask=fo_prune_mask, num_buckets=3,
    )

    # Edge-level diagnostics
    edge_rows = compute_edge_level_diagnostics(
        dataset_name=dataset_name, seed=seed,
        x=x, y=y, noisy_edge_index=noisy_edge_index, bad_edge_mask=bad_edge_mask,
        feature_risk=feature_risk, feature_similarity=feature_similarity,
        stability_result=stability_result,
        confidence_edge_result=confidence_edge_result,
        fo_prune_mask=fo_prune_mask, sr_prune_mask=sr_prune_mask,
        conf_prune_mask=conf_prune_mask,
        bucket_labels=bucket_info["bucket_labels"], device=device,
    )

    return edge_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Paired accuracy (supporting context only)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_paired_accuracy(edge_df, config, data, device, prune_ratio):
    """Compute paired downstream accuracy for Feature-only, StabilityResidual,
    and Confidence-control. Supporting context only."""
    # This is expensive; we'll compute it from the edge diagnostics instead
    # by just checking the prune mask statistics
    return None  # Skip for now; edge-quality AUC is the primary evidence


# ═══════════════════════════════════════════════════════════════════════════════
# Output contract
# ═══════════════════════════════════════════════════════════════════════════════


def write_output_contract(edge_df, output_dir, mode, global_auc, conf_bucket_df,
                          high_conf_bucket_df, matched_df, partial_corr):
    """Write result.md, metrics.json, and summary CSVs."""
    exp_dir = Path("experiments") / EXP_ID
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save edge diagnostics
    mode_dir = logs_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(mode_dir / "edge_diagnostics.csv", index=False)

    # Save confidence bucket summary
    conf_bucket_df.to_csv(mode_dir / "confidence_bucket_summary.csv", index=False)

    # Save AUC summary
    auc_rows = []
    for k, v in global_auc.items():
        auc_rows.append({"scope": "global", "score_type": k.replace("_auc", ""), "auc": v})

    if not conf_bucket_df.empty:
        for _, row in conf_bucket_df.iterrows():
            for col in conf_bucket_df.columns:
                if col.endswith("_auc"):
                    auc_rows.append({
                        "scope": f"conf_bucket_{int(row['conf_bucket'])}",
                        "score_type": col.replace("_auc", ""),
                        "auc": row[col],
                    })

    if not high_conf_bucket_df.empty:
        for _, row in high_conf_bucket_df.iterrows():
            for col in high_conf_bucket_df.columns:
                if col.endswith("_auc"):
                    auc_rows.append({
                        "scope": f"high_ambig_conf_bucket_{int(row['conf_bucket'])}",
                        "score_type": col.replace("_auc", ""),
                        "auc": row[col],
                    })

    auc_summary_df = pd.DataFrame(auc_rows)
    auc_summary_df.to_csv(mode_dir / "auc_summary.csv", index=False)

    # Save matched analysis
    if not matched_df.empty:
        matched_df.to_csv(mode_dir / "matched_analysis.csv", index=False)

    # Compute key metrics
    global_resid_auc = global_auc.get("stability_residual_final_auc", 0.5)
    global_conf_auc = global_auc.get("confidence_edge_score_auc", 0.5)
    global_raw_auc = global_auc.get("raw_stability_score_auc", 0.5)
    global_feature_auc = global_auc.get("feature_risk_auc", 0.5)

    # Same-confidence-bucket AUC delta (mean across buckets)
    if not conf_bucket_df.empty and "resid_auc_minus_conf_auc" in conf_bucket_df.columns:
        same_conf_auc_delta = float(conf_bucket_df["resid_auc_minus_conf_auc"].mean())
    else:
        same_conf_auc_delta = 0.0

    # High-ambiguity same-confidence-bucket AUC delta
    if not high_conf_bucket_df.empty and "resid_auc_minus_conf_auc" in high_conf_bucket_df.columns:
        high_ambig_auc_delta = float(high_conf_bucket_df["resid_auc_minus_conf_auc"].mean())
    else:
        high_ambig_auc_delta = 0.0

    # Matched bad-edge rate delta
    if not matched_df.empty and "sr_minus_conf_bad_rate" in matched_df.columns:
        matched_delta = float(matched_df["sr_minus_conf_bad_rate"].mean())
    else:
        matched_delta = 0.0

    # Partial correlation
    partial_resid_coef = partial_corr.get("residual_coef_standardized", 0.0) if partial_corr.get("status") == "ok" else 0.0
    partial_auc_improvement = partial_corr.get("auc_improvement_from_residual", 0.0) if partial_corr.get("status") == "ok" else 0.0

    # Decision rules
    stability_not_confidence_only = (
        global_resid_auc > global_conf_auc + 0.005
        or same_conf_auc_delta > 0.005
        or partial_resid_coef > 0.05
    )
    confidence_risk_reduced = stability_not_confidence_only and same_conf_auc_delta > 0.0

    failure_modes = []
    if not stability_not_confidence_only:
        failure_modes.append("Stability signal is not distinguishable from confidence")
    if same_conf_auc_delta <= 0:
        failure_modes.append("Residual stability does not improve AUC within confidence buckets")
    if global_resid_auc <= global_conf_auc:
        failure_modes.append("Global StabilityResidual AUC <= Confidence AUC")
    if partial_resid_coef <= 0:
        failure_modes.append("Partial correlation: residual stability coefficient is non-positive")

    claim_recommendation = (
        "support_stability_beyond_confidence"
        if confidence_risk_reduced and not failure_modes
        else "acknowledge_confidence_proximity"
    )

    # Stability vs confidence delta (from existing P1 evidence)
    stability_vs_confidence_delta_pp = 0.31  # from P1 paired test

    metrics = {
        "exp_id": EXP_ID,
        "status": "completed" if mode == "full" else "smoke_completed",
        "confidence_risk_reduced": bool(confidence_risk_reduced),
        "stability_not_confidence_only": bool(stability_not_confidence_only),
        "stability_vs_confidence_delta_pp": stability_vs_confidence_delta_pp,
        "matched_bad_edge_rate_delta": matched_delta,
        "same_confidence_bucket_auc_delta": same_conf_auc_delta,
        "high_ambiguity_confidence_bucket_auc_delta": high_ambig_auc_delta,
        "residual_stability_auc_after_confidence_control": partial_auc_improvement,
        "global_auc": {
            "feature_risk": global_feature_auc,
            "confidence": global_conf_auc,
            "raw_stability": global_raw_auc,
            "residualized_stability": global_auc.get("residualized_stability_auc", 0.5),
            "stability_residual_final": global_resid_auc,
        },
        "partial_correlation": partial_corr,
        "claim_recommendation": claim_recommendation,
        "failure_modes": failure_modes,
        "num_edges": len(edge_df),
        "num_datasets": edge_df["dataset"].nunique(),
        "num_seeds": edge_df["seed"].nunique(),
    }

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Write result.md
    result_md = f"""# Confidence Risk Audit Result

## Executive Summary

- Mode: `{mode}`.
- Edges analyzed: `{len(edge_df)}`.
- Datasets: `{list(edge_df["dataset"].unique())}`.
- Seeds: `{edge_df["seed"].nunique()}`.

## 1. Is Stability Distinguishable from Confidence?

**Global AUC comparison** (higher = better bad-edge detection):

| Score | AUC |
| --- | --- |
| Feature Risk | {global_feature_auc:.4f} |
| Confidence | {global_conf_auc:.4f} |
| Raw Stability | {global_raw_auc:.4f} |
| Residualized Stability | {global_auc.get("residualized_stability_auc", 0.5):.4f} |
| StabilityResidual Final | {global_resid_auc:.4f} |

StabilityResidual AUC delta over Confidence: **{global_resid_auc - global_conf_auc:+.4f}**.

**Same-confidence-bucket AUC delta**: {same_conf_auc_delta:+.4f}
(Positive = residual stability improves bad-edge detection within confidence strata.)

**Partial correlation** (residual stability coefficient after controlling for
feature risk and confidence): {partial_resid_coef:+.4f}
(AUC improvement from adding residual: {partial_auc_improvement:+.4f})

**Conclusion**: {"Stability provides signal beyond confidence." if stability_not_confidence_only else "Stability signal is largely explained by confidence."}

## 2. Does the Distinction Hold in High-Ambiguity FSCC Edges?

High-ambiguity same-confidence-bucket AUC delta: {high_ambig_auc_delta:+.4f}.

{"Residual stability still improves detection in High-Ambiguity edges after confidence control." if high_ambig_auc_delta > 0.005 else "The distinction weakens in High-Ambiguity edges."}

## 3. Confidence-Bucket AUC Analysis

{_markdown_table(conf_bucket_df, ["conf_bucket", "count", "bad_count", "feature_risk_auc", "confidence_edge_score_auc", "stability_residual_final_auc", "resid_auc_minus_conf_auc"]) if not conf_bucket_df.empty else "_No data._"}

## 4. Confidence-Matched Bad-Edge Rate

{_markdown_table(matched_df, ["conf_stratum", "count", "bad_count", "sr_bad_edge_rate", "conf_bad_edge_rate", "sr_minus_conf_bad_rate"]) if not matched_df.empty else "_No data._"}

## 5. What Should the Paper Claim?

**Claim recommendation**: `{claim_recommendation}`.

{"StabilityResidual provides edge-quality evidence beyond what confidence alone captures, particularly in feature-ambiguous regions. The paper can maintain the current claim with an explicit discussion of the confidence relationship." if confidence_risk_reduced else "The paper should acknowledge that confidence is a strong related signal and frame stability as a complementary rather than independent source of edge evidence."}

## 6. What Should Be Admitted in Limitations?

- Feature+Confidence is close to Feature+Stability in the P1 paired test ({stability_vs_confidence_delta_pp:+.2f} pp, p=0.20).
- {"Residual stability adds modest but detectable signal beyond confidence within confidence strata." if same_conf_auc_delta > 0 else "Within confidence strata, residual stability does not clearly add signal."}
- {"The partial correlation analysis confirms residual stability contributes after controlling for feature risk and confidence." if partial_resid_coef > 0 else "The partial correlation does not confirm independent residual stability contribution."}

## 7. Output Files

- Edge diagnostics: `{mode_dir / 'edge_diagnostics.csv'}`
- Confidence bucket summary: `{mode_dir / 'confidence_bucket_summary.csv'}`
- AUC summary: `{mode_dir / 'auc_summary.csv'}`
- Matched analysis: `{mode_dir / 'matched_analysis.csv'}`
"""

    (exp_dir / "result.md").write_text(result_md)

    # Failure analysis
    failure_md = f"""# Failure Analysis: Confidence Risk Audit

## Failure Modes

{chr(10).join(f'- {m}' for m in failure_modes) if failure_modes else '- None under the configured decision rules.'}

## Reviewer-Risk Interpretation

- If confidence AUC is close to or exceeds StabilityResidual AUC, reviewers will
  reasonably frame stability as uncertainty under another name.
- If same-confidence-bucket AUC delta is negative or near zero, the residual
  stability signal does not survive confidence control.
- If partial correlation coefficient is non-positive, residual stability does not
  predict bad edges after controlling for feature risk and confidence.

## Required Paper Updates

{"The current claim can be maintained with explicit confidence discussion." if not failure_modes else "The claim must be revised to acknowledge confidence proximity."}

Update:
- `paper_draft/limitations.md`
- `paper_draft/rebuttal_risks.md`
"""
    (exp_dir / "failure_analysis.md").write_text(failure_md)


def _markdown_table(frame, columns, float_digits=4):
    """Render a compact markdown table."""
    if frame is None or frame.empty:
        return "_No data._"

    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.{float_digits}f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment_matrix(datasets, seeds, device, output_dir, mode="full"):
    """Run the full confidence risk audit."""
    config = DEFAULT_CONFIG.copy()
    noise_type = "feature_similar_cross_class"
    noise_ratio = 0.3
    prune_ratio = 0.2

    all_edge_rows = []
    total = len(datasets) * len(seeds)
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

        for seed in seeds:
            logger.info(f"Seed: {seed}")
            set_seed(seed)

            # Inject noise
            noise_result = inject_noise(
                edge_index=data.edge_index, num_nodes=data.num_nodes,
                noise_type=noise_type, noise_ratio=noise_ratio,
                x=data.x, y=data.y, train_mask=data.train_mask, seed=seed,
            )
            noisy_edge_index = noise_result["noisy_edge_index"]
            bad_edge_mask = noise_result["bad_edge_mask"]

            x = data.x.to(device)
            y = data.y.to(device)
            train_mask = data.train_mask.to(device)
            val_mask = data.val_mask.to(device)
            noisy_edge_index = noisy_edge_index.to(device)
            bad_edge_mask = bad_edge_mask.to(device)

            feature_risk = compute_feature_risk(x, noisy_edge_index, device)
            feature_similarity = compute_feature_similarity(x, noisy_edge_index, device)

            try:
                edge_rows = run_single_seed(
                    dataset_name=dataset_name, seed=seed,
                    data=data, noisy_edge_index=noisy_edge_index,
                    bad_edge_mask=bad_edge_mask,
                    feature_risk=feature_risk, feature_similarity=feature_similarity,
                    x=x, y=y, train_mask=train_mask, val_mask=val_mask,
                    device=device, config=config, prune_ratio=prune_ratio,
                )
                all_edge_rows.extend(edge_rows)
            except Exception as e:
                logger.error(f"Failed: {dataset_name}/seed{seed}: {e}")
                import traceback
                traceback.print_exc()

            completed += 1
            logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    # Build DataFrame
    edge_df = pd.DataFrame(all_edge_rows)

    if edge_df.empty:
        logger.error("No edge rows produced!")
        return edge_df

    # Save edge diagnostics
    os.makedirs(output_dir, exist_ok=True)
    edge_df.to_csv(os.path.join(output_dir, "edge_diagnostics.csv"), index=False)
    logger.info(f"Edge diagnostics: {len(edge_df)} rows")

    # Run analyses
    logger.info("Running global AUC analysis...")
    global_auc = run_global_auc_analysis(edge_df)

    logger.info("Running confidence bucket analysis...")
    conf_bucket_df = run_confidence_bucket_analysis(edge_df, num_buckets=5)

    logger.info("Running High-Ambiguity confidence bucket analysis...")
    high_conf_bucket_df = run_high_ambiguity_confidence_bucket_analysis(edge_df, num_buckets=5)

    logger.info("Running confidence-matched analysis...")
    matched_df = run_matched_analysis(edge_df, num_strata=10)

    logger.info("Running partial correlation diagnostic...")
    partial_corr = run_partial_correlation_diagnostic(edge_df)

    # Save summaries
    conf_bucket_df.to_csv(os.path.join(output_dir, "confidence_bucket_summary.csv"), index=False)
    if not high_conf_bucket_df.empty:
        high_conf_bucket_df.to_csv(os.path.join(output_dir, "high_ambig_confidence_bucket_summary.csv"), index=False)
    matched_df.to_csv(os.path.join(output_dir, "matched_analysis.csv"), index=False)

    # Write output contract
    write_output_contract(
        edge_df=edge_df, output_dir=output_dir, mode=mode,
        global_auc=global_auc, conf_bucket_df=conf_bucket_df,
        high_conf_bucket_df=high_conf_bucket_df, matched_df=matched_df,
        partial_corr=partial_corr,
    )

    return edge_df


def run_smoke(device, output_dir):
    """Smoke test: single dataset, single seed."""
    return run_experiment_matrix(
        datasets=["Cora"],
        seeds=[0],
        device=device,
        output_dir=output_dir,
        mode="smoke",
    )


def run_full(device, output_dir):
    """Full diagnostic: 3 datasets, 10 seeds."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        seeds=list(range(10)),
        device=device,
        output_dir=output_dir,
        mode="full",
    )


def main():
    parser = argparse.ArgumentParser(description="Confidence Risk Audit")
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
        output_dir = f"experiments/{EXP_ID}/logs/{args.mode}"

    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "smoke":
        df = run_smoke(device, output_dir)
    elif args.mode == "full":
        df = run_full(device, output_dir)

    if df is not None and len(df) > 0:
        logger.info(f"\nTotal edges: {len(df)}")
        logger.info(f"Datasets: {df['dataset'].unique().tolist()}")
        logger.info(f"Seeds: {df['seed'].nunique()}")
    else:
        logger.error("No results produced!")


if __name__ == "__main__":
    main()
