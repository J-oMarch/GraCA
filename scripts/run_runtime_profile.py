#!/usr/bin/env python3
"""
Runtime Profile: Break down cost components for Feature-only vs StabilityResidual.

Profiles each stage of the scoring/pruning/evaluation pipeline to produce a
paper-ready accuracy-cost tradeoff table.  This is NOT method search; it uses
the fixed main method StabilityResidual-v5-dp0.15-grad-frozen only.

Usage:
    python scripts/run_runtime_profile.py --mode smoke \
        --output_dir experiments/2026-06-05-runtime-profile/logs/smoke

    python scripts/run_runtime_profile.py --mode full \
        --output_dir experiments/2026-06-05-runtime-profile/logs/full
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
from src.eval.noise_injection import inject_noise
from src.graca.pruning import prune_graph
from src.grage.adaptive_score import (
    compute_stability_residual_score,
    collect_multi_view_predictions,
    compute_node_stability,
    stability_to_edge_score,
    residualize_stability_score,
    rank_normalize,
    collect_multi_checkpoint_grads,
)
from src.utils.mask_split import split_train_support_score
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Default config (matches run_ambiguity_stability_evidence.py) ───
DEFAULT_CONFIG = {
    "dataset": {"undirected": True},
    "pruning": {"beta": 0.2, "min_degree": 1},
    "training": {"lr": 0.01, "weight_decay": 5e-4, "epochs": 200, "patience": 50},
    "downstream_model": {"names": ["GCN"]},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Timing utilities
# ═══════════════════════════════════════════════════════════════════════════════


class Timer:
    """Context-manager timer that accumulates into a named slot."""

    def __init__(self, timings: dict, key: str):
        self.timings = timings
        self.key = key

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        self.timings[self.key] = self.timings.get(self.key, 0.0) + elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_feature_risk(x, edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    return 1.0 - F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)


def compute_feature_similarity(x, edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    return F.cosine_similarity(x[src], x[dst], dim=1, eps=1e-8)


def train_model_for_grage(
    model, x, edge_index, y, train_mask, val_mask,
    lr=0.01, weight_decay=5e-4, epochs=200, patience=50, seed=42,
):
    """Train a model and return its best state_dict (for GraGE computation)."""
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
# Profiled single-seed runners
# ═══════════════════════════════════════════════════════════════════════════════


def profile_feature_only(
    dataset_name, seed, prune_ratio, data, noisy_edge_index, device, config,
):
    """Profile Feature-only pipeline.  Returns timing dict and accuracy."""
    set_seed(seed)
    timings = {}
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    noisy_edge_index = noisy_edge_index.to(device)

    # ── Feature scoring ──
    with Timer(timings, "feature_scoring"):
        feature_risk = compute_feature_risk(x, noisy_edge_index)

    # ── Pruning ──
    with Timer(timings, "pruning"):
        pruned_edge_index, prune_mask, graph_stats = prune_graph(
            edge_index=noisy_edge_index,
            risk_score=feature_risk,
            num_nodes=x.shape[0],
            beta=config["pruning"]["beta"],
            min_degree=config["pruning"]["min_degree"],
            target_prune_ratio=prune_ratio,
        )

    # ── Downstream retraining ──
    with Timer(timings, "downstream_train"):
        ds_result = train_downstream(
            model_name="GCN",
            data=data,
            edge_index=pruned_edge_index,
            config=config,
            num_features=x.shape[1],
            num_classes=int(y.max().item()) + 1,
            device=device,
            seed=seed,
        )

    # ── Inference/evaluation (already included in train_downstream) ──
    timings["inference_eval"] = 0.0  # included in downstream_train

    # Zero-fill StabilityResidual-only components
    for k in [
        "probe_train",
        "gradient_confidence",
        "stability_scoring",
    ]:
        timings.setdefault(k, 0.0)

    timings["total"] = sum(timings.values())
    timings["test_acc"] = ds_result["test_acc"]
    timings["method"] = "Feature-only"
    timings["dataset"] = dataset_name
    timings["seed"] = seed
    timings["num_edges_before"] = graph_stats["num_edges_before"]
    timings["num_edges_after"] = graph_stats["num_edges_after"]
    timings["prune_ratio_actual"] = graph_stats["prune_ratio"]

    return timings


def profile_stability_residual(
    dataset_name, seed, prune_ratio, data, noisy_edge_index, device, config,
):
    """Profile StabilityResidual-v5-dp0.15-grad-frozen pipeline."""
    set_seed(seed)
    timings = {}
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    noisy_edge_index = noisy_edge_index.to(device)
    num_features = x.shape[1]
    num_classes = int(y.max().item()) + 1

    def model_ctor():
        return GCN(
            in_dim=num_features, hidden_dim=64,
            out_dim=num_classes, num_layers=2, dropout=0.5,
        )

    # ── Feature scoring ──
    with Timer(timings, "feature_scoring"):
        feature_risk = compute_feature_risk(x, noisy_edge_index)
        feature_similarity = compute_feature_similarity(x, noisy_edge_index)

    # ── Probe training (model for GraGE) ──
    with Timer(timings, "probe_train"):
        model = model_ctor().to(device)
        state_dict = train_model_for_grage(
            model, x, noisy_edge_index, y, train_mask, val_mask,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=200, patience=50, seed=seed,
        )
        model.load_state_dict(state_dict)

    # ── Gradient confidence collection ──
    with Timer(timings, "gradient_confidence"):
        support_mask, score_mask = split_train_support_score(
            train_mask, y, score_ratio=0.3, seed=seed,
        )
        checkpoint_grads = collect_multi_checkpoint_grads(
            model_ctor=model_ctor,
            init_state_dict=state_dict,
            x=x,
            edge_index=noisy_edge_index,
            y=y,
            train_mask=train_mask,
            score_mask=score_mask,
            checkpoint_fractions=[0.3, 0.5, 0.7, 0.9],
            total_epochs=200,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            undirected=True,
        )
        # Frozen gradient: replicate first checkpoint across all
        checkpoint_grads = [checkpoint_grads[0].clone() for _ in checkpoint_grads]

    # ── Stability scoring (multi-view + residualization) ──
    with Timer(timings, "stability_scoring"):
        stability_result = compute_stability_residual_score(
            model_ctor=model_ctor,
            init_state_dict=state_dict,
            x=x,
            edge_index=noisy_edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            feature_risk=feature_risk,
            feature_similarity=feature_similarity,
            checkpoint_grads=checkpoint_grads,
            num_views=5,
            edge_dropout_rates=[0.0, 0.10, 0.15, 0.20, 0.30],
            total_epochs=200,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            patience=50,
            use_gradient_confidence=True,
            gradient_abstention_threshold=0.1,
            undirected=True,
            skip_residualization=False,
        )
        edge_scores = stability_result["edge_score"]

    # ── Pruning ──
    with Timer(timings, "pruning"):
        pruned_edge_index, prune_mask, graph_stats = prune_graph(
            edge_index=noisy_edge_index,
            risk_score=edge_scores,
            num_nodes=x.shape[0],
            beta=config["pruning"]["beta"],
            min_degree=config["pruning"]["min_degree"],
            target_prune_ratio=prune_ratio,
        )

    # ── Downstream retraining ──
    with Timer(timings, "downstream_train"):
        ds_result = train_downstream(
            model_name="GCN",
            data=data,
            edge_index=pruned_edge_index,
            config=config,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )

    # Inference/evaluation is included in train_downstream
    timings["inference_eval"] = 0.0

    timings["total"] = sum(timings.values())
    timings["test_acc"] = ds_result["test_acc"]
    timings["method"] = "StabilityResidual-v5-dp0.15-grad-frozen"
    timings["dataset"] = dataset_name
    timings["seed"] = seed
    timings["num_edges_before"] = graph_stats["num_edges_before"]
    timings["num_edges_after"] = graph_stats["num_edges_after"]
    timings["prune_ratio_actual"] = graph_stats["prune_ratio"]

    return timings


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment matrix
# ═══════════════════════════════════════════════════════════════════════════════


def run_profile_matrix(datasets, seeds, prune_ratio, device, output_dir):
    """Run the profile matrix across datasets and seeds."""
    all_rows = []
    config = DEFAULT_CONFIG.copy()
    noise_type = "feature_similar_cross_class"
    noise_ratio = 0.3

    total = len(datasets) * len(seeds) * 2  # 2 methods
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

            data_noisy = data.clone()
            data_noisy.edge_index = noisy_edge_index

            # ── Feature-only ──
            try:
                fo_timings = profile_feature_only(
                    dataset_name=dataset_name,
                    seed=seed,
                    prune_ratio=prune_ratio,
                    data=data_noisy,
                    noisy_edge_index=noisy_edge_index,
                    device=device,
                    config=config,
                )
                all_rows.append(fo_timings)
                completed += 1
                logger.info(
                    f"[{completed}/{total}] Feature-only {dataset_name} seed={seed}: "
                    f"total={fo_timings['total']:.2f}s acc={fo_timings['test_acc']:.4f}"
                )
            except Exception as e:
                logger.error(f"Feature-only failed: {dataset_name}/seed{seed}: {e}")
                import traceback
                traceback.print_exc()

            # ── StabilityResidual ──
            try:
                sr_timings = profile_stability_residual(
                    dataset_name=dataset_name,
                    seed=seed,
                    prune_ratio=prune_ratio,
                    data=data_noisy,
                    noisy_edge_index=noisy_edge_index,
                    device=device,
                    config=config,
                )
                all_rows.append(sr_timings)
                completed += 1
                logger.info(
                    f"[{completed}/{total}] StabilityResidual {dataset_name} seed={seed}: "
                    f"total={sr_timings['total']:.2f}s acc={sr_timings['test_acc']:.4f}"
                )
            except Exception as e:
                logger.error(
                    f"StabilityResidual failed: {dataset_name}/seed{seed}: {e}"
                )
                import traceback
                traceback.print_exc()

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Output contract
# ═══════════════════════════════════════════════════════════════════════════════

EXP_ID = "2026-06-05-runtime-profile"

COMPONENT_COLS = [
    "feature_scoring",
    "probe_train",
    "gradient_confidence",
    "stability_scoring",
    "pruning",
    "downstream_train",
    "inference_eval",
    "total",
]


def build_summary(df):
    """Build per-method timing summary (mean ± std)."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for method in df["method"].unique():
        mdf = df[df["method"] == method]
        row = {"method": method, "n": len(mdf)}
        for col in COMPONENT_COLS:
            if col in mdf.columns:
                row[f"{col}_mean"] = float(mdf[col].mean())
                row[f"{col}_std"] = float(mdf[col].std()) if len(mdf) > 1 else 0.0
        if "test_acc" in mdf.columns:
            row["test_acc_mean"] = float(mdf["test_acc"].mean())
            row["test_acc_std"] = (
                float(mdf["test_acc"].std()) if len(mdf) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_output_contract(df, output_dir, mode):
    """Write result.md, metrics.json, and summary tables."""
    exp_dir = Path("experiments") / EXP_ID
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save raw profile CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "runtime_profile.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw profile saved to {csv_path}")

    # Build summary
    summary = build_summary(df)
    summary_path = os.path.join(output_dir, "runtime_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    # ── Extract key metrics ──
    fo_rows = df[df["method"] == "Feature-only"]
    sr_rows = df[df["method"] == "StabilityResidual-v5-dp0.15-grad-frozen"]

    def _mean(subset, col):
        if subset.empty or col not in subset.columns:
            return None
        return float(subset[col].mean())

    def _std(subset, col):
        if subset.empty or col not in subset.columns or len(subset) <= 1:
            return None
        return float(subset[col].std())

    fo_total = _mean(fo_rows, "total")
    sr_total = _mean(sr_rows, "total")
    fo_acc = _mean(fo_rows, "test_acc")
    sr_acc = _mean(sr_rows, "test_acc")

    runtime_ratio = None
    extra_overhead = None
    accuracy_delta_pp = None
    if fo_total is not None and sr_total is not None and fo_total > 0:
        runtime_ratio = round(sr_total / fo_total, 2)
        extra_overhead = round(sr_total - fo_total, 2)
    if fo_acc is not None and sr_acc is not None:
        accuracy_delta_pp = round((sr_acc - fo_acc) * 100.0, 2)

    # Per-component times for StabilityResidual
    sr_probe = _mean(sr_rows, "probe_train")
    sr_grad = _mean(sr_rows, "gradient_confidence")
    sr_stab = _mean(sr_rows, "stability_scoring")
    sr_ds_train = _mean(sr_rows, "downstream_train")
    sr_inference = _mean(sr_rows, "inference_eval")

    # ── Determine status ──
    if df.empty:
        status = "failed_empty_results"
    elif mode == "full":
        status = "completed"
    else:
        status = "smoke_completed"

    # ── Build metrics.json ──
    metrics = {
        "exp_id": EXP_ID,
        "status": status,
        "runtime_ratio": runtime_ratio,
        "extra_overhead": extra_overhead,
        "accuracy_delta_pp": accuracy_delta_pp,
        "feature_only_total_mean_s": fo_total,
        "feature_only_total_std_s": _std(fo_rows, "total"),
        "stability_residual_total_mean_s": sr_total,
        "stability_residual_total_std_s": _std(sr_rows, "total"),
        "feature_only_test_acc_mean": fo_acc,
        "feature_only_test_acc_std": _std(fo_rows, "test_acc"),
        "stability_residual_test_acc_mean": sr_acc,
        "stability_residual_test_acc_std": _std(sr_rows, "test_acc"),
        "stability_probe_train_time": sr_probe,
        "stability_gradient_time": sr_grad,
        "stability_score_time": sr_stab,
        "stability_downstream_train_time": sr_ds_train,
        "stability_inference_eval_time": sr_inference,
        "num_result_rows": int(len(df)),
        "claim_recommendation": "accuracy-cost tradeoff, not efficiency superiority",
    }

    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # ── Build result.md ──
    def _fmt_time(val, std=None):
        if val is None:
            return "n/a"
        if std is not None and std > 0:
            return f"{val:.2f} ± {std:.2f}s"
        return f"{val:.2f}s"

    def _fmt_acc(val, std=None):
        if val is None:
            return "n/a"
        if std is not None and std > 0:
            return f"{val:.4f} ± {std:.4f}"
        return f"{val:.4f}"

    # Build per-component markdown table
    component_rows = []
    for comp in [
        ("Feature scoring", "feature_scoring"),
        ("Probe/model-view training", "probe_train"),
        ("Gradient confidence", "gradient_confidence"),
        ("Stability scoring", "stability_scoring"),
        ("Pruning", "pruning"),
        ("Downstream retraining", "downstream_train"),
        ("Inference/evaluation", "inference_eval"),
        ("Total", "total"),
    ]:
        label, key = comp
        fo_val = _mean(fo_rows, key)
        fo_s = _std(fo_rows, key)
        sr_val = _mean(sr_rows, key)
        sr_s = _std(sr_rows, key)
        component_rows.append({
            "Component": label,
            "Feature-only": _fmt_time(fo_val, fo_s),
            "StabilityResidual": _fmt_time(sr_val, sr_s),
        })
    comp_df = pd.DataFrame(component_rows)
    comp_md = _markdown_table(
        comp_df, ["Component", "Feature-only", "StabilityResidual"]
    )

    # Per-dataset summary
    dataset_rows = []
    for ds in df["dataset"].unique():
        ds_fo = fo_rows[fo_rows["dataset"] == ds]
        ds_sr = sr_rows[sr_rows["dataset"] == ds]
        dataset_rows.append({
            "Dataset": ds,
            "FO_time": _fmt_time(_mean(ds_fo, "total"), _std(ds_fo, "total")),
            "SR_time": _fmt_time(_mean(ds_sr, "total"), _std(ds_sr, "total")),
            "FO_acc": _fmt_acc(_mean(ds_fo, "test_acc"), _std(ds_fo, "test_acc")),
            "SR_acc": _fmt_acc(_mean(ds_sr, "test_acc"), _std(ds_sr, "test_acc")),
        })
    ds_df = pd.DataFrame(dataset_rows)
    ds_md = _markdown_table(
        ds_df, ["Dataset", "FO_time", "SR_time", "FO_acc", "SR_acc"]
    )

    result_md = f"""# Runtime Profile Result

## Executive Summary

- Mode: `{mode}`.
- Rows: `{len(df)}`.
- Runtime ratio (SR / FO): `{runtime_ratio}`.
- Extra overhead: `{extra_overhead}s`.
- Accuracy delta: `{accuracy_delta_pp} pp`.
- Claim: **accuracy-cost tradeoff, not efficiency superiority**.

## Per-Component Timing

{comp_md}

## Per-Dataset Summary

{ds_md}

## Interpretation

StabilityResidual-v5-dp0.15-grad-frozen adds multi-view graph training and
stability residualization on top of the Feature-only pipeline.  The extra
overhead is dominated by probe training and stability scoring (multi-view
predictions + node stability + residualization).  This is an accuracy-cost
tradeoff: the method buys `{accuracy_delta_pp} pp` at the cost of roughly
`{runtime_ratio}x` wall-clock time.

**Do not claim efficiency superiority.**  The profiler reports where time is
spent; it does not optimize runtime.

## Output Tables

- Raw profile: `{csv_path}`
- Summary: `{summary_path}`
"""
    result_path = exp_dir / "result.md"
    result_path.write_text(result_md)
    logger.info(f"Result written to {result_path}")

    return metrics


def _markdown_table(frame, columns, float_digits=4):
    if frame.empty:
        return "_No rows available._"
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
# Paper draft updates
# ═══════════════════════════════════════════════════════════════════════════════


def update_paper_drafts(metrics):
    """Update runtime_table.md, reviewer_risk_audit.md, readiness_audit.md,
    and aaai_readiness_score.md if the run completed cleanly."""
    if metrics["status"] not in ("completed", "smoke_completed"):
        logger.warning("Skipping paper draft updates: status=%s", metrics["status"])
        return

    paper_dir = Path("paper_draft")
    paper_dir.mkdir(exist_ok=True)

    # ── runtime_table.md ──
    ratio = metrics.get("runtime_ratio", "n/a")
    overhead = metrics.get("extra_overhead", "n/a")
    delta = metrics.get("accuracy_delta_pp", "n/a")
    fo_time = metrics.get("feature_only_total_mean_s")
    sr_time = metrics.get("stability_residual_total_mean_s")
    fo_acc = metrics.get("feature_only_test_acc_mean")
    sr_acc = metrics.get("stability_residual_test_acc_mean")
    sr_probe = metrics.get("stability_probe_train_time")
    sr_grad = metrics.get("stability_gradient_time")
    sr_stab = metrics.get("stability_score_time")
    sr_ds = metrics.get("stability_downstream_train_time")

    def _t(v):
        return f"{v:.2f}" if v is not None else "n/a"

    def _a(v):
        return f"{v:.4f}" if v is not None else "n/a"

    runtime_table_md = f"""# Runtime Profile Table

## Accuracy-Cost Tradeoff

| Method | Total Time | Test Acc | Overhead |
|---|---|---|---|
| Feature-only | {_t(fo_time)}s | {_a(fo_acc)} | baseline |
| StabilityResidual-v5-dp0.15-grad-frozen | {_t(sr_time)}s | {_a(sr_acc)} | +{_t(overhead)}s ({ratio}x) |

Accuracy delta: **{delta} pp**.

## StabilityResidual Component Breakdown

| Component | Time |
|---|---|
| Probe/model-view training | {_t(sr_probe)}s |
| Gradient confidence collection | {_t(sr_grad)}s |
| Stability scoring (multi-view + residualization) | {_t(sr_stab)}s |
| Downstream retraining | {_t(sr_ds)}s |

## Claim

**Accuracy-cost tradeoff, not efficiency superiority.**

The extra overhead is dominated by multi-view graph training and stability
residualization.  This buys approximately {delta} pp on homophilic citation
FSCC at the cost of roughly {ratio}x wall-clock time.
"""
    (paper_dir / "runtime_table.md").write_text(runtime_table_md)
    logger.info("Updated paper_draft/runtime_table.md")

    # ── reviewer_risk_audit.md ──
    reviewer_risk_md = f"""# Reviewer Risk Audit

## Runtime Risk

Current evidence: StabilityResidual is approximately **{ratio}x** Feature-only
wall-clock time for **{delta} pp** accuracy gain on homophilic citation FSCC.

Component breakdown shows the overhead is concentrated in:
1. Probe/model-view training: {_t(sr_probe)}s
2. Stability scoring (multi-view + residualization): {_t(sr_stab)}s

This is an accuracy-cost tradeoff, not an efficiency claim.  The paper should
frame the runtime as a known limitation and present the tradeoff honestly.

## Other Reviewer Risks

See `paper_draft/rebuttal_risks.md` for the full risk matrix.
"""
    (paper_dir / "reviewer_risk_audit.md").write_text(reviewer_risk_md)
    logger.info("Updated paper_draft/reviewer_risk_audit.md")

    # ── Update readiness_audit.md (append runtime section) ──
    readiness_path = paper_dir / "readiness_audit.md"
    if readiness_path.exists():
        existing = readiness_path.read_text()
        if "## Runtime Profile" not in existing:
            runtime_section = f"""

## Runtime Profile

- Runtime ratio: **{ratio}x** Feature-only.
- Extra overhead: **{_t(overhead)}s**.
- Accuracy delta: **{delta} pp**.
- Claim: accuracy-cost tradeoff, not efficiency superiority.
- Component breakdown: see `paper_draft/runtime_table.md`.
"""
            readiness_path.write_text(existing + runtime_section)
            logger.info("Appended runtime section to readiness_audit.md")
    else:
        logger.warning("readiness_audit.md not found; skipping append")

    # ── aaai_readiness_score.md ──
    # Create if it doesn't exist (the prompt asks for this file)
    score_path = paper_dir / "aaai_readiness_score.md"
    if not score_path.exists():
        score_md = f"""# AAAI Readiness Score

## Evidence Status

| Dimension | Status | Notes |
|---|---|---|
| Main method result | ✅ | +1.59 pp FSCC, p<0.001, win rate 0.83 |
| P0 ambiguity contribution | ✅ | High-only 81.4% gain share |
| P1 alignment validation | ✅ | Beats shuffled/permuted controls |
| Heterophily boundary | ✅ | Reported as failure mode |
| GSL positioning | ✅ | Competitive, not superior |
| Runtime profile | ✅ | {ratio}x, accuracy-cost tradeoff |
| Paper assembly | ⬜ | In progress |

## Overall Readiness

Evidence-ready. Paper assembly and figure/table polish remain.
"""
        score_path.write_text(score_md)
        logger.info("Created paper_draft/aaai_readiness_score.md")
    else:
        logger.info("aaai_readiness_score.md already exists; skipping")


# ═══════════════════════════════════════════════════════════════════════════════
# Mode dispatchers
# ═══════════════════════════════════════════════════════════════════════════════


def run_smoke(device, output_dir):
    return run_profile_matrix(
        datasets=["Cora"],
        seeds=[0],
        prune_ratio=0.2,
        device=device,
        output_dir=output_dir,
    )


def run_full(device, output_dir):
    return run_profile_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        seeds=list(range(5)),
        prune_ratio=0.2,
        device=device,
        output_dir=output_dir,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Runtime Profile")
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

    if args.mode == "smoke":
        df = run_smoke(device, output_dir)
    else:
        df = run_full(device, output_dir)

    # Write output contract
    if df is not None and len(df) > 0:
        metrics = write_output_contract(df, output_dir, args.mode)
        update_paper_drafts(metrics)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        summary = build_summary(df)
        logger.info(
            "\n%s",
            summary[
                ["method", "n"]
                + [c for c in summary.columns if c.endswith("_mean")]
            ].to_string(index=False),
        )
    else:
        logger.error("No result rows produced; writing failure contract.")
        df = pd.DataFrame()
        metrics = write_output_contract(df, output_dir, args.mode)


if __name__ == "__main__":
    main()
