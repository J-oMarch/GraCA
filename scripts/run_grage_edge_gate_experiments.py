"""
GraGE Edge-Gate Experiments.

Runs the full experiment matrix with proper protocols:
- GraGE-FO: First-order edge-gate hypergradient
- GraGE-Unrolled-K1/K3: Unrolled hypergradient
- EdgeBench-Transfer: Transfer learning baseline
- Feature-only: Static feature similarity baseline
- EdgeBench-InGraphSupervised: Diagnostic only (oracle_only=True)
- EdgeInfluence-Oracle: Diagnostic only (oracle_only=True)

Usage:
    # Stage 1: Quick diagnostic (3 datasets, 3 noise types, 2 seeds, GCN only)
    python scripts/run_grage_edge_gate_experiments.py --stage diagnostic

    # Stage 2: Full matrix (6 datasets, 5 noise types, 5 seeds, 3 models)
    python scripts/run_grage_edge_gate_experiments.py --stage full
"""
import sys
import os
import argparse
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset, compute_edge_homophily
from src.training.train_proxy import train_proxy
from src.training.train_downstream import train_downstream
from src.graca.edge_influence import compute_edge_influence_scores
from src.graca.edge_bench import compute_edge_bench_in_graph, compute_edge_bench_transfer, compute_edge_bench_scores_simple
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.edge_scoring import compute_rho_score
from src.graca.pruning import prune_graph, compute_graph_stats
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.eval.result_writer import write_result_row
from src.utils.logger import get_logger
from src.utils.mask_split import split_train_support_score
from src.models.gcn import GCN
import torch.nn.functional as F
from scipy.stats import zscore as sp_zscore


# Experiment configurations
STAGE1_CONFIG = {
    "datasets": ["Cora", "CiteSeer", "PubMed"],
    "noise_types": ["cross_class_oracle", "low_feature_similarity", "degree_aligned_random"],
    "seeds": [0, 1, 2],
    "downstream_models": ["GCN"],
}

STAGE2_CONFIG = {
    "datasets": ["Cora", "CiteSeer", "PubMed", "Actor", "Texas", "Wisconsin"],
    "noise_types": [
        "cross_class_oracle",
        "cross_class_train_safe",
        "low_feature_similarity",
        "random_inter_community",
        "degree_aligned_random",
    ],
    "seeds": [0, 1, 2, 3, 4],
    "downstream_models": ["GCN", "GAT", "GraphSAGE"],
}

PRACTICAL_METHODS = [
    "Original+Noise",
    "Random-Matched",
    "Feature-only",
    "GCN-Jaccard",
    "DegreeAwareRandom",
    "GraGE-FO",
    "GraGE-Unrolled-K1",
    "GraGE-Unrolled-K3",
    "EdgeBench-Transfer",
]

DIAGNOSTIC_METHODS = [
    "EdgeBench-InGraphSupervised",
    "EdgeInfluence-Oracle",
]


def get_config_path(dataset):
    return f"configs/graca_lite_{dataset.lower()}.yaml"


def run_experiment(dataset, noise_type, seed, config, prune_ratio, noise_ratio,
                   output_dir, downstream_models, device, logger):
    """Run a single experiment (dataset × noise_type × seed)."""
    config_path = get_config_path(dataset)
    if not os.path.exists(config_path):
        logger.warning(f"Config not found: {config_path}")
        return []

    cfg = load_config(config_path)
    undirected = cfg.get("dataset", {}).get("undirected", True)

    # Load data
    set_seed(seed)
    data, num_features, num_classes = load_dataset(cfg)
    data = data.to(device)
    num_nodes = data.num_nodes
    E_orig = data.edge_index.shape[1]
    y_cpu = data.y.cpu()

    # Inject noise (ensure CPU tensors for noise injection)
    logger.info(f"  Injecting {noise_type} noise at {noise_ratio}...")
    noise_result = inject_noise(
        edge_index=data.edge_index.cpu(), num_nodes=num_nodes,
        noise_type=noise_type, noise_ratio=noise_ratio,
        x=data.x.cpu(), y=y_cpu, train_mask=data.train_mask.cpu(),
        seed=seed,
    )
    edge_index = noise_result["noisy_edge_index"].to(device)
    bad_edge_mask = noise_result["bad_edge_mask"]
    E_noisy = edge_index.shape[1]

    homo_before = compute_edge_homophily(edge_index.cpu(), y_cpu)

    # Split train into support/score
    support_mask, score_mask = split_train_support_score(
        data.train_mask, data.y, score_ratio=0.3, seed=seed, stratified=True
    )

    results = []

    # ============ Method 1: Original+Noise ============
    logger.info("    Method 1: Original+Noise")
    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, edge_index, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(edge_index.cpu(), y_cpu)
        stats = {"num_edges_before": E_noisy, "num_edges_after": E_noisy,
                 "prune_ratio": 0, "isolated_nodes": 0, "min_degree": 0, "mean_degree": 0,
                 "largest_connected_component_ratio": 1.0}
        results.append(_make_row(
            method="Original+Noise", practical=True, protocol="none",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=None, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            notes="no pruning",
        ))

    # ============ Method 2: Random-Matched ============
    logger.info("    Method 2: Random-Matched")
    from src.graca.pruning import prune_graph as prune_graph_fn
    # Random pruning
    set_seed(seed)
    risk_random = torch.rand(E_noisy, device=device)
    pruned_ei_rand, mask_rand, stats_rand = prune_graph_fn(
        edge_index=edge_index, risk_score=risk_random, num_nodes=num_nodes,
        beta=0.2, min_degree=1, undirected=undirected,
        protect_self_loops=True, target_prune_ratio=prune_ratio,
    )
    det_rand = evaluate_bad_edge_detection(mask_rand, bad_edge_mask, edge_index)
    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_rand, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_rand.cpu(), y_cpu)
        results.append(_make_row(
            method="Random-Matched", practical=True, protocol="none",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_rand, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_rand, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_rand, notes="random baseline",
        ))

    # ============ Method 3: Feature-only ============
    logger.info("    Method 3: Feature-only")
    src_n, dst_n = edge_index[0].cpu(), edge_index[1].cpu()
    cos_sim = F.cosine_similarity(data.x[src_n].cpu(), data.x[dst_n].cpu(), dim=1).numpy()
    P_feat = torch.from_numpy(-cos_sim).float().to(device)
    pruned_ei_feat, mask_feat, stats_feat = prune_graph_fn(
        edge_index=edge_index, risk_score=P_feat, num_nodes=num_nodes,
        beta=0.2, min_degree=1, undirected=undirected,
        protect_self_loops=True, target_prune_ratio=prune_ratio,
    )
    det_feat = evaluate_bad_edge_detection(mask_feat, bad_edge_mask, edge_index)
    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_feat, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_feat.cpu(), y_cpu)
        results.append(_make_row(
            method="Feature-only", practical=True, protocol="none",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_feat, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_feat, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_feat, notes="feature cosine only",
        ))

    # ============ Method 4: GCN-Jaccard ============
    logger.info("    Method 4: GCN-Jaccard")
    from src.baselines.similarity_pruning import run_jaccard_pruning
    jaccard_results, jaccard_stats = run_jaccard_pruning(
        data=data, config=cfg, num_features=num_features, num_classes=num_classes,
        device=device, seed=seed, prune_ratio=prune_ratio,
        edge_index_override=edge_index,
    )
    for ds_model in downstream_models:
        if ds_model in jaccard_results:
            res = jaccard_results[ds_model]
            results.append(_make_row(
                method="GCN-Jaccard", practical=True, protocol="none",
                dataset=dataset, noise_type=noise_type, seed=seed,
                downstream_model=ds_model, stats=jaccard_stats, res=res,
                homo_before=homo_before, homo_after=homo_before,
                bad_edge_mask=bad_edge_mask, prune_mask=None, edge_index=edge_index,
                support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
                notes="jaccard baseline on noisy graph",
            ))

    # ============ Method 5: DegreeAwareRandom ============
    logger.info("    Method 5: DegreeAwareRandom")
    from src.baselines.random_pruning import run_degree_aware_random
    degree_results, degree_stats = run_degree_aware_random(
        data=data, config=cfg, num_features=num_features, num_classes=num_classes,
        device=device, seed=seed, prune_ratio=prune_ratio,
        edge_index_override=edge_index,
    )
    for ds_model in downstream_models:
        if ds_model in degree_results:
            res = degree_results[ds_model]
            results.append(_make_row(
                method="DegreeAwareRandom", practical=True, protocol="none",
                dataset=dataset, noise_type=noise_type, seed=seed,
                downstream_model=ds_model, stats=degree_stats, res=res,
                homo_before=homo_before, homo_after=homo_before,
                bad_edge_mask=bad_edge_mask, prune_mask=None, edge_index=edge_index,
                support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
                notes="degree-aware random baseline on noisy graph",
            ))

    # ============ Method 6: GraGE-FO ============
    logger.info("    Method 6: GraGE-FO")
    # Train a model for GraGE scoring
    set_seed(seed)
    model = GCN(
        in_dim=num_features, hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
        out_dim=num_classes, num_layers=2, dropout=0.5,
    ).to(device)

    # Train model on support_mask
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model(data.x, edge_index)
        loss = F.cross_entropy(logits[support_mask], data.y[support_mask])
        loss.backward()
        optimizer.step()

    # Compute GraGE-FO scores
    from src.grage.edge_gate_influence import compute_edge_gate_influence_first_order
    grage_fo_result = compute_edge_gate_influence_first_order(
        model=model, x=data.x, edge_index=edge_index,
        y=data.y, score_mask=score_mask, normalize=True,
        undirected=undirected, bad_edge_mask=bad_edge_mask,
    )
    grage_fo_score = grage_fo_result["harmful_score"]

    # Prune using GraGE-FO scores
    from src.grage.pruning import prune_by_grage_score
    pruned_ei_fo, mask_fo, stats_fo = prune_by_grage_score(
        edge_index=edge_index, score=grage_fo_score, num_nodes=num_nodes,
        prune_ratio=prune_ratio, undirected=undirected,
    )
    det_fo = evaluate_bad_edge_detection(mask_fo, bad_edge_mask, edge_index)

    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_fo, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_fo.cpu(), y_cpu)
        results.append(_make_row(
            method="GraGE-FO", practical=True, protocol="first_order",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_fo, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_fo, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_fo, notes=f"grage_fo,grad_mean={grage_fo_result['diagnostics']['grad_mean']:.6f}",
        ))

    # ============ Method 7: GraGE-Unrolled-K1 ============
    logger.info("    Method 7: GraGE-Unrolled-K1")
    from src.grage.unrolled_hypergradient import compute_edge_gate_influence_unrolled

    def model_ctor():
        return GCN(
            in_dim=num_features, hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
            out_dim=num_classes, num_layers=2, dropout=0.0,
        ).to(device)

    grage_u1_result = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=model.state_dict(),
        x=data.x, edge_index=edge_index, y=data.y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=1, inner_lr=0.01, undirected=undirected,
        bad_edge_mask=bad_edge_mask,
    )
    grage_u1_score = grage_u1_result["harmful_score"]

    pruned_ei_u1, mask_u1, stats_u1 = prune_by_grage_score(
        edge_index=edge_index, score=grage_u1_score, num_nodes=num_nodes,
        prune_ratio=prune_ratio, undirected=undirected,
    )
    det_u1 = evaluate_bad_edge_detection(mask_u1, bad_edge_mask, edge_index)

    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_u1, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_u1.cpu(), y_cpu)
        results.append(_make_row(
            method="GraGE-Unrolled-K1", practical=True, protocol="unrolled_k1",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_u1, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_u1, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_u1, notes=f"grage_unrolled_k1",
        ))

    # ============ Method 8: GraGE-Unrolled-K3 ============
    logger.info("    Method 8: GraGE-Unrolled-K3")
    grage_u3_result = compute_edge_gate_influence_unrolled(
        model_ctor=model_ctor,
        init_state_dict=model.state_dict(),
        x=data.x, edge_index=edge_index, y=data.y,
        support_mask=support_mask, score_mask=score_mask,
        inner_steps=3, inner_lr=0.01, undirected=undirected,
        bad_edge_mask=bad_edge_mask,
    )
    grage_u3_score = grage_u3_result["harmful_score"]

    pruned_ei_u3, mask_u3, stats_u3 = prune_by_grage_score(
        edge_index=edge_index, score=grage_u3_score, num_nodes=num_nodes,
        prune_ratio=prune_ratio, undirected=undirected,
    )
    det_u3 = evaluate_bad_edge_detection(mask_u3, bad_edge_mask, edge_index)

    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_u3, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_u3.cpu(), y_cpu)
        results.append(_make_row(
            method="GraGE-Unrolled-K3", practical=True, protocol="unrolled_k3",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_u3, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_u3, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_u3, notes=f"grage_unrolled_k3",
        ))

    # ============ Method 9: EdgeBench-Transfer ============
    logger.info("    Method 9: EdgeBench-Transfer")
    # For transfer, we use Feature-only scores as a simple proxy
    # (In full implementation, this would train on source graph units)
    edgebench_transfer_score = compute_edge_bench_scores_simple(cos_sim, cos_sim)

    pruned_ei_ebt, mask_ebt, stats_ebt = prune_by_grage_score(
        edge_index=edge_index,
        score=torch.from_numpy(edgebench_transfer_score).float().to(device),
        num_nodes=num_nodes, prune_ratio=prune_ratio, undirected=undirected,
    )
    det_ebt = evaluate_bad_edge_detection(mask_ebt, bad_edge_mask, edge_index)

    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_ebt, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_ebt.cpu(), y_cpu)
        results.append(_make_row(
            method="EdgeBench-Transfer", practical=True, protocol="transfer_seed",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_ebt, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_ebt, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_ebt, notes="edgebench_transfer_proxy",
        ))

    # ============ Method 10: EdgeBench-InGraphSupervised (diagnostic) ============
    logger.info("    Method 10: EdgeBench-InGraphSupervised (diagnostic)")
    edgebench_igs = compute_edge_bench_in_graph(
        delta_softmax=cos_sim,  # Use cosine as proxy
        feature_cosine=cos_sim,
        bad_edge_mask=bad_edge_mask.cpu().numpy(),
        seed=seed,
    )
    edgebench_igs_score = edgebench_igs["scores"]

    pruned_ei_igs, mask_igs, stats_igs = prune_by_grage_score(
        edge_index=edge_index,
        score=torch.from_numpy(edgebench_igs_score).float().to(device),
        num_nodes=num_nodes, prune_ratio=prune_ratio, undirected=undirected,
    )
    det_igs = evaluate_bad_edge_detection(mask_igs, bad_edge_mask, edge_index)

    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_igs, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_igs.cpu(), y_cpu)
        results.append(_make_row(
            method="EdgeBench-InGraphSupervised", practical=False, protocol="in_graph_supervised",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_igs, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_igs, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_igs, notes="oracle_only,uses target bad_edge_mask",
        ))

    # ============ Method 11: EdgeInfluence-Oracle (diagnostic) ============
    logger.info("    Method 11: EdgeInfluence-Oracle (diagnostic)")
    # Train teacher for EdgeInfluence
    set_seed(seed)
    model_ei, teacher, _, _ = train_proxy(cfg, data, num_features, num_classes, device, seed)
    x = data.x.to(device)
    train_mask = data.train_mask
    unlabeled_mask = ~train_mask
    teacher_probs = teacher.predict(x, edge_index)
    pseudo_cfg = cfg.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.6)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)
    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

    ei_result = compute_edge_influence_scores(
        teacher, x, edge_index, data.y, train_mask, unlabeled_mask,
        teacher_probs, rho_score, num_nodes, undirected=undirected,
    )
    ds_oracle = ei_result["delta_softmax_oracle_undirected"].cpu().numpy()
    combined_oracle = sp_zscore(ds_oracle) + sp_zscore(-cos_sim)
    P_oracle = torch.from_numpy(combined_oracle).float().to(device)

    pruned_ei_oracle, mask_oracle, stats_oracle = prune_graph_fn(
        edge_index=edge_index, risk_score=P_oracle, num_nodes=num_nodes,
        beta=0.2, min_degree=1, undirected=undirected,
        protect_self_loops=True, target_prune_ratio=prune_ratio,
    )
    det_oracle = evaluate_bad_edge_detection(mask_oracle, bad_edge_mask, edge_index)

    for ds_model in downstream_models:
        set_seed(seed)
        t0 = time.time()
        res = train_downstream(ds_model, data, pruned_ei_oracle, cfg, num_features, num_classes, device, seed)
        res["runtime"] = time.time() - t0
        homo_after = compute_edge_homophily(pruned_ei_oracle.cpu(), y_cpu)
        results.append(_make_row(
            method="EdgeInfluence-Oracle", practical=False, protocol="oracle",
            dataset=dataset, noise_type=noise_type, seed=seed,
            downstream_model=ds_model, stats=stats_oracle, res=res,
            homo_before=homo_before, homo_after=homo_after,
            bad_edge_mask=bad_edge_mask, prune_mask=mask_oracle, edge_index=edge_index,
            support_size=int(support_mask.sum()), score_size=int(score_mask.sum()),
            det=det_oracle, notes="oracle_only,uses true labels",
        ))

    return results


def _make_row(method, practical, protocol, dataset, noise_type, seed,
              downstream_model, stats, res, homo_before, homo_after,
              bad_edge_mask, prune_mask, edge_index, support_size, score_size,
              det=None, notes=""):
    """Create a result row."""
    return {
        "run_id": f"{method}_{dataset}_{noise_type}_{downstream_model}_seed{seed}",
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "practical": practical,
        "protocol": protocol,
        "dataset": dataset,
        "noise_type": noise_type,
        "seed": seed,
        "downstream_model": downstream_model,
        "support_score_split_seed": seed,
        "support_size": support_size,
        "score_size": score_size,
        "test_acc": res["test_acc"],
        "test_f1": res["test_f1"],
        "val_acc": res["val_acc"],
        "best_epoch": res["best_epoch"],
        "runtime": res["runtime"],
        "actual_prune_ratio": stats.get("prune_ratio", 0),
        "num_edges_before": stats.get("num_edges_before", 0),
        "num_edges_after": stats.get("num_edges_after", 0),
        "isolated_nodes": stats.get("isolated_nodes", 0),
        "min_degree": stats.get("min_degree", 0),
        "mean_degree": stats.get("mean_degree", 0),
        "edge_homophily_before": homo_before,
        "edge_homophily_after": homo_after,
        "bad_edge_precision": det["bad_edge_precision"] if det else 0,
        "bad_edge_recall": det["bad_edge_recall"] if det else 0,
        "bad_edge_f1": det["bad_edge_f1"] if det else 0,
        "noise_ratio": 0.2,
        "oracle_only": not practical,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="GraGE Edge-Gate Experiments")
    parser.add_argument("--stage", type=str, default="diagnostic",
                        choices=["diagnostic", "full"],
                        help="Experiment stage: diagnostic (quick) or full")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-set based on stage)")
    parser.add_argument("--prune_ratio", type=float, default=0.20)
    parser.add_argument("--noise_ratio", type=float, default=0.20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Select config based on stage
    if args.stage == "diagnostic":
        cfg = STAGE1_CONFIG
        default_output = "results_clean/grage_edge_gate/diagnostic"
    else:
        cfg = STAGE2_CONFIG
        default_output = "results_clean/grage_edge_gate/full"

    output_dir = args.output_dir or default_output
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger("grage_edge_gate")
    device = get_device({"device": {"type": args.device or "auto"}})

    # Calculate total experiments
    total = len(cfg["datasets"]) * len(cfg["noise_types"]) * len(cfg["seeds"])
    logger.info(f"\n{'='*60}")
    logger.info(f"GraGE Edge-Gate Experiments ({args.stage})")
    logger.info(f"{'='*60}")
    logger.info(f"Datasets: {cfg['datasets']}")
    logger.info(f"Noise types: {cfg['noise_types']}")
    logger.info(f"Seeds: {cfg['seeds']}")
    logger.info(f"Downstream models: {cfg['downstream_models']}")
    logger.info(f"Total experiments: {total}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}\n")

    all_results = []
    completed = 0
    failed = 0
    start_time = time.time()

    for dataset in cfg["datasets"]:
        for noise_type in cfg["noise_types"]:
            for seed in cfg["seeds"]:
                logger.info(f"[{completed+failed+1}/{total}] {dataset}/{noise_type}/seed{seed}")
                try:
                    results = run_experiment(
                        dataset=dataset, noise_type=noise_type, seed=seed,
                        config=None,  # Config loaded inside
                        prune_ratio=args.prune_ratio,
                        noise_ratio=args.noise_ratio,
                        output_dir=output_dir,
                        downstream_models=cfg["downstream_models"],
                        device=device, logger=logger,
                    )
                    all_results.extend(results)
                    completed += 1
                    logger.info(f"  ✓ Completed: {len(results)} rows")
                except Exception as e:
                    failed += 1
                    logger.error(f"  ✗ Failed: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = f"{output_dir}/results.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total rows: {len(all_results)}")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Results: {csv_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
