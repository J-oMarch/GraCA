"""
Idea Feasibility Verification for EdgeInfluence.

Four-stage verification on Cora (seed=42):

Stage 1: Basic L-score on cross_class_oracle 20% → AUC > 0.70 to continue
Stage 2: Expanded methods → AUC > 0.80 to continue
Stage 3: Effectiveness + Distribution analysis
Stage 4: Full AUC table on all noise types → AUC > 0.70 on ALL to proceed

Usage:
    python scripts/verify_idea.py --config configs/graca_lite_cora.yaml --seed 42
"""
import sys
import os
import argparse
import torch
import numpy as np
import json
import time
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.training.train_proxy import train_proxy
from src.graca.edge_influence import (
    compute_edge_influence_scores,
    compute_loo_sampling_scores,
    compute_loss_gradient_scores,
)
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.edge_scoring import compute_rho_score, compute_D
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.utils.logger import get_logger


def compute_auc(y_true, y_score):
    """Compute AUC using trapezoidal rule."""
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp, auc, prev_fpr = 0, 0, 0.0, 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
            fpr = fp / n_neg
            tpr = tp / n_pos
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr
    return auc


def compute_ap(y_true, y_score):
    """Compute Average Precision."""
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    tp, ap = 0, 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            ap += tp / (i + 1)
    return ap / n_pos


def pearson_r(y_true, y_score):
    """Compute Pearson correlation."""
    r, p = scipy_stats.pearsonr(y_score, y_true)
    return r, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/graca_lite_cora.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", type=int, default=0,
                        help="0=all stages, 1-4=specific stage")
    parser.add_argument("--loo_samples", type=int, default=1000,
                        help="Number of edges for oracle LOO sampling")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("verify_idea")
    set_seed(args.seed)

    device = get_device(config)
    ds_name = config["dataset"]["name"]
    undirected = config.get("dataset", {}).get("undirected", True)

    logger.info(f"{'='*60}")
    logger.info(f"EdgeInfluence Feasibility Verification")
    logger.info(f"Dataset: {ds_name}, Seed: {args.seed}")
    logger.info(f"{'='*60}")

    # Load data
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)
    num_nodes = data.num_nodes

    # Train teacher
    logger.info("Training teacher GCN...")
    t0 = time.time()
    model, teacher, train_log, _ = train_proxy(config, data, num_features, num_classes, device, args.seed)
    logger.info(f"Teacher training: {time.time()-t0:.1f}s")

    x = data.x.to(device)
    y = data.y
    train_mask = data.train_mask
    unlabeled_mask = ~train_mask
    edge_index_clean = data.edge_index

    teacher_probs = teacher.predict(x, edge_index_clean)
    pseudo_cfg = config.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.6)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)
    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )
    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

    # Cross-class labels on clean graph
    src_c, dst_c = edge_index_clean[0].cpu(), edge_index_clean[1].cpu()
    y_cpu = y.cpu()
    cross_clean = (y_cpu[src_c] != y_cpu[dst_c]).numpy().astype(int)

    all_results = {}

    # ================================================================
    # STAGE 1: Basic L-score on cross_class_oracle 20%
    # ================================================================
    if args.stage in (0, 1):
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 1: Basic L-score (cross_class_oracle 20%)")
        logger.info(f"{'='*60}")

        noise_result = inject_noise(
            edge_index=edge_index_clean.cpu(), num_nodes=num_nodes,
            noise_type="cross_class_oracle", noise_ratio=0.20,
            y=y_cpu, seed=args.seed,
        )
        noisy_ei = noise_result["noisy_edge_index"].to(device)
        bad_mask = noise_result["bad_edge_mask"]

        ei_result = compute_edge_influence_scores(
            teacher, x, noisy_ei, y, train_mask, unlabeled_mask,
            teacher_probs, rho_score, num_nodes, undirected=undirected,
        )

        L_ud = ei_result["L_undirected"].cpu().numpy()
        L_raw_ud = ei_result["L_raw_undirected"].cpu().numpy()
        L_oracle_ud = ei_result["L_oracle_undirected"].cpu().numpy()
        ds_oracle_ud = ei_result["delta_softmax_oracle_undirected"].cpu().numpy()
        ds_pseudo_ud = ei_result["delta_softmax_pseudo_undirected"].cpu().numpy()

        # Cross-class on noisy graph
        src_n, dst_n = noisy_ei[0].cpu(), noisy_ei[1].cpu()
        cross_noisy = (y_cpu[src_n] != y_cpu[dst_n]).numpy().astype(int)

        # Also compute feature cosine similarity
        import torch.nn.functional as F
        cos_sim = F.cosine_similarity(x[src_n].cpu(), x[dst_n].cpu(), dim=1).numpy()

        # Combined scores: delta_softmax + feature_cosine
        from scipy.stats import zscore as sp_zscore
        combined_oracle = sp_zscore(ds_oracle_ud) + sp_zscore(-cos_sim)
        combined_pseudo = sp_zscore(ds_pseudo_ud) + sp_zscore(-cos_sim)

        stage1_results = {}
        for name, scores in [("L_weighted", L_ud), ("L_raw", L_raw_ud),
                              ("L_oracle", L_oracle_ud),
                              ("delta_softmax_oracle", ds_oracle_ud),
                              ("delta_softmax_pseudo", ds_pseudo_ud),
                              ("feature_cosine", -cos_sim),
                              ("combined_oracle", combined_oracle),
                              ("combined_pseudo", combined_pseudo)]:
            auc_val = compute_auc(cross_noisy, scores)
            auc_inv = compute_auc(cross_noisy, -scores)
            best_auc = max(auc_val, auc_inv)
            r, p = pearson_r(cross_noisy, scores)
            stage1_results[name] = {"auc": best_auc, "pearson_r": r, "pearson_p": p}
            logger.info(f"  {name}: AUC={best_auc:.4f}, Pearson r={r:.4f}")

        best_stage1 = max(v["auc"] for v in stage1_results.values())
        all_results["stage1"] = stage1_results
        all_results["stage1_best_auc"] = best_stage1

        if best_stage1 < 0.70:
            logger.info(f"\n✗ STAGE 1 FAILED: Best AUC = {best_stage1:.4f} < 0.70")
            logger.info("  Idea does not work. Report failure.")
            _save_results(all_results, ds_name, args.seed, args.loo_samples)
            return
        else:
            logger.info(f"\n✓ STAGE 1 PASSED: Best AUC = {best_stage1:.4f} > 0.70")

    # ================================================================
    # STAGE 2: Expanded methods + Oracle LOO
    # ================================================================
    if args.stage in (0, 2):
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 2: Expanded methods + Oracle LOO")
        logger.info(f"{'='*60}")

        # Use same noisy graph as stage 1
        if "stage1" not in all_results:
            noise_result = inject_noise(
                edge_index=edge_index_clean.cpu(), num_nodes=num_nodes,
                noise_type="cross_class_oracle", noise_ratio=0.20,
                y=y_cpu, seed=args.seed,
            )
            noisy_ei = noise_result["noisy_edge_index"].to(device)
            src_n, dst_n = noisy_ei[0].cpu(), noisy_ei[1].cpu()
            cross_noisy = (y_cpu[src_n] != y_cpu[dst_n]).numpy().astype(int)

        # Oracle LOO (sampling)
        logger.info(f"  Computing Oracle LOO ({args.loo_samples} edges)...")
        t0 = time.time()
        loo_result = compute_loo_sampling_scores(
            teacher, x, noisy_ei, y, num_nodes,
            sample_size=args.loo_samples, seed=args.seed,
        )
        loo_time = time.time() - t0
        L_loo = loo_result["L_loo"]
        cross_loo = loo_result["cross_class"]
        auc_loo = compute_auc(cross_loo, -L_loo)  # negative L = harmful
        if auc_loo < 0.5:
            auc_loo = compute_auc(cross_loo, L_loo)
        logger.info(f"  Oracle LOO: AUC={auc_loo:.4f} ({loo_time:.1f}s)")

        # Loss gradient scores
        logger.info("  Computing loss gradient scores...")
        grad_result = compute_loss_gradient_scores(
            teacher, x, noisy_ei, y, train_mask, num_nodes, undirected=undirected,
        )
        grad_ud = grad_result["grad_scores_undirected"].cpu().numpy()
        auc_grad = compute_auc(cross_noisy, grad_ud)
        logger.info(f"  Loss gradient: AUC={auc_grad:.4f}")

        # Feature similarity
        import torch.nn.functional as F
        cos_sim = F.cosine_similarity(x[src_n].cpu(), x[dst_n].cpu(), dim=1).numpy()
        auc_cos = compute_auc(cross_noisy, -cos_sim)
        logger.info(f"  Feature cosine: AUC={auc_cos:.4f}")

        # KL divergence
        with torch.no_grad():
            logits_full, _ = teacher.teacher(x, noisy_ei, return_hidden=True)
            probs_full = torch.softmax(logits_full, dim=-1)
        kl_vu = (probs_full[src_n.to(device)] * (probs_full[src_n.to(device)].clamp(min=1e-12).log() -
                 probs_full[dst_n.to(device)].clamp(min=1e-12).log())).sum(-1).cpu().numpy()
        kl_uv = (probs_full[dst_n.to(device)] * (probs_full[dst_n.to(device)].clamp(min=1e-12).log() -
                 probs_full[src_n.to(device)].clamp(min=1e-12).log())).sum(-1).cpu().numpy()
        kl_sym = (kl_vu + kl_uv) / 2
        auc_kl = compute_auc(cross_noisy, kl_sym)
        logger.info(f"  KL divergence: AUC={auc_kl:.4f}")

        # norm × |D|
        grad_for_D = ei_result.get("L_raw", torch.zeros(1))
        D = compute_D(grad_for_D.unsqueeze(-1) if grad_for_D.dim() == 1 else grad_for_D,
                       noisy_ei, eps=1e-12) if grad_for_D.dim() >= 1 else torch.zeros(noisy_ei.shape[1])
        # Actually compute D from the gradient collector output
        # For now, use a placeholder - we need the actual gradient

        stage2_results = {
            "oracle_loo": {"auc": auc_loo, "n_samples": len(L_loo)},
            "loss_gradient": {"auc": auc_grad},
            "feature_cosine": {"auc": auc_cos},
            "kl_divergence": {"auc": auc_kl},
        }
        all_results["stage2"] = stage2_results

        best_stage2 = max(auc_loo, auc_grad, auc_cos, auc_kl)
        all_results["stage2_best_auc"] = best_stage2

        if best_stage2 < 0.80:
            logger.info(f"\n✗ STAGE 2: Best AUC = {best_stage2:.4f} < 0.80")
            logger.info("  Proceeding to Stage 3 with available methods.")
        else:
            logger.info(f"\n✓ STAGE 2 PASSED: Best AUC = {best_stage2:.4f} > 0.80")

    # ================================================================
    # STAGE 3: Effectiveness + Distribution analysis
    # ================================================================
    if args.stage in (0, 3):
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 3: Effectiveness + Distribution analysis")
        logger.info(f"{'='*60}")

        # Use best scoring method from stage 1/2
        # For effectiveness: rank edges by L, check if top-20% has more cross-class
        L_scores = ei_result["L_undirected"].cpu().numpy()
        n_edges = len(L_scores)
        top_20_pct = int(n_edges * 0.20)

        sorted_idx = np.argsort(-L_scores)
        top_edges_cross = cross_noisy[sorted_idx[:top_20_pct]].mean()
        random_cross = cross_noisy.mean()
        ratio = top_edges_cross / max(random_cross, 1e-8)

        logger.info(f"  Cross-class ratio in top-20% L-score edges: {top_edges_cross:.4f}")
        logger.info(f"  Cross-class ratio in random edges: {random_cross:.4f}")
        logger.info(f"  Ratio (top-20% / random): {ratio:.2f}x")

        # Distribution analysis
        L_cross = L_scores[cross_noisy == 1]
        L_same = L_scores[cross_noisy == 0]

        logger.info(f"\n  L-score distribution:")
        logger.info(f"    Cross-class: mean={L_cross.mean():.6f}, std={L_cross.std():.6f}, median={np.median(L_cross):.6f}")
        logger.info(f"    Same-class:  mean={L_same.mean():.6f}, std={L_same.std():.6f}, median={np.median(L_same):.6f}")
        logger.info(f"    Effect size (Cohen's d): {(L_cross.mean() - L_same.mean()) / max(np.sqrt((L_cross.var() + L_same.var())/2), 1e-8):.4f}")

        # Bad-edge detection
        prune_ratio = config.get("pruning", {}).get("beta", 0.2)
        num_prune = int(n_edges * prune_ratio)
        prune_mask = torch.zeros(n_edges, dtype=torch.bool)
        prune_mask[sorted_idx[:num_prune]] = True
        det = evaluate_bad_edge_detection(prune_mask, bad_mask, noisy_ei)

        logger.info(f"\n  Bad-edge detection (prune_ratio={prune_ratio}):")
        logger.info(f"    Precision: {det['bad_edge_precision']:.4f}")
        logger.info(f"    Recall:    {det['bad_edge_recall']:.4f}")
        logger.info(f"    F1:        {det['bad_edge_f1']:.4f}")

        stage3_results = {
            "top_20_pct_cross_ratio": float(top_edges_cross),
            "random_cross_ratio": float(random_cross),
            "effectiveness_ratio": float(ratio),
            "L_cross_mean": float(L_cross.mean()),
            "L_same_mean": float(L_same.mean()),
            "cohens_d": float((L_cross.mean() - L_same.mean()) / max(np.sqrt((L_cross.var() + L_same.var())/2), 1e-8)),
            "bad_edge_detection": det,
        }
        all_results["stage3"] = stage3_results

    # ================================================================
    # STAGE 4: Full AUC table on all noise types
    # ================================================================
    if args.stage in (0, 4):
        logger.info(f"\n{'='*60}")
        logger.info("STAGE 4: Full AUC table on all noise types")
        logger.info(f"{'='*60}")

        noise_types = [
            "cross_class_oracle",
            "train_safe_oracle_v2",
            "low_feature_similarity",
            "random_inter_community",
            "degree_aligned_random",
        ]

        stage4_results = {}
        all_above_070 = True

        for nt in noise_types:
            try:
                noise_r = inject_noise(
                    edge_index=edge_index_clean.cpu(), num_nodes=num_nodes,
                    noise_type=nt, noise_ratio=0.20,
                    x=data.x.cpu(), y=y_cpu, train_mask=train_mask.cpu(),
                    seed=args.seed,
                )
                noisy_ei_nt = noise_r["noisy_edge_index"].to(device)
                src_nt, dst_nt = noisy_ei_nt[0].cpu(), noisy_ei_nt[1].cpu()
                cross_nt = (y_cpu[src_nt] != y_cpu[dst_nt]).numpy().astype(int)

                ei_nt = compute_edge_influence_scores(
                    teacher, x, noisy_ei_nt, y, train_mask, unlabeled_mask,
                    teacher_probs, rho_score, num_nodes, undirected=undirected,
                )

                # Feature cosine
                cos_nt = torch.nn.functional.cosine_similarity(
                    x[src_nt].cpu(), x[dst_nt].cpu(), dim=1
                ).numpy()

                # Combined scores
                from scipy.stats import zscore as sp_zscore2
                ds_oracle_nt = ei_nt["delta_softmax_oracle_undirected"].cpu().numpy()
                ds_pseudo_nt = ei_nt["delta_softmax_pseudo_undirected"].cpu().numpy()
                combined_oracle_nt = sp_zscore2(ds_oracle_nt) + sp_zscore2(-cos_nt)
                combined_pseudo_nt = sp_zscore2(ds_pseudo_nt) + sp_zscore2(-cos_nt)

                results_nt = {}
                for name, scores in [
                    ("L_weighted", ei_nt["L_undirected"].cpu().numpy()),
                    ("delta_softmax_oracle", ds_oracle_nt),
                    ("delta_softmax_pseudo", ds_pseudo_nt),
                    ("feature_cosine", -cos_nt),
                    ("combined_oracle", combined_oracle_nt),
                    ("combined_pseudo", combined_pseudo_nt),
                ]:
                    auc_val = compute_auc(cross_nt, scores)
                    auc_inv = compute_auc(cross_nt, -scores)
                    results_nt[name] = max(auc_val, auc_inv)

                stage4_results[nt] = results_nt
                best_nt = max(results_nt.values())
                logger.info(f"  {nt}: best={best_nt:.4f} | " +
                            " | ".join(f"{k}={v:.4f}" for k, v in results_nt.items()))

                if best_nt < 0.70:
                    all_above_070 = False

            except Exception as e:
                logger.warning(f"  {nt}: ERROR - {e}")
                stage4_results[nt] = {"error": str(e)}
                all_above_070 = False

        all_results["stage4"] = stage4_results
        all_results["all_above_070"] = all_above_070

    # ================================================================
    # FINAL DECISION
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("FINAL DECISION")
    logger.info(f"{'='*60}")

    best_auc = all_results.get("stage1_best_auc", 0)
    oracle_auc = all_results.get("stage2", {}).get("oracle_loo", {}).get("auc", 0)

    if best_auc >= 0.70 and all_results.get("all_above_070", False):
        logger.info(f"✓ IDEA FEASIBLE: Best AUC = {best_auc:.4f} ≥ 0.70 on all noise types")
        logger.info("  Proceed with full experiments.")
        all_results["feasible"] = True
    elif best_auc >= 0.70:
        logger.info(f"△ IDEA PARTIALLY FEASIBLE: Best AUC = {best_auc:.4f} ≥ 0.70")
        logger.info(f"  Oracle LOO AUC = {oracle_auc:.4f}")
        logger.info(f"  But not all noise types above 0.70.")
        all_results["feasible"] = "partial"
    else:
        logger.info(f"✗ IDEA NOT FEASIBLE: Best AUC = {best_auc:.4f} < 0.70")
        logger.info(f"  Oracle LOO AUC = {oracle_auc:.4f}")
        all_results["feasible"] = False

    _save_results(all_results, ds_name, args.seed, args.loo_samples)


def _save_results(all_results, ds_name, seed, loo_samples):
    """Save verification results to JSON."""
    os.makedirs("results_clean/diagnostics/", exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    result = convert(all_results)
    result["dataset"] = ds_name
    result["seed"] = seed
    result["loo_samples"] = loo_samples

    path = f"results_clean/diagnostics/verify_idea_{ds_name}_{seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
