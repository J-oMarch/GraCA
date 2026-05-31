"""
Phase 1: Idea Feasibility Verification for EdgeInfluence.

Runs on Cora with seed 42. Checks if edge-level loss-change scores
can distinguish cross-class edges from same-class edges.

Computes:
  - Pearson correlation between EdgeInfluence and indicator "endpoints from different classes"
  - AUC and Average Precision (AP)

Decision:
  - AUC > 0.75: proceed with full experiments
  - AUC < 0.75: idea does not work, report honestly

Usage:
    python scripts/verify_idea.py --config configs/graca_lite_cora.yaml --seed 42
"""
import sys
import os
import argparse
import torch
import numpy as np
import time
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.training.train_proxy import train_proxy
from src.graca.edge_influence import compute_edge_influence_scores
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.graca.edge_scoring import compute_rho_score
from src.eval.noise_injection import inject_noise, evaluate_bad_edge_detection
from src.utils.logger import get_logger


def compute_auc(y_true, y_score):
    """Compute AUC using trapezoidal rule."""
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    n_pos = y_true.sum()
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
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    tp, ap = 0, 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            ap += tp / (i + 1)
    return ap / n_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/graca_lite_cora.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_type", type=str, default="cross_class_oracle",
                        choices=["cross_class_oracle", "low_feature_similarity"])
    parser.add_argument("--noise_ratio", type=float, default=0.20)
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

    # Inject noise for bad-edge detection evaluation
    logger.info(f"Injecting {args.noise_type} noise at {args.noise_ratio}...")
    noise_result = inject_noise(
        edge_index=data.edge_index.cpu(),
        num_nodes=num_nodes,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        x=data.x.cpu(),
        y=data.y.cpu(),
        train_mask=data.train_mask.cpu(),
        seed=args.seed,
    )
    noisy_edge_index = noise_result["noisy_edge_index"].to(device)
    bad_edge_mask = noise_result["bad_edge_mask"]
    E_noisy = noisy_edge_index.shape[1]
    E_orig = data.edge_index.shape[1]

    logger.info(f"Original edges: {E_orig}, Noisy edges: {E_noisy}, "
                f"Injected pairs: {noise_result['num_injected_edges']}")

    # Step 1: Train teacher to convergence
    logger.info("\nStep 1: Training teacher GCN to convergence...")
    t0 = time.time()
    model, teacher, train_log, saved_checkpoints = train_proxy(
        config, data, num_features, num_classes, device, args.seed
    )
    teacher_train_time = time.time() - t0
    logger.info(f"Teacher training time: {teacher_train_time:.1f}s")

    # Get teacher predictions
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    unlabeled_mask = ~train_mask

    teacher_probs = teacher.predict(x, data.edge_index)  # use clean graph for teacher

    # Compute rho_score
    pseudo_cfg = config.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.6)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )

    rho_score = compute_rho_score(teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho)

    # Step 2: Compute EdgeInfluence on NOISY graph
    logger.info("\nStep 2: Computing EdgeInfluence scores on noisy graph...")
    t0 = time.time()
    ei_result = compute_edge_influence_scores(
        teacher=teacher,
        x=x,
        edge_index=noisy_edge_index,
        y=y,
        train_mask=train_mask,
        unlabeled_mask=unlabeled_mask,
        teacher_probs=teacher_probs,
        rho_score=rho_score,
        num_nodes=num_nodes,
        undirected=undirected,
    )
    scoring_time = time.time() - t0
    logger.info(f"EdgeInfluence scoring time: {scoring_time:.1f}s")

    L = ei_result["L_weighted"].cpu().numpy()
    L_undirected = ei_result["L_undirected"].cpu().numpy() if ei_result["L_undirected"] is not None else L

    diagnostics = ei_result["diagnostics"]
    logger.info(f"\nEdgeInfluence diagnostics:")
    for k, v in diagnostics.items():
        logger.info(f"  {k}: {v:.6f}")

    # Step 3: Evaluate using oracle cross-class labels (for diagnostic only)
    logger.info("\nStep 3: Evaluating with oracle cross-class labels (diagnostic)...")
    y_cpu = data.y.cpu()
    src_noisy = noisy_edge_index[0].cpu()
    dst_noisy = noisy_edge_index[1].cpu()

    # Cross-class indicator: 1 if endpoints have different labels
    cross_class = (y_cpu[src_noisy] != y_cpu[dst_noisy]).numpy().astype(int)

    # Compute correlation and AUC for each score variant
    results = {}
    for name, scores in [("L_weighted", L), ("L_undirected", L_undirected)]:
        pearson_r, pearson_p = scipy_stats.pearsonr(scores, cross_class)
        auc = compute_auc(cross_class, scores)
        ap = compute_ap(cross_class, scores)
        results[name] = {"pearson_r": pearson_r, "pearson_p": pearson_p, "auc": auc, "ap": ap}

    # Step 4: Evaluate bad-edge detection with injected noise
    logger.info("\nStep 4: Evaluating bad-edge detection with injected noise...")
    prune_mask = torch.zeros(E_noisy, dtype=torch.bool)
    # Prune top-k edges by influence score (higher = more harmful)
    prune_ratio = config.get("pruning", {}).get("beta", 0.2)
    num_prune = int(E_noisy * prune_ratio)
    _, top_indices = torch.topk(torch.from_numpy(L_undirected), num_prune)
    prune_mask[top_indices] = True

    det = evaluate_bad_edge_detection(prune_mask, bad_edge_mask, noisy_edge_index)

    # Step 5: Also evaluate on clean graph (same-class vs cross-class)
    logger.info("\nStep 5: Evaluating on clean graph (oracle cross-class)...")
    ei_clean = compute_edge_influence_scores(
        teacher=teacher,
        x=x,
        edge_index=data.edge_index,
        y=y,
        train_mask=train_mask,
        unlabeled_mask=unlabeled_mask,
        teacher_probs=teacher_probs,
        rho_score=rho_score,
        num_nodes=num_nodes,
        undirected=undirected,
    )
    L_clean = ei_clean["L_weighted"].cpu().numpy()
    L_clean_ud = ei_clean["L_undirected"].cpu().numpy() if ei_clean["L_undirected"] is not None else L_clean

    y_cpu = data.y.cpu()
    src_clean = data.edge_index[0].cpu()
    dst_clean = data.edge_index[1].cpu()
    cross_class_clean = (y_cpu[src_clean] != y_cpu[dst_clean]).numpy().astype(int)

    pearson_r_clean, pearson_p_clean = scipy_stats.pearsonr(L_clean_ud, cross_class_clean)
    auc_clean = compute_auc(cross_class_clean, L_clean_ud)
    ap_clean = compute_ap(cross_class_clean, L_clean_ud)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"FEASIBILITY VERIFICATION RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"\n--- Noisy Graph ({args.noise_type} {args.noise_ratio}) ---")
    for name, metrics in results.items():
        logger.info(f"  {name}:")
        logger.info(f"    Pearson r = {metrics['pearson_r']:.4f} (p = {metrics['pearson_p']:.2e})")
        logger.info(f"    AUC = {metrics['auc']:.4f}")
        logger.info(f"    AP  = {metrics['ap']:.4f}")

    logger.info(f"\n--- Noisy Graph Bad-Edge Detection ---")
    logger.info(f"  Noise type: {args.noise_type}")
    logger.info(f"  Noise ratio: {args.noise_ratio}")
    logger.info(f"  Prune ratio: {prune_ratio}")
    logger.info(f"  Bad-edge Precision: {det['bad_edge_precision']:.4f}")
    logger.info(f"  Bad-edge Recall:    {det['bad_edge_recall']:.4f}")
    logger.info(f"  Bad-edge F1:        {det['bad_edge_f1']:.4f}")
    logger.info(f"  Clean edges mistakenly removed: {det['clean_edge_mistakenly_removed_ratio']:.4f}")

    logger.info(f"\n--- Clean Graph (Oracle Cross-Class) ---")
    logger.info(f"  Pearson r = {pearson_r_clean:.4f} (p = {pearson_p_clean:.2e})")
    logger.info(f"  AUC = {auc_clean:.4f}")
    logger.info(f"  AP  = {ap_clean:.4f}")

    # Decision
    best_auc = max(results[name]["auc"] for name in results)
    best_auc_clean = auc_clean

    logger.info(f"\n{'='*60}")
    if best_auc >= 0.75 or best_auc_clean >= 0.75:
        logger.info(f"✓ IDEA FEASIBLE: Best AUC = {max(best_auc, best_auc_clean):.4f} > 0.75")
        logger.info(f"  Proceed with full experiments.")
    elif best_auc >= 0.65 or best_auc_clean >= 0.65:
        logger.info(f"△ IDEA MARGINAL: Best AUC = {max(best_auc, best_auc_clean):.4f}")
        logger.info(f"  Signal exists but weak. Consider improving scoring.")
    else:
        logger.info(f"✗ IDEA NOT FEASIBLE: Best AUC = {max(best_auc, best_auc_clean):.4f} < 0.75")
        logger.info(f"  EdgeInfluence scores cannot distinguish cross-class edges.")
    logger.info(f"{'='*60}")

    # Save results
    import json
    os.makedirs("results_clean/diagnostics/", exist_ok=True)
    result_summary = {
        "dataset": ds_name,
        "seed": args.seed,
        "noise_type": args.noise_type,
        "noise_ratio": args.noise_ratio,
        "noisy_graph": {name: {k: float(v) for k, v in m.items()} for name, m in results.items()},
        "noisy_bad_edge_detection": {k: float(v) for k, v in det.items()},
        "clean_oracle": {
            "pearson_r": float(pearson_r_clean),
            "auc": float(auc_clean),
            "ap": float(ap_clean),
        },
        "diagnostics": {k: float(v) for k, v in diagnostics.items()},
        "feasible": bool(best_auc >= 0.75 or best_auc_clean >= 0.75),
    }
    with open(f"results_clean/diagnostics/verify_idea_{ds_name}_{args.seed}.json", "w") as f:
        json.dump(result_summary, f, indent=2)


if __name__ == "__main__":
    main()
