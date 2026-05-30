"""
Analyze GraCA edge scores on noisy graphs to diagnose detection capability.

Outputs:
1. Score distributions for injected bad edges vs clean edges
2. AUC/AP for each score component (D, M, rho, H, R, P)
3. Results saved to results_clean/diagnostics/

Usage:
    python scripts/analyze_edge_scores.py --config configs/graca_lite_cora.yaml --seed 0 \
        --noise_type low_feature_similarity --noise_ratio 0.10
"""
import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.load_data import load_dataset
from src.training.train_proxy import train_proxy
from src.graca.gradient_collector import collect_hidden_gradients
from src.graca.edge_scoring import compute_edge_scores, average_undirected_scores, compute_rho_score
from src.graca.pseudo_label import compute_soft_pseudo_labels
from src.eval.noise_injection import inject_noise
from src.utils.logger import get_logger


def compute_auc(y_true, y_score):
    """Compute AUC using trapezoidal rule."""
    # Sort by score descending
    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    y_score = y_score[sorted_idx]

    # Compute TPR and FPR
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

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

    tp = 0
    ap = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            ap += tp / (i + 1)

    return ap / n_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", type=str, default="low_feature_similarity",
                        choices=["cross_class_train_safe", "cross_class_oracle",
                                 "low_feature_similarity", "random_inter_community"])
    parser.add_argument("--noise_ratio", type=float, default=0.10)
    parser.add_argument("--output_dir", type=str, default="results_clean/diagnostics/")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("analyze_edge_scores")
    set_seed(args.seed)

    device = get_device(config)
    ds_name = config["dataset"]["name"]

    # Load data
    data, num_features, num_classes = load_dataset(config)
    data = data.to(device)

    # Inject noise
    logger.info(f"Injecting {args.noise_type} noise at {args.noise_ratio}...")
    noise_result = inject_noise(
        edge_index=data.edge_index.cpu(),
        num_nodes=data.num_nodes,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        x=data.x.cpu(),
        y=data.y.cpu(),
        train_mask=data.train_mask.cpu(),
        seed=args.seed,
    )
    noisy_edge_index = noise_result["noisy_edge_index"].to(device)
    bad_edge_mask = noise_result["bad_edge_mask"]

    logger.info(f"Original edges: {data.edge_index.shape[1]}, "
                f"Noisy edges: {noisy_edge_index.shape[1]}, "
                f"Injected pairs: {noise_result['num_injected_edges']}")

    # Train proxy model
    logger.info("Training ProxyGNN...")
    model, teacher, train_log, saved_checkpoints = train_proxy(
        config, data, num_features, num_classes, device, args.seed
    )

    # Collect gradients on noisy graph
    logger.info("Collecting gradients on noisy graph...")
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    unlabeled_mask = ~train_mask

    teacher_probs = teacher.predict(x, data.edge_index)  # use clean graph for teacher
    pseudo_cfg = config.get("pseudo", {})
    tau = pseudo_cfg.get("tau", 0.6)
    alpha = pseudo_cfg.get("alpha", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

    q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )

    scoring_cfg = config.get("scoring", {})
    deterministic = scoring_cfg.get("deterministic", True)

    grad_result = collect_hidden_gradients(
        model=model, x=x, edge_index=noisy_edge_index, y=y,
        teacher_probs=teacher_probs, rho_train=rho_train,
        train_mask=train_mask, unlabeled_mask=unlabeled_mask,
        lambda_s=scoring_cfg.get("lambda_s", 1.0),
        collect_layer=scoring_cfg.get("collect_layer", "all"),
        deterministic=deterministic,
    )

    grad = grad_result["grad"]

    # Compute rho_score
    rho_score = compute_rho_score(
        teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
    )

    # Compute all edge scores
    logger.info("Computing edge scores...")
    edge_scores = compute_edge_scores(
        grad=grad, edge_index=noisy_edge_index, rho_score=rho_score,
        num_nodes=data.num_nodes, eta=scoring_cfg.get("eta", 1.0), epsilon_rho=eps_rho,
    )

    # Get scores
    D = edge_scores["D"].cpu().numpy()
    M = edge_scores["M"].cpu().numpy()
    rho = edge_scores["rho_vu"].cpu().numpy()
    H = edge_scores["H"].cpu().numpy()
    R = edge_scores["R"].cpu().numpy()
    P = edge_scores["P"].cpu().numpy()

    # Average undirected scores for P
    undirected = config.get("dataset", {}).get("undirected", True)
    if undirected:
        P_avg = average_undirected_scores(noisy_edge_index, edge_scores["P"]).cpu().numpy()
    else:
        P_avg = P

    bad_np = bad_edge_mask.numpy().astype(int)

    # Compute AUC and AP for each score
    results = {}
    for name, scores in [("D", D), ("M", M), ("rho", rho), ("H", H), ("R", R), ("P", P), ("P_avg", P_avg)]:
        # For D: negative D should predict bad edges (bad edges have lower direction consistency)
        if name == "D":
            auc = compute_auc(bad_np, -scores)  # use -D as predictor
            ap = compute_ap(bad_np, -scores)
        elif name == "H":
            # Higher H means edge is helpful, so -H predicts bad
            auc = compute_auc(bad_np, -scores)
            ap = compute_ap(bad_np, -scores)
        else:
            # Higher R, P, rho, M should predict bad
            auc = compute_auc(bad_np, scores)
            ap = compute_ap(bad_np, scores)

        results[name] = {"auc": auc, "ap": ap}

        # Distribution stats
        bad_scores = scores[bad_np == 1]
        clean_scores = scores[bad_np == 0]
        results[name]["bad_mean"] = float(bad_scores.mean()) if len(bad_scores) > 0 else 0
        results[name]["bad_std"] = float(bad_scores.std()) if len(bad_scores) > 0 else 0
        results[name]["clean_mean"] = float(clean_scores.mean()) if len(clean_scores) > 0 else 0
        results[name]["clean_std"] = float(clean_scores.std()) if len(clean_scores) > 0 else 0

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save score distributions
    dist_df = pd.DataFrame({
        "edge_idx": range(len(D)),
        "D": D, "M": M, "rho": rho, "H": H, "R": R, "P": P, "P_avg": P_avg,
        "is_bad": bad_np,
    })
    dist_path = os.path.join(args.output_dir,
        f"edge_score_distribution_{ds_name}_{args.noise_type}_{args.noise_ratio}_{args.seed}.csv")
    dist_df.to_csv(dist_path, index=False)

    # Save AUC/AP summary
    summary_rows = []
    for name, metrics in results.items():
        summary_rows.append({
            "dataset": ds_name,
            "noise_type": args.noise_type,
            "noise_ratio": args.noise_ratio,
            "seed": args.seed,
            "score_component": name,
            "auc": metrics["auc"],
            "ap": metrics["ap"],
            "bad_mean": metrics["bad_mean"],
            "bad_std": metrics["bad_std"],
            "clean_mean": metrics["clean_mean"],
            "clean_std": metrics["clean_std"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir,
        f"score_auc_summary_{ds_name}_{args.noise_type}_{args.noise_ratio}_{args.seed}.csv")
    summary_df.to_csv(summary_path, index=False)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Edge Score Diagnostic Results")
    logger.info("=" * 60)
    logger.info(f"Dataset: {ds_name}, Noise: {args.noise_type} {args.noise_ratio}")
    logger.info(f"Injected pairs: {noise_result['num_injected_edges']}")
    logger.info("-" * 60)
    logger.info(f"{'Score':<10} {'AUC':>8} {'AP':>8} {'Bad Mean':>10} {'Clean Mean':>10}")
    logger.info("-" * 60)
    for name, metrics in results.items():
        logger.info(f"{name:<10} {metrics['auc']:>8.4f} {metrics['ap']:>8.4f} "
                     f"{metrics['bad_mean']:>10.4f} {metrics['clean_mean']:>10.4f}")
    logger.info("=" * 60)

    # Check if any score component has meaningful signal
    best_auc = max(m["auc"] for m in results.values())
    if best_auc < 0.55:
        logger.warning("WARNING: All AUC values close to 0.5 - gradient signal does NOT distinguish bad edges!")
        logger.warning("Consider modifying scoring components before running main experiments.")
    elif best_auc < 0.65:
        logger.info("NOTE: AUC is modest. Scoring has weak signal for bad edge detection.")
    else:
        logger.info(f"Good signal detected. Best AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
