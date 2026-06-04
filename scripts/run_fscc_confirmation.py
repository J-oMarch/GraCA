#!/usr/bin/env python3
"""
FSCC Matched-Budget Confirmation Rerun: Primary, controls, and heterophily stages.

Usage:
    python scripts/run_fscc_confirmation.py --stage primary \
        --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/primary

    python scripts/run_fscc_confirmation.py --stage controls \
        --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/controls

    python scripts/run_fscc_confirmation.py --stage heterophily \
        --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily
"""
import os
import sys
import argparse
import logging
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_adaptive_grage_search import (
    run_experiment_matrix,
    run_random_matched_baseline,
    DEFAULT_CONFIG,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Method configurations ───
METHODS = [
    {"name": "Feature-only", "type": "feature_only"},
    {"name": "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "type": "hybrid_baseline",
     "lambda_pos": 0.1, "lambda_neg": 0.5, "score_ratio": 0.3},
    {"name": "MCGC-cw3.0-lp0.1-ln0.5", "type": "mcgc",
     "lambda_pos": 0.1, "lambda_neg": 0.5,
     "consistency_weight": 3.0, "score_ratio": 0.3,
     "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
     "total_epochs": 200},
    {"name": "DegreeAwareRandom", "type": "degree_aware_random"},
    {"name": "GCN-Jaccard", "type": "jaccard"},
]


def run_primary(device, output_dir):
    """Primary: 3 datasets × 1 noise × 20 seeds × 6 methods = 360 runs."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        noise_types=["feature_similar_cross_class"],
        noise_ratio=0.3,
        seeds=list(range(20)),
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=METHODS,
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_controls(device, output_dir):
    """Controls: 3 datasets × 3 noise × 10 seeds × 6 methods = 540 runs."""
    return run_experiment_matrix(
        datasets=["Cora", "CiteSeer", "PubMed"],
        noise_types=["cross_class_oracle", "low_feature_similarity", "degree_aligned_random"],
        noise_ratio=0.3,
        seeds=list(range(10)),
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=METHODS,
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def run_heterophily(device, output_dir):
    """Heterophily: 3 datasets × 2 noise × 5 seeds × 6 methods = 180 runs.

    Gracefully skips missing datasets.
    """
    hetero_datasets = []
    for name in ["Texas", "Wisconsin", "Actor"]:
        try:
            from src.data.load_data import load_dataset
            test_config = DEFAULT_CONFIG.copy()
            test_config["dataset"]["name"] = name
            load_dataset(test_config)
            hetero_datasets.append(name)
        except Exception as e:
            logger.warning(f"Skipping heterophily dataset {name}: {e}")

    if not hetero_datasets:
        logger.warning("No heterophily datasets available, skipping stage.")
        return None

    return run_experiment_matrix(
        datasets=hetero_datasets,
        noise_types=["feature_similar_cross_class", "degree_aligned_random"],
        noise_ratio=0.3,
        seeds=list(range(5)),
        downstream_model="GCN",
        prune_ratio=0.2,
        method_configs=METHODS,
        device=device,
        output_dir=output_dir,
        include_random_matched=True,
    )


def main():
    parser = argparse.ArgumentParser(description="FSCC Confirmation Rerun")
    parser.add_argument("--stage", choices=["primary", "controls", "heterophily"],
                        required=True, help="Experiment stage")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    base = "experiments/2026-06-04-fscc-confirmation-rerun/logs"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{base}/{args.stage}"

    import time
    start = time.time()

    if args.stage == "primary":
        df = run_primary(device, output_dir)
    elif args.stage == "controls":
        df = run_controls(device, output_dir)
    elif args.stage == "heterophily":
        df = run_heterophily(device, output_dir)

    elapsed = time.time() - start
    logger.info(f"Stage '{args.stage}' completed in {elapsed:.1f}s")

    if df is not None and len(df) > 0:
        summary = df.groupby("method")["test_acc"].agg(["mean", "std", "count"])
        summary = summary.sort_values("mean", ascending=False)
        logger.info(f"\n{summary.to_string()}")


if __name__ == "__main__":
    main()
