"""
GraGE Experiment Runner: Run the full experiment matrix.

Matrix: 6 datasets × 5 noise types × N seeds × 7 methods

Datasets: Cora, CiteSeer, PubMed, Actor, Texas, Wisconsin
Noise types: cross_class_oracle, cross_class_train_safe, low_feature_similarity,
             random_inter_community, degree_aligned_random
Methods: Original+Noise, Random-Matched, GCN-Jaccard, DegreeAwareRandom,
         Feature-only, EdgeBench, EdgeInfluence-Oracle

Usage:
    # Run all datasets and noise types
    python scripts/run_grage_experiments.py

    # Run specific datasets
    python scripts/run_grage_experiments.py --datasets Cora CiteSeer PubMed

    # Run specific noise types
    python scripts/run_grage_experiments.py --noise_types cross_class_oracle low_feature_similarity

    # Run with custom seeds
    python scripts/run_grage_experiments.py --seeds 0 1 2 3 4

    # Resume from specific dataset/noise combination
    python scripts/run_grage_experiments.py --resume_from Cora_cross_class_oracle
"""
import sys
import os
import argparse
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATASETS = ["Cora", "CiteSeer", "PubMed", "Actor", "Texas", "Wisconsin"]
NOISE_TYPES = [
    "cross_class_oracle",
    "cross_class_train_safe",
    "low_feature_similarity",
    "random_inter_community",
    "degree_aligned_random",
]
SEEDS = [0, 1, 2, 3, 4]
PRUNE_RATIO = 0.20
NOISE_RATIO = 0.20


def get_config_path(dataset):
    """Get config file path for dataset."""
    return f"configs/graca_lite_{dataset.lower()}.yaml"


def run_single_experiment(dataset, noise_type, seed, prune_ratio, noise_ratio,
                          output_dir, log_dir, resume_from=None):
    """Run a single experiment (dataset × noise_type × seed)."""
    config = get_config_path(dataset)
    if not os.path.exists(config):
        print(f"  [SKIP] Config not found: {config}")
        return False

    # Check if already completed
    csv_path = f"{output_dir}/controlled_v2_results.csv"
    run_id_prefix = f"noisy_edge_{noise_type}"
    if os.path.exists(csv_path) and resume_from:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('dataset') == dataset and
                    row.get('noise_type') == noise_type and
                    str(row.get('seed')) == str(seed) and
                    row.get('method') == 'EdgeBench'):
                    print(f"  [SKIP] Already completed: {dataset}/{noise_type}/seed{seed}")
                    return True

    cmd = [
        "conda", "run", "-n", "graca",
        "python", "scripts/run_controlled_comparison.py",
        "--config", config,
        "--seed", str(seed),
        "--prune_ratio", str(prune_ratio),
        "--noisy",
        "--noise_type", noise_type,
        "--noise_ratio", str(noise_ratio),
        "--output_dir", output_dir,
    ]

    log_file = f"{log_dir}/{dataset}_{noise_type}_seed{seed}.log"
    os.makedirs(log_dir, exist_ok=True)

    print(f"  Running: {dataset}/{noise_type}/seed{seed}")
    t0 = time.time()

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=600,  # 10 minutes per experiment
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"    ✓ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"    ✗ Failed (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout (600s)")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GraGE Experiment Runner")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Datasets to run")
    parser.add_argument("--noise_types", nargs="+", default=NOISE_TYPES,
                        help="Noise types to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Random seeds")
    parser.add_argument("--prune_ratio", type=float, default=PRUNE_RATIO,
                        help="Edge pruning ratio")
    parser.add_argument("--noise_ratio", type=float, default=NOISE_RATIO,
                        help="Noise injection ratio")
    parser.add_argument("--output_dir", type=str, default="results_clean/grage_v1/",
                        help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs/grage_experiments/",
                        help="Log directory")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from specific dataset_noise combination")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    # Validate datasets
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"Error: Unknown dataset '{ds}'. Available: {DATASETS}")
            sys.exit(1)

    # Validate noise types
    for nt in args.noise_types:
        if nt not in NOISE_TYPES:
            print(f"Error: Unknown noise type '{nt}'. Available: {NOISE_TYPES}")
            sys.exit(1)

    # Calculate total experiments
    total = len(args.datasets) * len(args.noise_types) * len(args.seeds)
    print(f"\n{'='*60}")
    print(f"GraGE Experiment Runner")
    print(f"{'='*60}")
    print(f"Datasets: {args.datasets}")
    print(f"Noise types: {args.noise_types}")
    print(f"Seeds: {args.seeds}")
    print(f"Prune ratio: {args.prune_ratio}")
    print(f"Noise ratio: {args.noise_ratio}")
    print(f"Total experiments: {total}")
    print(f"Output: {args.output_dir}")
    print(f"Logs: {args.log_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN - Commands that would be executed:")
        for dataset in args.datasets:
            for noise_type in args.noise_types:
                for seed in args.seeds:
                    config = get_config_path(dataset)
                    cmd = f"python scripts/run_controlled_comparison.py --config {config} --seed {seed} --prune_ratio {args.prune_ratio} --noisy --noise_type {noise_type} --noise_ratio {args.noise_ratio} --output_dir {args.output_dir}"
                    print(f"  {cmd}")
        return

    # Run experiments
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for dataset in args.datasets:
        for noise_type in args.noise_types:
            print(f"\n[{completed+failed+skipped+1}/{total}] {dataset} + {noise_type}")

            for seed in args.seeds:
                success = run_single_experiment(
                    dataset=dataset,
                    noise_type=noise_type,
                    seed=seed,
                    prune_ratio=args.prune_ratio,
                    noise_ratio=args.noise_ratio,
                    output_dir=args.output_dir,
                    log_dir=args.log_dir,
                    resume_from=args.resume_from,
                )

                if success:
                    completed += 1
                else:
                    failed += 1

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Experiment Summary")
    print(f"{'='*60}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Results: {args.output_dir}/controlled_v2_results.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
