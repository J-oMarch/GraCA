"""
Master experiment runner for the core experiment matrix.

Runs:
- Clean graph experiments: 6 datasets × 3 downstream models × 10 seeds × 8 methods
- Noisy graph experiments: 4 datasets × 3 downstream models × 10 seeds × 7 methods × 4 noise types × 4 noise ratios

Usage:
    # Run everything
    python scripts/run_core_matrix.py --phase all

    # Run clean only
    python scripts/run_core_matrix.py --phase clean --seeds 0-9

    # Run noisy only for a specific dataset
    python scripts/run_core_matrix.py --phase noisy --dataset Cora --seeds 0-9
"""
import sys
import os
import argparse
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CLEAN_DATASETS = ["Cora", "CiteSeer", "PubMed", "AmazonComputers", "AmazonPhoto", "CoauthorCS"]
NOISY_DATASETS = ["Cora", "CiteSeer", "PubMed", "AmazonComputers"]
DOWNSTREAM_MODELS = ["GCN", "GAT", "GraphSAGE"]
NOISE_TYPES = ["low_feature_similarity", "cross_class_train_safe", "random_inter_community", "cross_class_oracle"]
NOISE_RATIOS = [0.05, 0.10, 0.20, 0.30]

CONFIG_MAP = {
    "Cora": "configs/graca_lite_cora.yaml",
    "CiteSeer": "configs/graca_lite_citeseer.yaml",
    "PubMed": "configs/graca_lite_pubmed.yaml",
    "AmazonComputers": "configs/graca_lite_amazoncomputers.yaml",
    "AmazonPhoto": "configs/graca_lite_amazonphoto.yaml",
    "CoauthorCS": "configs/graca_lite_coauthorcs.yaml",
}


def run_cmd(cmd, description):
    """Run a command and return success/failure."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"{status}: {description} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "clean", "noisy", "ablation", "oracle"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0-9",
                        help="Seed range: '0-9' or '0,1,2' or '0'")
    parser.add_argument("--noise_type", type=str, default=None)
    parser.add_argument("--noise_ratio", type=float, default=None)
    parser.add_argument("--method", type=str, default=None,
                        help="For clean: all, graca, baselines. For noisy: all, graca, random, homophily, original")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Parse seeds
    if "-" in args.seeds:
        start, end = args.seeds.split("-")
        seeds = list(range(int(start), int(end) + 1))
    elif "," in args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [int(args.seeds)]

    datasets_clean = [args.dataset] if args.dataset else CLEAN_DATASETS
    datasets_noisy = [args.dataset] if args.dataset else NOISY_DATASETS
    noise_types = [args.noise_type] if args.noise_type else NOISE_TYPES
    noise_ratios = [args.noise_ratio] if args.noise_ratio else NOISE_RATIOS

    total_runs = 0
    successful = 0
    failed = 0

    # ============ CLEAN GRAPH EXPERIMENTS ============
    if args.phase in ("all", "clean"):
        print("\n" + "#" * 60)
        print("# CLEAN GRAPH EXPERIMENTS")
        print("#" * 60)

        for ds in datasets_clean:
            config = CONFIG_MAP.get(ds)
            if not config:
                print(f"WARNING: No config for {ds}, skipping")
                continue

            for seed in seeds:
                # GraCA-lite
                if args.method in (None, "all", "graca"):
                    cmd = f"python scripts/run_graca.py --config {config} --seed {seed}"
                    if not args.dry_run:
                        if run_cmd(cmd, f"GraCA-lite on {ds} seed={seed}"):
                            successful += 1
                        else:
                            failed += 1
                    total_runs += 1

                # Baselines (Original, Random, Homophily, etc.)
                if args.method in (None, "all", "baselines"):
                    for baseline in ["original", "random", "homophily"]:
                        cmd = f"python scripts/run_baselines.py --config {config} --seed {seed} --baseline {baseline}"
                        if not args.dry_run:
                            if run_cmd(cmd, f"{baseline} on {ds} seed={seed}"):
                                successful += 1
                            else:
                                failed += 1
                        total_runs += 1

    # ============ NOISY GRAPH EXPERIMENTS ============
    if args.phase in ("all", "noisy"):
        print("\n" + "#" * 60)
        print("# NOISY GRAPH EXPERIMENTS")
        print("#" * 60)

        for ds in datasets_noisy:
            config = CONFIG_MAP.get(ds)
            if not config:
                print(f"WARNING: No config for {ds}, skipping")
                continue

            for noise_type in noise_types:
                for noise_ratio in noise_ratios:
                    for seed in seeds:
                        cmd = (f"python scripts/run_noisy_edge_experiment.py "
                               f"--config {config} --seed {seed} "
                               f"--noise_type {noise_type} --noise_ratio {noise_ratio}")
                        if args.method:
                            cmd += f" --method {args.method}"

                        if not args.dry_run:
                            if run_cmd(cmd, f"Noisy {ds} {noise_type} {noise_ratio} seed={seed}"):
                                successful += 1
                            else:
                                failed += 1
                        total_runs += 1

    # ============ ABLATION EXPERIMENTS ============
    if args.phase in ("all", "ablation"):
        print("\n" + "#" * 60)
        print("# ABLATION EXPERIMENTS")
        print("#" * 60)

        ablation_datasets = ["Cora", "CiteSeer", "PubMed"] if not args.dataset else [args.dataset]

        for ds in ablation_datasets:
            config = CONFIG_MAP.get(ds)
            if not config:
                continue

            for seed in seeds:
                cmd = (f"python scripts/run_ablation_noisy.py "
                       f"--config {config} --seed {seed} "
                       f"--noise_type low_feature_similarity --noise_ratio 0.10")
                if not args.dry_run:
                    if run_cmd(cmd, f"Ablation on {ds} seed={seed}"):
                        successful += 1
                    else:
                        failed += 1
                total_runs += 1

    # ============ ORACLE EXPERIMENTS ============
    if args.phase in ("all", "oracle"):
        print("\n" + "#" * 60)
        print("# ORACLE EXPERIMENTS")
        print("#" * 60)

        oracle_datasets = ["Cora", "CiteSeer", "PubMed"] if not args.dataset else [args.dataset]

        for ds in oracle_datasets:
            oracle_config = f"configs/oracle_{ds.lower()}.yaml"
            if not os.path.exists(oracle_config):
                oracle_config = CONFIG_MAP.get(ds)
                if not oracle_config:
                    continue

            for seed in seeds:
                cmd = f"python scripts/run_oracle.py --config {oracle_config} --seed {seed}"
                if not args.dry_run:
                    if run_cmd(cmd, f"Oracle on {ds} seed={seed}"):
                        successful += 1
                    else:
                        failed += 1
                total_runs += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY")
    print(f"=" * 60)
    print(f"Total runs: {total_runs}")
    if not args.dry_run:
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
    else:
        print("(Dry run - no experiments executed)")


if __name__ == "__main__":
    main()
