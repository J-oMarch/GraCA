"""
Master experiment runner: runs GraCA-lite, baselines, and oracle for all datasets and seeds.
"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["Cora", "CiteSeer", "PubMed"]
HETERO_DATASETS = ["Actor", "Texas", "Cornell", "Wisconsin"]
SEEDS = [0, 1, 2, 3, 4]


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {desc} failed with return code {result.returncode}")
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-graca", action="store_true")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASETS + HETERO_DATASETS
    seeds = args.seeds if args.seeds else SEEDS
    seeds_str = " ".join(str(s) for s in seeds)

    for ds in datasets:
        config = f"configs/graca_lite_{ds.lower()}.yaml"
        if not os.path.exists(config):
            print(f"Config not found: {config}, skipping {ds}")
            continue

        # GraCA-lite
        if not args.skip_graca:
            for seed in seeds:
                run_cmd(
                    f"conda run -n graca python scripts/run_graca.py --config {config} --seed {seed}",
                    f"GraCA-lite on {ds} seed={seed}"
                )

        # Baselines
        if not args.skip_baselines:
            for seed in seeds:
                run_cmd(
                    f"conda run -n graca python scripts/run_baselines.py --config {config} --seed {seed}",
                    f"Baselines on {ds} seed={seed}"
                )

        # Oracle
        if not args.skip_oracle:
            oracle_config = f"configs/oracle_{ds.lower()}.yaml"
            if os.path.exists(oracle_config):
                for seed in seeds:
                    run_cmd(
                        f"conda run -n graca python scripts/run_oracle.py --config {oracle_config} --seed {seed}",
                        f"Oracle on {ds} seed={seed}"
                    )

        # Ablation
        if not args.skip_ablation:
            for seed in seeds:
                run_cmd(
                    f"conda run -n graca python scripts/run_ablation.py --config {config} --seed {seed}",
                    f"Ablation on {ds} seed={seed}"
                )

    # Aggregate
    run_cmd(
        "conda run -n graca python scripts/aggregate_results.py",
        "Aggregating results"
    )


if __name__ == "__main__":
    main()
