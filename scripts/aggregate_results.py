"""
Aggregate experiment results and print summary tables.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.aggregate import aggregate_results, print_main_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/main/")
    parser.add_argument("--output", type=str, default="results/aggregated/main_summary.csv")
    parser.add_argument("--exclude_oracle", type=str, default="true")
    parser.add_argument("--include_baselines", action="store_true")
    args = parser.parse_args()

    exclude_oracle = args.exclude_oracle.lower() == "true"

    # Aggregate main results
    print("\n" + "=" * 60)
    print("Main Results (GraCA-lite)")
    print("=" * 60)
    aggregate_results(args.input, args.output, exclude_oracle)
    print_main_table(args.input, exclude_oracle)

    # Aggregate baselines if requested
    if args.include_baselines:
        print("\n" + "=" * 60)
        print("Baseline Results")
        print("=" * 60)
        aggregate_results(
            "results/baselines/",
            "results/aggregated/baseline_summary.csv",
            exclude_oracle,
        )
        print_main_table("results/baselines/", exclude_oracle)

    # Aggregate ablation
    if os.path.exists("results/ablation/"):
        print("\n" + "=" * 60)
        print("Ablation Results")
        print("=" * 60)
        aggregate_results(
            "results/ablation/",
            "results/aggregated/ablation_summary.csv",
            exclude_oracle,
        )

    # Aggregate oracle
    if os.path.exists("results/oracle/"):
        print("\n" + "=" * 60)
        print("Oracle Results (upper bound, diagnostic only)")
        print("=" * 60)
        aggregate_results(
            "results/oracle/",
            "results/aggregated/oracle_summary.csv",
            exclude_oracle=False,
        )


if __name__ == "__main__":
    main()
