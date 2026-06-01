"""
Build GraGE paper tables from controlled_v2 results.

Generates:
1. Tab.1: Main Results (Downstream Accuracy) - 6 datasets × 7 methods × 3 models
2. Tab.2: Edge Detection AUC (MAIN TABLE) - EdgeBench vs EdgeInfluence vs Feature-only
3. Tab.3: Noise Type Analysis - AUC per noise type
4. Tab.4: Teacher Sensitivity - EdgeInfluence-Pseudo vs EdgeBench (no teacher needed)

Usage:
    python scripts/build_grage_tables.py --results_dir results_clean/controlled_v2/
    python scripts/build_grage_tables.py --results_dir results_clean/grage_v1/ --output_dir paper_tables_grage/
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_results(results_dir):
    """Load all CSV files from results directory."""
    csv_files = list(Path(results_dir).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    dfs = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f)
            df["_source"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Error reading {f}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files loaded")

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def build_tab1_main_results(df, output_dir):
    """Tab.1: Main Results - Downstream accuracy comparison.

    Format: Dataset | Model | Original | Random | GCN-Jaccard | DegreeAware | Feature-only | EdgeBench | EI-Oracle
    """
    # Filter to noisy experiments only
    noisy_df = df[df["experiment_type"] == "noisy_edge"].copy()

    if noisy_df.empty:
        print("Warning: No noisy edge experiments found")
        return

    methods = ["Original+Noise", "Random-Matched", "GCN-Jaccard", "DegreeAwareRandom",
               "Feature-only", "EdgeBench", "EdgeInfluence-Oracle"]

    # Check which methods are available
    available_methods = [m for m in methods if m in noisy_df["method"].unique()]
    print(f"Available methods: {available_methods}")

    # Group by dataset, method, downstream_model
    grouped = noisy_df.groupby(["dataset", "method", "downstream_model"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        test_f1_mean=("test_f1", "mean"),
        test_f1_std=("test_f1", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    # Pivot to wide format
    pivot_acc = grouped.pivot_table(
        index=["dataset", "downstream_model"],
        columns="method",
        values=["test_acc_mean", "test_acc_std"],
        aggfunc="first"
    )

    # Build table
    rows = []
    for ds in sorted(noisy_df["dataset"].unique()):
        for model in sorted(noisy_df["downstream_model"].unique()):
            row = {"Dataset": ds, "Model": model}
            for method in available_methods:
                mask = (grouped["dataset"] == ds) & \
                       (grouped["method"] == method) & \
                       (grouped["downstream_model"] == model)
                subset = grouped[mask]
                if not subset.empty:
                    mean = subset["test_acc_mean"].values[0]
                    std = subset["test_acc_std"].values[0]
                    row[f"{method}_mean"] = mean
                    row[f"{method}_std"] = std
                    row[method] = f"{mean:.4f} ± {std:.4f}"
                else:
                    row[method] = "—"
            rows.append(row)

    tab1 = pd.DataFrame(rows)
    tab1.to_csv(f"{output_dir}/tab1_main_results.csv", index=False)
    print(f"Tab.1 saved: {output_dir}/tab1_main_results.csv")

    return tab1


def build_tab2_edge_auc(df, output_dir):
    """Tab.2: Edge Detection AUC (MAIN TABLE).

    Format: Dataset | NoiseType | EdgeBench AUC | EdgeInfluence AUC | Feature-only AUC
    """
    noisy_df = df[df["experiment_type"] == "noisy_edge"].copy()

    if noisy_df.empty:
        print("Warning: No noisy edge experiments found")
        return

    # Extract bad-edge detection metrics
    methods_with_detection = ["EdgeBench", "EdgeInfluence-Pseudo", "EdgeInfluence-Oracle", "Feature-only"]

    rows = []
    for ds in sorted(noisy_df["dataset"].unique()):
        for noise in sorted(noisy_df["noise_type"].unique()):
            row = {"Dataset": ds, "NoiseType": noise}

            for method in methods_with_detection:
                mask = (noisy_df["dataset"] == ds) & \
                       (noisy_df["noise_type"] == noise) & \
                       (noisy_df["method"] == method)
                subset = noisy_df[mask]

                if not subset.empty:
                    # Get bad-edge F1 (as proxy for AUC)
                    f1_mean = subset["bad_edge_f1"].mean()
                    f1_std = subset["bad_edge_f1"].std()
                    prec_mean = subset["bad_edge_precision"].mean()
                    rec_mean = subset["bad_edge_recall"].mean()

                    row[f"{method}_F1"] = f"{f1_mean:.4f} ± {f1_std:.4f}"
                    row[f"{method}_F1_mean"] = f1_mean
                    row[f"{method}_Precision"] = f"{prec_mean:.4f}"
                    row[f"{method}_Recall"] = f"{rec_mean:.4f}"
                else:
                    row[f"{method}_F1"] = "—"
                    row[f"{method}_F1_mean"] = 0.0

            rows.append(row)

    tab2 = pd.DataFrame(rows)
    tab2.to_csv(f"{output_dir}/tab2_edge_auc.csv", index=False)
    print(f"Tab.2 saved: {output_dir}/tab2_edge_auc.csv")

    return tab2


def build_tab3_noise_type_analysis(df, output_dir):
    """Tab.3: Noise Type Analysis - How each method performs across noise types.

    Format: Method | cross_class_oracle | cross_class_train_safe | low_feature_similarity | ...
    """
    noisy_df = df[df["experiment_type"] == "noisy_edge"].copy()

    if noisy_df.empty:
        print("Warning: No noisy edge experiments found")
        return

    methods = ["Original+Noise", "Random-Matched", "GCN-Jaccard", "DegreeAwareRandom",
               "Feature-only", "EdgeBench", "EdgeInfluence-Oracle"]
    noise_types = sorted(noisy_df["noise_type"].unique())

    rows = []
    for method in methods:
        row = {"Method": method}
        for noise in noise_types:
            mask = (noisy_df["method"] == method) & (noisy_df["noise_type"] == noise)
            subset = noisy_df[mask]
            if not subset.empty:
                acc_mean = subset["test_acc"].mean()
                acc_std = subset["test_acc"].std()
                row[noise] = f"{acc_mean:.4f} ± {acc_std:.4f}"
                row[f"{noise}_mean"] = acc_mean
            else:
                row[noise] = "—"
                row[f"{noise}_mean"] = 0.0
        rows.append(row)

    tab3 = pd.DataFrame(rows)
    tab3.to_csv(f"{output_dir}/tab3_noise_type_analysis.csv", index=False)
    print(f"Tab.3 saved: {output_dir}/tab3_noise_type_analysis.csv")

    return tab3


def build_tab4_teacher_sensitivity(df, output_dir):
    """Tab.4: Teacher Sensitivity - EdgeBench vs EdgeInfluence-Pseudo.

    Shows that EdgeBench doesn't need a teacher while EdgeInfluence-Pseudo does.
    """
    noisy_df = df[df["experiment_type"] == "noisy_edge"].copy()

    if noisy_df.empty:
        print("Warning: No noisy edge experiments found")
        return

    methods = ["EdgeBench", "EdgeInfluence-Pseudo", "EdgeInfluence-Oracle"]

    rows = []
    for ds in sorted(noisy_df["dataset"].unique()):
        for method in methods:
            mask = (noisy_df["dataset"] == ds) & (noisy_df["method"] == method)
            subset = noisy_df[mask]
            if not subset.empty:
                acc_mean = subset["test_acc"].mean()
                acc_std = subset["test_acc"].std()
                f1_mean = subset["bad_edge_f1"].mean()

                # Check if teacher was used
                notes = subset["notes"].iloc[0] if "notes" in subset.columns else ""
                uses_teacher = "clean_teacher=False" in notes or "clean_teacher=True" in notes
                oracle_label = "oracle_label=True" in notes

                rows.append({
                    "Dataset": ds,
                    "Method": method,
                    "Test Acc": f"{acc_mean:.4f} ± {acc_std:.4f}",
                    "Bad-edge F1": f"{f1_mean:.4f}",
                    "Uses Teacher": "Yes" if uses_teacher else "No",
                    "Oracle Labels": "Yes" if oracle_label else "No",
                    "Acc_mean": acc_mean,
                    "F1_mean": f1_mean,
                })

    tab4 = pd.DataFrame(rows)
    tab4.to_csv(f"{output_dir}/tab4_teacher_sensitivity.csv", index=False)
    print(f"Tab.4 saved: {output_dir}/tab4_teacher_sensitivity.csv")

    return tab4


def build_summary_stats(df, output_dir):
    """Build summary statistics for the paper."""
    noisy_df = df[df["experiment_type"] == "noisy_edge"].copy()

    if noisy_df.empty:
        return

    summary = {
        "total_experiments": len(noisy_df),
        "datasets": sorted(noisy_df["dataset"].unique()),
        "noise_types": sorted(noisy_df["noise_type"].unique()),
        "methods": sorted(noisy_df["method"].unique()),
        "seeds": sorted(noisy_df["seed"].unique()),
    }

    # Per-method average accuracy
    method_acc = noisy_df.groupby("method")["test_acc"].agg(["mean", "std", "count"])
    summary["method_accuracy"] = method_acc.to_dict()

    # Save summary
    with open(f"{output_dir}/summary_stats.txt", "w") as f:
        f.write("GraGE Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total experiments: {summary['total_experiments']}\n")
        f.write(f"Datasets: {summary['datasets']}\n")
        f.write(f"Noise types: {summary['noise_types']}\n")
        f.write(f"Methods: {summary['methods']}\n")
        f.write(f"Seeds: {summary['seeds']}\n\n")
        f.write("Method Accuracy (mean ± std):\n")
        for method in method_acc.index:
            mean = method_acc.loc[method, "mean"]
            std = method_acc.loc[method, "std"]
            f.write(f"  {method}: {mean:.4f} ± {std:.4f}\n")

    print(f"Summary saved: {output_dir}/summary_stats.txt")


def main():
    parser = argparse.ArgumentParser(description="Build GraGE paper tables")
    parser.add_argument("--results_dir", type=str, default="results_clean/controlled_v2/",
                        help="Directory with controlled_v2 results")
    parser.add_argument("--output_dir", type=str, default="paper_tables_grage/",
                        help="Output directory for tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} rows")

    print("\nBuilding tables...")
    build_tab1_main_results(df, args.output_dir)
    build_tab2_edge_auc(df, args.output_dir)
    build_tab3_noise_type_analysis(df, args.output_dir)
    build_tab4_teacher_sensitivity(df, args.output_dir)
    build_summary_stats(df, args.output_dir)

    print(f"\nAll tables saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
