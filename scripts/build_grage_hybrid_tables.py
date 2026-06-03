#!/usr/bin/env python3
"""
Build tables and decision report for GraGE-Hybrid experiments.

Usage:
    python scripts/build_grage_hybrid_tables.py --results results_clean/grage_hybrid_sweep/results.csv
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_path):
    """Load results CSV."""
    df = pd.read_csv(results_path)
    return df


def compute_main_practical_acc(df, output_dir):
    """Table 1: Main practical accuracy comparison."""
    # Group by method, noise_type, dataset and compute mean/std
    grouped = df.groupby(["method", "noise_type", "dataset"]).agg({
        "test_acc": ["mean", "std"],
        "test_f1": ["mean", "std"],
        "val_acc": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]

    # Save
    output_path = os.path.join(output_dir, "main_practical_acc.csv")
    grouped.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return grouped


def compute_hybrid_vs_feature_only(df, output_dir):
    """Table 2: Hybrid methods vs Feature-only comparison."""
    # Get Feature-only baseline
    feature_only = df[df["method"] == "Feature-only"].copy()
    feature_only_mean = feature_only.groupby(["dataset", "noise_type", "seed"])["test_acc"].mean().reset_index()
    feature_only_mean.rename(columns={"test_acc": "feature_only_acc"}, inplace=True)

    # Get all hybrid methods
    hybrid_methods = df[df["method"] != "Feature-only"].copy()

    # Merge
    merged = hybrid_methods.merge(
        feature_only_mean,
        on=["dataset", "noise_type", "seed"],
        how="left"
    )
    merged["delta"] = merged["test_acc"] - merged["feature_only_acc"]

    # Compute mean delta per method/noise_type
    delta_summary = merged.groupby(["method", "noise_type"]).agg({
        "delta": ["mean", "std", "count"],
        "test_acc": "mean",
    }).reset_index()
    delta_summary.columns = ["_".join(col).strip("_") for col in delta_summary.columns]

    # Paired t-test for each method vs Feature-only
    ttest_results = []
    for method in merged["method"].unique():
        method_data = merged[merged["method"] == method]
        for noise in method_data["noise_type"].unique():
            noise_data = method_data[method_data["noise_type"] == noise]
            if len(noise_data) >= 3:
                t_stat, p_val = stats.ttest_rel(
                    noise_data["test_acc"].values,
                    noise_data["feature_only_acc"].values
                )
                ttest_results.append({
                    "method": method,
                    "noise_type": noise,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                })

    ttest_df = pd.DataFrame(ttest_results)

    # Save
    delta_path = os.path.join(output_dir, "hybrid_vs_feature_only.csv")
    delta_summary.to_csv(delta_path, index=False)
    print(f"Saved: {delta_path}")

    ttest_path = os.path.join(output_dir, "hybrid_vs_feature_only_ttest.csv")
    ttest_df.to_csv(ttest_path, index=False)
    print(f"Saved: {ttest_path}")

    return delta_summary, ttest_df


def compute_edge_detection_f1(df, output_dir):
    """Table 3: Edge detection F1 scores."""
    detection = df.groupby(["method", "noise_type"]).agg({
        "bad_edge_precision": "mean",
        "bad_edge_recall": "mean",
        "bad_edge_f1": "mean",
    }).reset_index()

    output_path = os.path.join(output_dir, "edge_detection_f1.csv")
    detection.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return detection


def compute_noise_type_breakdown(df, output_dir):
    """Table 4: Breakdown by noise type."""
    breakdown = df.groupby(["noise_type", "method"]).agg({
        "test_acc": ["mean", "std"],
        "bad_edge_f1": "mean",
    }).reset_index()
    breakdown.columns = ["_".join(col).strip("_") for col in breakdown.columns]

    output_path = os.path.join(output_dir, "noise_type_breakdown.csv")
    breakdown.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return breakdown


def compute_ablation_pos_neg_degree(df, output_dir):
    """Table 5: Ablation on positive/negative gradient and degree normalization."""
    # Compare methods with different lambda_pos, lambda_neg, degree_norm
    ablation = df[df["method"].str.contains("Hybrid-FO")].copy()

    ablation_summary = ablation.groupby(["method", "lambda_pos", "lambda_neg", "degree_norm"]).agg({
        "test_acc": ["mean", "std"],
        "bad_edge_f1": "mean",
    }).reset_index()
    ablation_summary.columns = ["_".join(col).strip("_") for col in ablation_summary.columns]

    output_path = os.path.join(output_dir, "ablation_pos_neg_degree.csv")
    ablation_summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return ablation_summary


def compute_clean_graph_non_degradation(df, output_dir):
    """Table 6: Clean graph non-degradation check."""
    # Check if any method significantly degrades on clean graphs
    # Note: This requires clean graph experiments (no noise)
    if "Original" not in df["method"].values:
        print("Note: No clean graph experiments found, skipping non-degradation check")
        return None

    clean_graph = df[df["noise_type"] == "clean"]
    degradation = clean_graph.groupby("method")["test_acc"].mean().reset_index()

    output_path = os.path.join(output_dir, "clean_graph_non_degradation.csv")
    degradation.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return degradation


def generate_decision_report(df, output_dir, delta_summary, ttest_df):
    """Generate the decision report."""
    report_path = os.path.join(output_dir, "GRAGE_HYBRID_DECISION_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# GraGE-Hybrid Decision Report\n\n")
        f.write("## Summary\n\n")

        # Key question: Does GraGE-Hybrid > Feature-only?
        f.write("### 1. Does GraGE-Hybrid exceed Feature-only?\n\n")

        feature_only_acc = df[df["method"] == "Feature-only"]["test_acc"].mean()
        f.write(f"Feature-only mean accuracy: {feature_only_acc:.4f}\n\n")

        hybrid_methods = df[df["method"] != "Feature-only"]
        hybrid_mean = hybrid_methods.groupby("method")["test_acc"].mean()
        best_hybrid = hybrid_mean.idxmax()
        best_hybrid_acc = hybrid_mean.max()

        f.write(f"Best hybrid method: {best_hybrid} ({best_hybrid_acc:.4f})\n\n")

        if best_hybrid_acc > feature_only_acc:
            f.write(f"**Result: GraGE-Hybrid DOES exceed Feature-only** (delta = +{best_hybrid_acc - feature_only_acc:.4f})\n\n")
        else:
            f.write(f"**Result: GraGE-Hybrid DOES NOT exceed Feature-only** (delta = {best_hybrid_acc - feature_only_acc:.4f})\n\n")

        # Per noise type analysis
        f.write("### 2. Improvement by noise type\n\n")
        for noise in df["noise_type"].unique():
            noise_df = df[df["noise_type"] == noise]
            fo_acc = noise_df[noise_df["method"] == "Feature-only"]["test_acc"].mean()
            best_hybrid_acc = noise_df[noise_df["method"] != "Feature-only"].groupby("method")["test_acc"].mean().max()
            delta = best_hybrid_acc - fo_acc
            f.write(f"- {noise}: Feature-only={fo_acc:.4f}, Best hybrid={best_hybrid_acc:.4f}, delta={delta:+.4f}\n")

        f.write("\n")

        # feature_similar_cross_class analysis
        f.write("### 3. feature_similar_cross_class analysis\n\n")
        fscc = df[df["noise_type"] == "feature_similar_cross_class"]
        if len(fscc) > 0:
            fo_fscc = fscc[fscc["method"] == "Feature-only"]["test_acc"].mean()
            best_hybrid_fscc = fscc[fscc["method"] != "Feature-only"].groupby("method")["test_acc"].mean().max()
            f.write(f"Feature-only: {fo_fscc:.4f}\n")
            f.write(f"Best hybrid: {best_hybrid_fscc:.4f}\n")
            if best_hybrid_fscc > fo_fscc:
                f.write(f"**GraGE-Hybrid exceeds Feature-only on feature_similar_cross_class**\n\n")
            else:
                f.write(f"**GraGE-Hybrid does NOT exceed Feature-only on feature_similar_cross_class**\n\n")

        # low_feature_similarity analysis
        f.write("### 4. low_feature_similarity analysis (sanity check)\n\n")
        lfs = df[df["noise_type"] == "low_feature_similarity"]
        if len(lfs) > 0:
            fo_lfs = lfs[lfs["method"] == "Feature-only"]["test_acc"].mean()
            best_hybrid_lfs = lfs[lfs["method"] != "Feature-only"].groupby("method")["test_acc"].mean().max()
            f.write(f"Feature-only: {fo_lfs:.4f}\n")
            f.write(f"Best hybrid: {best_hybrid_lfs:.4f}\n")
            f.write(f"Delta: {best_hybrid_lfs - fo_lfs:+.4f}\n\n")

        # Positive/negative gradient analysis
        f.write("### 5. Positive/negative gradient effectiveness\n\n")
        pos_methods = df[df["method"].str.contains("pos")]
        neg_methods = df[df["method"].str.contains("neg")]
        if len(pos_methods) > 0:
            f.write(f"Methods with positive gradient: mean acc = {pos_methods['test_acc'].mean():.4f}\n")
        if len(neg_methods) > 0:
            f.write(f"Methods with negative gradient: mean acc = {neg_methods['test_acc'].mean():.4f}\n\n")

        # Degree normalization analysis
        f.write("### 6. Degree normalization effect\n\n")
        degree_methods = df[df["degree_norm"] == True]
        non_degree_methods = df[df["degree_norm"] == False]
        if len(degree_methods) > 0 and len(non_degree_methods) > 0:
            f.write(f"With degree normalization: mean acc = {degree_methods['test_acc'].mean():.4f}\n")
            f.write(f"Without degree normalization: mean acc = {non_degree_methods['test_acc'].mean():.4f}\n\n")

        # Final conclusion
        f.write("### 7. Final Conclusion\n\n")
        if best_hybrid_acc > feature_only_acc:
            f.write("**GraGE-Hybrid provides gains over Feature-only.**\n")
            f.write("The training-dynamics calibration successfully improves upon static feature smoothness.\n")
        else:
            f.write("**Current training-dynamics calibration does not provide reliable gains beyond static feature smoothness.**\n")
            f.write("The edge-gate hypergradient approach needs further investigation.\n")

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Build GraGE-Hybrid tables")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for tables")
    args = parser.parse_args()

    # Load results
    df = load_results(args.results)
    print(f"Loaded {len(df)} results")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "paper_tables_grage_hybrid"

    os.makedirs(output_dir, exist_ok=True)

    # Build tables
    print("\nBuilding tables...")
    grouped = compute_main_practical_acc(df, output_dir)
    delta_summary, ttest_df = compute_hybrid_vs_feature_only(df, output_dir)
    detection = compute_edge_detection_f1(df, output_dir)
    breakdown = compute_noise_type_breakdown(df, output_dir)
    ablation = compute_ablation_pos_neg_degree(df, output_dir)
    clean_graph = compute_clean_graph_non_degradation(df, output_dir)

    # Generate decision report
    print("\nGenerating decision report...")
    generate_decision_report(df, output_dir, delta_summary, ttest_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
