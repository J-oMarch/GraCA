"""
Build final paper tables from results_clean/.

Generates:
1. paper_tables_clean/table1_clean_accuracy.csv
2. paper_tables_clean/table2_noisy_accuracy.csv
3. paper_tables_clean/table3_bad_edge_detection.csv
4. paper_tables_clean/table4_ablation_noisy.csv
5. paper_tables_clean/table5_oracle_gap.csv
6. paper_tables_clean/table6_scalability.csv
7. paper_tables_clean/statistical_tests.csv

All tables include: mean, std, n_seeds, p_value_vs_original, p_value_vs_best_baseline

Usage:
    python scripts/build_final_tables.py --results_dir results_clean/ --output_dir paper_tables_clean/
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_all_results(results_dir):
    """Load all CSVs from results_dir, validating schema consistency."""
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
            raise ValueError(f"Error reading {f}: {e}")

    all_df = pd.concat(dfs, ignore_index=True)

    # Validate required columns exist
    required = ["dataset", "method", "downstream_model", "seed", "test_acc",
                "actual_prune_ratio", "experiment_type"]
    for col in required:
        if col not in all_df.columns:
            raise ValueError(f"Missing required column: {col}")

    return all_df


def paired_t_test(df, method_a, method_b, metric="test_acc"):
    """Paired t-test between two methods across same (dataset, seed) pairs."""
    a = df[df["method"] == method_a].set_index(["dataset", "seed"])[metric]
    b = df[df["method"] == method_b].set_index(["dataset", "seed"])[metric]

    # Find common (dataset, seed) pairs
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return np.nan

    a_vals = a.loc[common].values
    b_vals = b.loc[common].values

    _, p_value = stats.ttest_rel(a_vals, b_vals)
    return p_value


def build_table1_clean(df, output_dir):
    """Table 1: Clean graph accuracy."""
    clean = df[df["experiment_type"] == "clean"].copy()
    if clean.empty:
        print("WARNING: No clean experiment results found")
        return

    # Exclude oracle from practical table
    practical = clean[clean["oracle_only"] != True]

    # Group by dataset, method, downstream_model
    grouped = practical.groupby(["dataset", "method", "downstream_model"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        n_seeds=("seed", "nunique"),
        actual_prune_ratio_mean=("actual_prune_ratio", "mean"),
        runtime_mean=("runtime", "mean"),
    ).reset_index()

    # Add p-values vs Original
    grouped["p_value_vs_original"] = grouped.apply(
        lambda r: paired_t_test(
            practical[practical["dataset"] == r["dataset"]],
            r["method"], "Original", "test_acc"
        ) if r["method"] != "Original" else np.nan, axis=1
    )

    # Find best practical baseline per (dataset, downstream_model) and add p-value
    best_baselines = {}
    for (ds, model), grp in practical.groupby(["dataset", "downstream_model"]):
        non_graca = grp[~grp["method"].str.contains("GraCA|Oracle")]
        if not non_graca.empty:
            best = non_graca.groupby("method")["test_acc"].mean().idxmax()
            best_baselines[(ds, model)] = best

    grouped["p_value_vs_best_baseline"] = grouped.apply(
        lambda r: paired_t_test(
            practical[(practical["dataset"] == r["dataset"]) &
                      (practical["downstream_model"] == r["downstream_model"])],
            r["method"], best_baselines.get((r["dataset"], r["downstream_model"]), "Original"),
            "test_acc"
        ) if r["method"] not in ("Original", best_baselines.get((r["dataset"], r["downstream_model"]))) else np.nan,
        axis=1
    )

    # Format
    grouped["test_acc_formatted"] = grouped.apply(
        lambda r: f"{r['test_acc_mean']*100:.2f} ± {r['test_acc_std']*100:.2f}", axis=1
    )

    path = os.path.join(output_dir, "table1_clean_accuracy.csv")
    grouped.to_csv(path, index=False)
    print(f"Table 1 saved to {path} ({len(grouped)} rows)")


def build_table2_noisy(df, output_dir):
    """Table 2: Noisy graph accuracy."""
    noisy = df[df["experiment_type"] == "noisy_edge"].copy()
    if noisy.empty:
        print("WARNING: No noisy-edge experiment results found")
        return

    practical = noisy[noisy["oracle_only"] != True]

    grouped = practical.groupby(
        ["dataset", "noise_type", "noise_ratio", "method", "downstream_model"]
    ).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        n_seeds=("seed", "nunique"),
        actual_prune_ratio_mean=("actual_prune_ratio", "mean"),
    ).reset_index()

    # p-values vs Original+Noise
    grouped["p_value_vs_original"] = grouped.apply(
        lambda r: paired_t_test(
            practical[(practical["dataset"] == r["dataset"]) &
                      (practical["noise_type"] == r["noise_type"]) &
                      (practical["noise_ratio"] == r["noise_ratio"])],
            r["method"], "Original+Noise", "test_acc"
        ) if r["method"] != "Original+Noise" else np.nan, axis=1
    )

    grouped["test_acc_formatted"] = grouped.apply(
        lambda r: f"{r['test_acc_mean']*100:.2f} ± {r['test_acc_std']*100:.2f}", axis=1
    )

    path = os.path.join(output_dir, "table2_noisy_accuracy.csv")
    grouped.to_csv(path, index=False)
    print(f"Table 2 saved to {path} ({len(grouped)} rows)")


def build_table3_bad_edge(df, output_dir):
    """Table 3: Bad-edge detection metrics."""
    noisy = df[df["experiment_type"] == "noisy_edge"].copy()
    if noisy.empty:
        print("WARNING: No noisy-edge experiment results found")
        return

    # Exclude Original+Noise (no pruning = no detection)
    detection = noisy[~noisy["method"].isin(["Original+Noise"])].copy()

    grouped = detection.groupby(
        ["dataset", "noise_type", "noise_ratio", "method"]
    ).agg(
        precision_mean=("bad_edge_precision", "mean"),
        precision_std=("bad_edge_precision", "std"),
        recall_mean=("bad_edge_recall", "mean"),
        recall_std=("bad_edge_recall", "std"),
        f1_mean=("bad_edge_f1", "mean"),
        f1_std=("bad_edge_f1", "std"),
        clean_removed_mean=("clean_edge_mistakenly_removed_ratio", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    grouped["f1_formatted"] = grouped.apply(
        lambda r: f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}", axis=1
    )

    path = os.path.join(output_dir, "table3_bad_edge_detection.csv")
    grouped.to_csv(path, index=False)
    print(f"Table 3 saved to {path} ({len(grouped)} rows)")


def build_table4_ablation(df, output_dir):
    """Table 4: Ablation on noisy-edge."""
    ablation = df[df["experiment_type"] == "ablation"].copy()
    if ablation.empty:
        print("WARNING: No ablation results found")
        return

    grouped = ablation.groupby(
        ["dataset", "noise_type", "noise_ratio", "method", "downstream_model"]
    ).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        f1_mean=("bad_edge_f1", "mean"),
        f1_std=("bad_edge_f1", "std"),
        prune_ratio_mean=("actual_prune_ratio", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    path = os.path.join(output_dir, "table4_ablation_noisy.csv")
    grouped.to_csv(path, index=False)
    print(f"Table 4 saved to {path} ({len(grouped)} rows)")


def build_table5_oracle(df, output_dir):
    """Table 5: Oracle gap analysis."""
    oracle = df[df["experiment_type"] == "oracle"].copy()
    practical = df[(df["experiment_type"] == "clean") & (df["oracle_only"] != True)].copy()

    if oracle.empty or practical.empty:
        print("WARNING: Insufficient oracle/practical results for Table 5")
        return

    # Get GraCA-lite and GraCA-Oracle means per (dataset, downstream_model)
    graca_lite = practical[practical["method"] == "GraCA-lite"].groupby(
        ["dataset", "downstream_model"]
    )["test_acc"].mean()

    graca_oracle = oracle[oracle["method"] == "GraCA-Oracle"].groupby(
        ["dataset", "downstream_model"]
    )["test_acc"].mean()

    rows = []
    for key in graca_lite.index:
        if key in graca_oracle.index:
            ds, model = key
            rows.append({
                "dataset": ds,
                "downstream_model": model,
                "graca_lite_acc": graca_lite[key],
                "graca_oracle_acc": graca_oracle[key],
                "oracle_gap": graca_oracle[key] - graca_lite[key],
            })

    if rows:
        result_df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "table5_oracle_gap.csv")
        result_df.to_csv(path, index=False)
        print(f"Table 5 saved to {path} ({len(result_df)} rows)")
    else:
        print("WARNING: No matching oracle/practical pairs found")


def build_table6_scalability(df, output_dir):
    """Table 6: Scalability."""
    scalability = df[df["experiment_type"] == "scalability"].copy()
    if scalability.empty:
        # Try clean results as fallback
        scalability = df[df["experiment_type"] == "clean"].copy()

    if scalability.empty:
        print("WARNING: No scalability results found")
        return

    grouped = scalability.groupby(["dataset", "method"]).agg(
        runtime_mean=("runtime", "mean"),
        runtime_std=("runtime", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    path = os.path.join(output_dir, "table6_scalability.csv")
    grouped.to_csv(path, index=False)
    print(f"Table 6 saved to {path} ({len(grouped)} rows)")


def build_statistical_tests(df, output_dir):
    """Statistical tests: paired t-tests for all method pairs."""
    practical = df[(df["experiment_type"].isin(["clean", "noisy_edge"])) &
                   (df["oracle_only"] != True)].copy()

    rows = []
    for (exp_type, ds, model), grp in practical.groupby(["experiment_type", "dataset", "downstream_model"]):
        methods = grp["method"].unique()
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                p = paired_t_test(grp, m1, m2, "test_acc")
                if not np.isnan(p):
                    rows.append({
                        "experiment_type": exp_type,
                        "dataset": ds,
                        "downstream_model": model,
                        "method_a": m1,
                        "method_b": m2,
                        "p_value": p,
                        "significant_005": p < 0.05,
                        "significant_001": p < 0.01,
                    })

    if rows:
        result_df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "statistical_tests.csv")
        result_df.to_csv(path, index=False)
        print(f"Statistical tests saved to {path} ({len(result_df)} rows)")
    else:
        print("WARNING: No statistical tests could be computed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_clean/")
    parser.add_argument("--output_dir", type=str, default="paper_tables_clean/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}...")
    df = load_all_results(args.results_dir)
    print(f"Loaded {len(df)} rows from {df['_source'].nunique()} CSV files")
    print(f"Experiment types: {df['experiment_type'].value_counts().to_dict()}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print()

    build_table1_clean(df, args.output_dir)
    build_table2_noisy(df, args.output_dir)
    build_table3_bad_edge(df, args.output_dir)
    build_table4_ablation(df, args.output_dir)
    build_table5_oracle(df, args.output_dir)
    build_table6_scalability(df, args.output_dir)
    build_statistical_tests(df, args.output_dir)

    print(f"\nAll tables saved to {args.output_dir}")


if __name__ == "__main__":
    main()
