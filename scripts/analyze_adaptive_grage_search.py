#!/usr/bin/env python3
"""
Analyze Adaptive GraGE Search Results.

Reads search/validation CSV outputs and generates:
- candidate_search_results.csv
- candidate_validation_results.csv
- candidate_vs_feature_only_stats.csv
- method_ablation_summary.csv
- result.md
- metrics.json
- failure_analysis.md

Usage:
    python scripts/analyze_adaptive_grage_search.py \
        --search_csv experiments/2026-06-04-adaptive-grage-search/logs/search/results.csv \
        --output_dir experiments/2026-06-04-adaptive-grage-search
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats


def load_and_check(csv_path):
    """Load CSV and check required columns."""
    df = pd.read_csv(csv_path)
    required = ["dataset", "noise_type", "seed", "method", "test_acc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return df


def compute_paired_delta(df, candidate, baseline, noise_type, dataset=None):
    """Compute paired delta between candidate and baseline on matched seeds.

    Returns dict with mean_delta, std_delta, n, p_value, significant.
    """
    if dataset:
        cand = df[(df["method"] == candidate) & (df["noise_type"] == noise_type) & (df["dataset"] == dataset)]
        base = df[(df["method"] == baseline) & (df["noise_type"] == noise_type) & (df["dataset"] == dataset)]
    else:
        cand = df[(df["method"] == candidate) & (df["noise_type"] == noise_type)]
        base = df[(df["method"] == baseline) & (df["noise_type"] == noise_type)]

    # Merge on seed (and dataset if not specified)
    if dataset:
        merge_keys = ["seed"]
    else:
        merge_keys = ["seed", "dataset"]

    merged = cand.merge(base, on=merge_keys, suffixes=("_cand", "_base"))
    if len(merged) == 0:
        return None

    deltas = merged["test_acc_cand"] - merged["test_acc_base"]
    n = len(deltas)
    mean_delta = deltas.mean()
    std_delta = deltas.std()

    # Paired t-test
    if n >= 2:
        t_stat, p_value = stats.ttest_rel(merged["test_acc_cand"], merged["test_acc_base"])
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "candidate": candidate,
        "baseline": baseline,
        "noise_type": noise_type,
        "dataset": dataset or "all",
        "mean_delta": float(mean_delta),
        "std_delta": float(std_delta),
        "n": int(n),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "candidate_mean": float(merged["test_acc_cand"].mean()),
        "baseline_mean": float(merged["test_acc_base"].mean()),
    }


def compute_win_rate(df, candidate, baseline, noise_type, dataset=None):
    """Compute fraction of seeds where candidate beats baseline."""
    if dataset:
        cand = df[(df["method"] == candidate) & (df["noise_type"] == noise_type) & (df["dataset"] == dataset)]
        base = df[(df["method"] == baseline) & (df["noise_type"] == noise_type) & (df["dataset"] == dataset)]
    else:
        cand = df[(df["method"] == candidate) & (df["noise_type"] == noise_type)]
        base = df[(df["method"] == baseline) & (df["noise_type"] == noise_type)]

    if dataset:
        merge_keys = ["seed"]
    else:
        merge_keys = ["seed", "dataset"]

    merged = cand.merge(base, on=merge_keys, suffixes=("_cand", "_base"))
    if len(merged) == 0:
        return 0.0

    wins = (merged["test_acc_cand"] > merged["test_acc_base"]).sum()
    return float(wins / len(merged))


def select_best_candidate(search_df):
    """Select best candidate based on search results.

    Primary: paired delta over Feature-only on feature_similar_cross_class
    """
    candidates = [m for m in search_df["method"].unique()
                  if m not in ("Feature-only", "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "Random-Matched")]

    best_candidate = None
    best_delta = -float("inf")
    best_stats = None

    for cand in candidates:
        delta_info = compute_paired_delta(
            search_df, cand, "Feature-only", "feature_similar_cross_class"
        )
        if delta_info is None:
            continue

        delta = delta_info["mean_delta"]
        if delta > best_delta:
            best_delta = delta
            best_candidate = cand
            best_stats = delta_info

    return best_candidate, best_stats


def generate_search_results_table(search_df, output_dir):
    """Generate candidate_search_results.csv."""
    methods = search_df["method"].unique()
    noise_types = search_df["noise_type"].unique()

    rows = []
    for method in methods:
        for noise in noise_types:
            subset = search_df[(search_df["method"] == method) & (search_df["noise_type"] == noise)]
            if len(subset) == 0:
                continue
            rows.append({
                "method": method,
                "noise_type": noise,
                "test_acc_mean": subset["test_acc"].mean(),
                "test_acc_std": subset["test_acc"].std(),
                "n": len(subset),
                "bad_edge_f1_mean": subset["bad_edge_f1"].mean(),
                "runtime_mean": subset["runtime"].mean(),
            })

    result_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "candidate_search_results.csv")
    result_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return result_df


def generate_vs_feature_only_stats(search_df, best_candidate, output_dir):
    """Generate candidate_vs_feature_only_stats.csv."""
    noise_types = search_df["noise_type"].unique()
    datasets = search_df["dataset"].unique()

    rows = []
    for noise in noise_types:
        # Overall
        delta = compute_paired_delta(search_df, best_candidate, "Feature-only", noise)
        wr = compute_win_rate(search_df, best_candidate, "Feature-only", noise)
        if delta:
            rows.append({
                "noise_type": noise,
                "dataset": "all",
                "delta_mean": delta["mean_delta"],
                "delta_std": delta["std_delta"],
                "p_value": delta["p_value"],
                "significant": delta["significant"],
                "win_rate": wr,
                "n": delta["n"],
            })

        # Per dataset
        for ds in datasets:
            delta = compute_paired_delta(search_df, best_candidate, "Feature-only", noise, ds)
            wr = compute_win_rate(search_df, best_candidate, "Feature-only", noise, ds)
            if delta:
                rows.append({
                    "noise_type": noise,
                    "dataset": ds,
                    "delta_mean": delta["mean_delta"],
                    "delta_std": delta["std_delta"],
                    "p_value": delta["p_value"],
                    "significant": delta["significant"],
                    "win_rate": wr,
                    "n": delta["n"],
                })

    result_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "candidate_vs_feature_only_stats.csv")
    result_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return result_df


def generate_method_ablation(search_df, output_dir):
    """Generate method_ablation_summary.csv."""
    methods = search_df["method"].unique()
    rows = []

    for method in methods:
        subset = search_df[search_df["method"] == method]
        if len(subset) == 0:
            continue

        method_type = subset["method_type"].iloc[0] if "method_type" in subset.columns else "unknown"

        rows.append({
            "method": method,
            "method_type": method_type,
            "test_acc_mean": subset["test_acc"].mean(),
            "test_acc_std": subset["test_acc"].std(),
            "test_acc_min": subset["test_acc"].min(),
            "test_acc_max": subset["test_acc"].max(),
            "n": len(subset),
            "bad_edge_f1_mean": subset["bad_edge_f1"].mean(),
            "runtime_mean": subset["runtime"].mean(),
            # Per noise type means
            **{f"acc_{nt}": subset[subset["noise_type"] == nt]["test_acc"].mean()
               for nt in search_df["noise_type"].unique()},
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("test_acc_mean", ascending=False)
    path = os.path.join(output_dir, "method_ablation_summary.csv")
    result_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return result_df


def generate_result_md(search_df, best_candidate, best_stats, output_dir):
    """Generate result.md decision report."""
    # Compute all deltas for best candidate
    noise_types = search_df["noise_type"].unique()
    deltas = {}
    for noise in noise_types:
        d = compute_paired_delta(search_df, best_candidate, "Feature-only", noise)
        if d:
            deltas[noise] = d

    # Hybrid delta
    hybrid_deltas = {}
    for noise in noise_types:
        d = compute_paired_delta(search_df, best_candidate, "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", noise)
        if d:
            hybrid_deltas[noise] = d

    # Best candidate info
    cand_subset = search_df[search_df["method"] == best_candidate]
    cand_type = cand_subset["method_type"].iloc[0] if "method_type" in search_df.columns else "unknown"

    # Summary table
    summary_lines = []
    summary_lines.append("| Method | Noise Type | Delta vs Feature-only | p-value | Significant |")
    summary_lines.append("|--------|-----------|----------------------|---------|-------------|")
    for noise, d in deltas.items():
        sig = "Yes" if d["significant"] else "No"
        summary_lines.append(
            f"| {best_candidate} | {noise} | {d['mean_delta']:+.4f} ± {d['std_delta']:.4f} | "
            f"{d['p_value']:.4f} | {sig} |"
        )

    # Overall summary
    overall_mean = cand_subset.groupby("seed")["test_acc"].mean().mean()
    fo_mean = search_df[search_df["method"] == "Feature-only"].groupby("seed")["test_acc"].mean().mean()
    hybrid_mean = search_df[search_df["method"] == "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5"].groupby("seed")["test_acc"].mean().mean()

    fsc_delta = deltas.get("feature_similar_cross_class", {})
    fsc_hybrid_delta = hybrid_deltas.get("feature_similar_cross_class", {})

    md = f"""# Adaptive GraGE Search — Decision Report

## Summary

**Best candidate**: `{best_candidate}`
**Method type**: `{cand_type}`
**Candidate family**: {"Feature-Ambiguity-Adaptive Hybrid" if "faa" in best_candidate.lower() else "Multi-Checkpoint Gradient Consistency" if "mcgc" in best_candidate.lower() else cand_type}

## Overall Performance

| Method | Mean Test Acc |
|--------|--------------|
| Feature-only | {fo_mean:.4f} |
| GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 | {hybrid_mean:.4f} |
| {best_candidate} | {overall_mean:.4f} |

## Paired Deltas vs Feature-only

{chr(10).join(summary_lines)}

## Key Result: feature_similar_cross_class

- Feature-only mean: {fsc_delta.get('baseline_mean', 0):.4f}
- {best_candidate} mean: {fsc_delta.get('candidate_mean', 0):.4f}
- Delta: {fsc_delta.get('mean_delta', 0):+.4f} ± {fsc_delta.get('std_delta', 0):.4f}
- p-value: {fsc_delta.get('p_value', 1):.4f}
- Significant: {"Yes" if fsc_delta.get('significant', False) else "No"}

## vs Current Best Hybrid (feature_similar_cross_class)

- Hybrid mean: {fsc_hybrid_delta.get('baseline_mean', 0):.4f}
- {best_candidate} mean: {fsc_hybrid_delta.get('candidate_mean', 0):.4f}
- Delta: {fsc_hybrid_delta.get('mean_delta', 0):+.4f}

## Algorithmic Contribution

"""

    if "faa" in best_candidate.lower():
        md += """The FAA-Hybrid method adapts the gradient weighting based on feature ambiguity.
When features are similar between endpoints (high cosine similarity), static
feature risk is less informative and the method amplifies the training-dynamics
gradient signal. When features clearly differ, the method trusts static feature
risk.

**Key insight**: Not all edges should be scored the same way. Feature-ambiguous
edges require stronger reliance on training dynamics.
"""
    elif "mcgc" in best_candidate.lower():
        md += """The MCGC method uses gradient sign consistency across multiple training
checkpoints as a confidence signal. Edges whose harmful gradient is consistent
across training stages are more reliably harmful than edges with unstable
gradients.

**Key insight**: A single training snapshot may give noisy signals. Consistency
across checkpoints indicates reliable edge-level information.
"""
    else:
        md += f"Method type: {cand_type}. See code for details.\n"

    md += """
## Candidate Selected for Confirmation

"""

    if fsc_delta.get("mean_delta", 0) > 0:
        md += "**Yes** — the candidate shows positive delta on feature_similar_cross_class.\n"
        md += "A larger confirmation experiment is recommended.\n\n"
        md += "## Next Experiment\n\n"
        md += "Run a confirmation experiment with:\n"
        md += "- 5 seeds (0..4)\n"
        md += "- 3 datasets (Cora, CiteSeer, PubMed)\n"
        md += "- 3 noise types (feature_similar_cross_class, low_feature_similarity, degree_aligned_random)\n"
    else:
        md += "**No** — the candidate does not beat Feature-only on feature_similar_cross_class.\n"
        md += "Consider method redesign or alternative paper framing.\n"

    path = os.path.join(output_dir, "result.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"Saved: {path}")
    return md


def generate_metrics_json(search_df, best_candidate, best_stats, output_dir):
    """Generate metrics.json."""
    noise_types = search_df["noise_type"].unique()

    # Delta vs Feature-only on feature_similar_cross_class
    fsc_delta = compute_paired_delta(search_df, best_candidate, "Feature-only", "feature_similar_cross_class")
    fsc_delta_val = fsc_delta["mean_delta"] if fsc_delta else 0.0

    # Delta vs current hybrid on feature_similar_cross_class
    hybrid_delta = compute_paired_delta(search_df, best_candidate, "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "feature_similar_cross_class")
    hybrid_delta_val = hybrid_delta["mean_delta"] if hybrid_delta else 0.0

    # Win rate
    wr = compute_win_rate(search_df, best_candidate, "Feature-only", "feature_similar_cross_class")

    # Effect size (Cohen's d)
    cand_acc = search_df[search_df["method"] == best_candidate]["test_acc"]
    fo_acc = search_df[search_df["method"] == "Feature-only"]["test_acc"]
    pooled_std = np.sqrt((cand_acc.std()**2 + fo_acc.std()**2) / 2)
    cohens_d = (cand_acc.mean() - fo_acc.mean()) / max(pooled_std, 1e-8)

    # Determine candidate family
    if "faa" in best_candidate.lower():
        family = "Feature-Ambiguity-Adaptive Hybrid"
    elif "mcgc" in best_candidate.lower():
        family = "Multi-Checkpoint Gradient Consistency"
    else:
        family = "unknown"

    # Count candidate methods
    all_methods = search_df["method"].unique()
    num_candidates = len([m for m in all_methods
                          if m not in ("Feature-only", "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "Random-Matched")])

    # New files
    new_files = [
        "src/grage/adaptive_score.py",
        "tests/test_adaptive_score.py",
        "scripts/run_adaptive_grage_search.py",
        "scripts/analyze_adaptive_grage_search.py",
    ]

    # Determine status
    if fsc_delta_val > 0:
        status = "completed"
        claim_rec = f"{best_candidate} shows +{fsc_delta_val:.4f} delta over Feature-only on feature_similar_cross_class. Recommend confirmation experiment."
        selected = True
    else:
        status = "completed"
        claim_rec = f"No candidate beats Feature-only on feature_similar_cross_class. Best delta: {fsc_delta_val:+.4f}. Consider method redesign."
        selected = False

    metrics = {
        "exp_id": "2026-06-04-adaptive-grage-search",
        "status": status,
        "best_candidate": best_candidate,
        "candidate_family": family,
        "candidate_selected_for_confirmation": selected,
        "delta_vs_feature_only_feature_similar_cross_class": round(fsc_delta_val, 6),
        "delta_vs_current_hybrid_feature_similar_cross_class": round(hybrid_delta_val, 6),
        "low_feature_similarity_degradation_vs_feature_only": 0.0,  # Filled during validation
        "win_rate_vs_feature_only": round(wr, 4),
        "effect_size_vs_feature_only": round(float(cohens_d), 4),
        "failure_modes": [],
        "num_candidate_methods": num_candidates,
        "num_result_rows": len(search_df),
        "new_files_or_modules": new_files,
        "claim_recommendation": claim_rec,
    }

    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {path}")
    return metrics


def generate_failure_analysis(search_df, best_candidate, output_dir):
    """Generate failure_analysis.md."""
    all_methods = search_df["method"].unique()
    candidates = [m for m in all_methods
                  if m not in ("Feature-only", "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "Random-Matched")]

    fo_fsc = search_df[(search_df["method"] == "Feature-only") & (search_df["noise_type"] == "feature_similar_cross_class")]["test_acc"].mean()

    lines = ["# Failure Analysis\n"]
    lines.append("## Candidate Methods Evaluated\n")
    lines.append(f"Total candidates: {len(candidates)}\n")
    lines.append(f"Best candidate: {best_candidate}\n")
    lines.append(f"Feature-only baseline on feature_similar_cross_class: {fo_fsc:.4f}\n\n")

    lines.append("## Per-Candidate Analysis\n\n")

    for cand in sorted(candidates):
        cand_fsc = search_df[(search_df["method"] == cand) & (search_df["noise_type"] == "feature_similar_cross_class")]["test_acc"].mean()
        delta = cand_fsc - fo_fsc
        status = "✓ BEATS" if delta > 0 else "✗ LOSES"

        lines.append(f"### {cand}\n")
        lines.append(f"- **Status**: {status} Feature-only on feature_similar_cross_class\n")
        lines.append(f"- **Delta**: {delta:+.4f}\n")

        # Determine failure mode
        if delta <= 0:
            # Check if it's signal quality, over-regularization, etc.
            cand_all = search_df[search_df["method"] == cand]["test_acc"].mean()
            fo_all = search_df[search_df["method"] == "Feature-only"]["test_acc"].mean()
            overall_delta = cand_all - fo_all

            if overall_delta < -0.01:
                lines.append(f"- **Failure mode**: Overall degradation ({overall_delta:+.4f}). "
                             "Method adds noise rather than signal.\n")
            elif cand_fsc < fo_fsc - 0.005:
                lines.append("- **Failure mode**: Signal quality. Gradient signal not informative "
                             "enough on feature_similar_cross_class to overcome feature prior.\n")
            else:
                lines.append("- **Failure mode**: Marginal underperformance. Close to Feature-only "
                             "but not enough to justify complexity.\n")
        else:
            lines.append(f"- **Result**: Positive delta. Candidate is viable.\n")

        lines.append("\n")

    lines.append("## Common Failure Patterns\n\n")
    lines.append("1. **Signal quality**: The gradient signal from a single training snapshot "
                 "may be too noisy to reliably identify harmful edges beyond what feature "
                 "similarity already captures.\n")
    lines.append("2. **Support/score split instability**: The train-split into support/score "
                 "masks reduces the effective training data, making gradient estimates less "
                 "reliable.\n")
    lines.append("3. **Budget matching**: Pruning 20% of edges may not align with the actual "
                 "fraction of harmful edges, leading to over- or under-pruning.\n")
    lines.append("4. **Degree preservation conflict**: min_degree constraints prevent removing "
                 "edges to low-degree nodes even when they are harmful.\n")

    path = os.path.join(output_dir, "failure_analysis.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Adaptive GraGE Search Results")
    parser.add_argument("--search_csv", type=str, required=True,
                        help="Path to search results CSV")
    parser.add_argument("--validation_csv", type=str, default=None,
                        help="Path to validation results CSV (optional)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load search results
    print(f"\nLoading search results from: {args.search_csv}")
    search_df = load_and_check(args.search_csv)
    print(f"  {len(search_df)} rows, {search_df['method'].nunique()} methods, "
          f"{search_df['noise_type'].nunique()} noise types")

    # Generate search results table
    print("\n--- Generating search results table ---")
    generate_search_results_table(search_df, args.output_dir)

    # Select best candidate
    print("\n--- Selecting best candidate ---")
    best_candidate, best_stats = select_best_candidate(search_df)
    print(f"  Best candidate: {best_candidate}")
    if best_stats:
        print(f"  Delta vs Feature-only on feature_similar_cross_class: {best_stats['mean_delta']:+.4f}")

    # Generate vs Feature-only stats
    print("\n--- Generating vs Feature-only stats ---")
    generate_vs_feature_only_stats(search_df, best_candidate, args.output_dir)

    # Generate method ablation
    print("\n--- Generating method ablation ---")
    generate_method_ablation(search_df, args.output_dir)

    # Generate result.md
    print("\n--- Generating result.md ---")
    generate_result_md(search_df, best_candidate, best_stats, args.output_dir)

    # Generate metrics.json
    print("\n--- Generating metrics.json ---")
    generate_metrics_json(search_df, best_candidate, best_stats, args.output_dir)

    # Generate failure_analysis.md
    print("\n--- Generating failure_analysis.md ---")
    generate_failure_analysis(search_df, best_candidate, args.output_dir)

    # Process validation results if provided
    if args.validation_csv and os.path.exists(args.validation_csv):
        print(f"\n--- Loading validation results from: {args.validation_csv} ---")
        val_df = load_and_check(args.validation_csv)
        print(f"  {len(val_df)} rows")

        # Generate validation table
        val_table = generate_search_results_table(val_df, args.output_dir)
        val_path = os.path.join(args.output_dir, "candidate_validation_results.csv")
        val_table.to_csv(val_path, index=False)
        print(f"  Saved: {val_path}")

        # Update metrics with low_feature_similarity degradation
        lfs_fo = val_df[(val_df["method"] == "Feature-only") & (val_df["noise_type"] == "low_feature_similarity")]["test_acc"].mean()
        lfs_cand = val_df[(val_df["method"] == best_candidate) & (val_df["noise_type"] == "low_feature_similarity")]["test_acc"].mean()
        lfs_degradation = lfs_cand - lfs_fo

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["low_feature_similarity_degradation_vs_feature_only"] = round(lfs_degradation, 6)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Updated metrics.json with LFS degradation: {lfs_degradation:+.4f}")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
