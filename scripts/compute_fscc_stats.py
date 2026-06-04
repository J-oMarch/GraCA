#!/usr/bin/env python3
"""
Compute paired statistics for FSCC confirmation experiment.

Usage:
    python scripts/compute_fscc_stats.py \
        --primary_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/primary \
        --controls_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/controls \
        --heterophily_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily \
        --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/tables
"""
import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_csv_safe(path):
    """Load CSV, return None if not found."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info(f"Loaded {path}: {len(df)} rows")
        return df
    logger.warning(f"Missing: {path}")
    return None


def compute_paired_stats(df, group_cols=("dataset", "noise_type"), acc_col="test_acc"):
    """Compute paired statistics vs Feature-only for each (dataset, noise_type)."""
    results = []
    methods = [m for m in df["method"].unique() if m != "Feature-only"]

    for group_key, group_df in df.groupby(list(group_cols)):
        if isinstance(group_key, str):
            group_key = (group_key,)
        group_dict = dict(zip(group_cols, group_key))

        fo_df = group_df[group_df["method"] == "Feature-only"].set_index("seed")
        if len(fo_df) == 0:
            continue
        fo_accs = fo_df[acc_col]

        for method in methods:
            m_df = group_df[group_df["method"] == method].set_index("seed")
            if len(m_df) == 0:
                continue

            # Paired seeds
            common_seeds = sorted(set(fo_accs.index) & set(m_df[acc_col].index))
            if len(common_seeds) < 2:
                continue

            fo_paired = fo_accs.loc[common_seeds].values
            m_paired = m_df[acc_col].loc[common_seeds].values
            deltas = m_paired - fo_paired

            mean_delta = float(np.mean(deltas))
            std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
            mean_delta_pp = mean_delta * 100
            std_delta_pp = std_delta * 100

            # Paired t-test
            if len(deltas) >= 2:
                t_stat, t_pval = stats.ttest_rel(m_paired, fo_paired)
            else:
                t_stat, t_pval = 0.0, 1.0

            # Wilcoxon signed-rank
            wilcoxon_pval = np.nan
            if len(common_seeds) >= 6:
                try:
                    _, wilcoxon_pval = stats.wilcoxon(m_paired, fo_paired)
                except ValueError:
                    wilcoxon_pval = np.nan

            # Win rate
            win_rate = float(np.mean(m_paired > fo_paired))

            # Cohen's d (paired)
            cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0

            # Runtime
            mean_runtime = float(m_df["runtime"].loc[common_seeds].mean()) if "runtime" in m_df.columns else np.nan

            row = {
                **group_dict,
                "method": method,
                "n_seeds": len(common_seeds),
                "method_mean_acc": float(np.mean(m_paired)),
                "method_std_acc": float(np.std(m_paired, ddof=1)) if len(m_paired) > 1 else 0.0,
                "fo_mean_acc": float(np.mean(fo_paired)),
                "fo_std_acc": float(np.std(fo_paired, ddof=1)) if len(fo_paired) > 1 else 0.0,
                "delta_pp": mean_delta_pp,
                "delta_std_pp": std_delta_pp,
                "t_stat": float(t_stat),
                "t_pval": float(t_pval),
                "wilcoxon_pval": float(wilcoxon_pval) if not np.isnan(wilcoxon_pval) else np.nan,
                "win_rate": win_rate,
                "cohens_d": float(cohens_d),
                "mean_runtime": mean_runtime,
            }
            results.append(row)

    return pd.DataFrame(results)


def compute_summary_table(df, group_cols=("dataset", "noise_type"), acc_col="test_acc"):
    """Compute mean±std summary table per method across datasets."""
    summary = df.groupby(["method"] + list(group_cols))[acc_col].agg(
        ["mean", "std", "count"]
    ).reset_index()
    return summary


def print_decision(paired_df, output_dir):
    """Print decision based on PROJECT_STATE.md rules."""
    logger.info("\n" + "=" * 70)
    logger.info("DECISION SUMMARY")
    logger.info("=" * 70)

    # Focus on feature_similar_cross_class
    fscc = paired_df[paired_df.get("noise_type", paired_df.columns[0]) == "feature_similar_cross_class"] if "noise_type" in paired_df.columns else paired_df

    if len(fscc) == 0:
        logger.info("No feature_similar_cross_class results found.")
        return

    graage_methods = [m for m in fscc["method"].unique()
                      if "GraGE" in m or "MCGC" in m or "Hybrid" in m or "Selective" in m]

    for method in graage_methods:
        m_rows = fscc[fscc["method"] == method]
        if len(m_rows) == 0:
            continue

        mean_delta = m_rows["delta_pp"].mean()
        mean_pval = m_rows["t_pval"].mean()
        mean_win = m_rows["win_rate"].mean()
        mean_d = m_rows["cohens_d"].mean()
        sig_count = (m_rows["t_pval"] < 0.05).sum()
        total = len(m_rows)

        beat = mean_delta > 0 and mean_pval < 0.05 and mean_win > 0.5
        logger.info(f"\n{method}:")
        logger.info(f"  Mean delta: {mean_delta:+.2f} pp")
        logger.info(f"  Mean p-value: {mean_pval:.4f}")
        logger.info(f"  Mean win rate: {mean_win:.2f}")
        logger.info(f"  Mean Cohen's d: {mean_d:.3f}")
        logger.info(f"  Significant datasets: {sig_count}/{total}")
        logger.info(f"  Verdict: {'BEATS Feature-only' if beat else 'DOES NOT beat Feature-only'}")

    # Overall decision
    all_positive = all(
        fscc[fscc["method"] == m]["delta_pp"].mean() > 0
        for m in graage_methods
        if len(fscc[fscc["method"] == m]) > 0
    )
    any_sig = any(
        (fscc[fscc["method"] == m]["t_pval"] < 0.05).any()
        for m in graage_methods
        if len(fscc[fscc["method"] == m]) > 0
    )

    logger.info("\n" + "-" * 70)
    if all_positive and any_sig:
        verdict = "AAA direction viable"
    elif any_sig:
        verdict = "regime-specific"
    else:
        verdict = "needs stronger contribution or revise method"
    logger.info(f"OVERALL VERDICT: {verdict}")

    return verdict


def main():
    parser = argparse.ArgumentParser(description="Compute FSCC paired statistics")
    parser.add_argument("--primary_dir", type=str, required=True)
    parser.add_argument("--controls_dir", type=str, required=True)
    parser.add_argument("--heterophily_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    primary = load_csv_safe(os.path.join(args.primary_dir, "results.csv"))
    controls = load_csv_safe(os.path.join(args.controls_dir, "results.csv"))
    heterophily = load_csv_safe(os.path.join(args.heterophily_dir, "results.csv"))

    all_dfs = []
    if primary is not None:
        all_dfs.append(primary)
    if controls is not None:
        all_dfs.append(controls)
    if heterophily is not None:
        all_dfs.append(heterophily)

    if not all_dfs:
        logger.error("No result CSVs found.")
        return

    # Compute paired stats for each stage
    tables = {}
    verdicts = {}

    for label, df in [("primary", primary), ("controls", controls), ("heterophily", heterophily)]:
        if df is None or len(df) == 0:
            logger.warning(f"Skipping {label}: no data")
            continue

        paired = compute_paired_stats(df)
        summary = compute_summary_table(df)

        paired_path = os.path.join(args.output_dir, f"{label}_paired_stats.csv")
        summary_path = os.path.join(args.output_dir, f"{label}_summary.csv")
        paired.to_csv(paired_path, index=False)
        summary.to_csv(summary_path, index=False)
        tables[label] = paired_path

        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE: {label.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"\nPaired stats:\n{paired[['method','dataset','noise_type','delta_pp','t_pval','win_rate','cohens_d']].to_string()}")

        verdict = print_decision(paired, args.output_dir)
        verdicts[label] = verdict

    # Save overall paired stats
    if primary is not None:
        primary_paired = compute_paired_stats(primary)
        primary_paired.to_csv(os.path.join(args.output_dir, "primary_fscc.csv"), index=False)
        tables["primary_fscc"] = os.path.join(args.output_dir, "primary_fscc.csv")

    if controls is not None:
        controls_paired = compute_paired_stats(controls)
        controls_paired.to_csv(os.path.join(args.output_dir, "control_regimes.csv"), index=False)
        tables["control_regimes"] = os.path.join(args.output_dir, "control_regimes.csv")

    if heterophily is not None:
        hetero_paired = compute_paired_stats(heterophily)
        hetero_paired.to_csv(os.path.join(args.output_dir, "heterophily.csv"), index=False)
        tables["heterophily"] = os.path.join(args.output_dir, "heterophily.csv")

    logger.info(f"\nTables saved to {args.output_dir}")
    for k, v in tables.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
