"""
Generate paper tables from real CSV results.
Outputs to paper_tables/ directory.

Usage:
    python scripts/update_readme_tables.py
"""
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results():
    """Load all result CSVs."""
    results = {}
    for name, path in [
        ("main", "results/main/results.csv"),
        ("baselines", "results/baselines/baseline_results.csv"),
        ("oracle", "results/oracle/oracle_results.csv"),
        ("ablation", "results/ablation/ablation_results.csv"),
        ("noisy", "results/noisy_edges/noisy_edge_results.csv"),
    ]:
        if os.path.exists(path):
            results[name] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found")
    return results


def paired_t_test(df, method_a, method_b, dataset, model):
    """Perform paired t-test between two methods on same dataset/model."""
    a = df[(df["method"] == method_a) & (df["dataset"] == dataset) &
           (df["downstream_model"] == model)]["test_acc"].values
    b = df[(df["method"] == method_b) & (df["dataset"] == dataset) &
           (df["downstream_model"] == model)]["test_acc"].values
    n = min(len(a), len(b))
    if n < 2:
        return None
    t, p = stats.ttest_rel(a[:n], b[:n])
    return {"t_stat": t, "p_value": p, "significant": p < 0.05}


def generate_main_homophily(results):
    """Generate main homophilic dataset table."""
    if "main" not in results or "baselines" not in results:
        print("Missing main or baseline results")
        return

    main_df = results["main"]
    baseline_df = results["baselines"]
    all_df = pd.concat([main_df, baseline_df], ignore_index=True)

    homo_datasets = ["Cora", "CiteSeer", "PubMed"]
    methods = ["Original", "DropEdge", "Random Pruning", "Homophily Pruning", "GraCA-lite"]

    rows = []
    for ds in homo_datasets:
        for model in ["GCN", "GAT", "GraphSAGE"]:
            row = {"dataset": ds, "model": model}
            for method in methods:
                s = all_df[(all_df["dataset"] == ds) & (all_df["method"] == method) &
                          (all_df["downstream_model"] == model)]["test_acc"]
                if len(s) > 0:
                    row[method] = f"{s.mean()*100:.2f} ± {s.std()*100:.2f}"
                else:
                    row[method] = "-"
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("paper_tables", exist_ok=True)
    df.to_csv("paper_tables/main_homophily.csv", index=False)
    print("Generated paper_tables/main_homophily.csv")
    return df


def generate_main_heterophily(results):
    """Generate main heterophilic dataset table."""
    if "main" not in results or "baselines" not in results:
        return

    main_df = results["main"]
    baseline_df = results["baselines"]
    all_df = pd.concat([main_df, baseline_df], ignore_index=True)

    hetero_datasets = ["Actor", "Texas", "Cornell", "Wisconsin"]
    methods = ["Original", "GraCA-lite"]

    rows = []
    for ds in hetero_datasets:
        for model in ["GCN", "GAT", "GraphSAGE"]:
            row = {"dataset": ds, "model": model}
            for method in methods:
                s = all_df[(all_df["dataset"] == ds) & (all_df["method"] == method) &
                          (all_df["downstream_model"] == model)]["test_acc"]
                if len(s) > 0:
                    row[method] = f"{s.mean()*100:.2f} ± {s.std()*100:.2f}"
                else:
                    row[method] = "-"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("paper_tables/main_heterophily.csv", index=False)
    print("Generated paper_tables/main_heterophily.csv")
    return df


def generate_noisy_edge_table(results):
    """Generate noisy-edge robustness table."""
    if "noisy" not in results:
        return

    noisy_df = results["noisy"]

    rows = []
    for ds in noisy_df["dataset"].unique():
        for noise in sorted(noisy_df["actual_prune_ratio"].unique()):
            for method in noisy_df["method"].unique():
                s = noisy_df[(noisy_df["dataset"] == ds) & (noisy_df["method"] == method)]
                if len(s) > 0:
                    rows.append({
                        "dataset": ds,
                        "method": method,
                        "test_acc": f"{s['test_acc'].mean()*100:.2f} ± {s['test_acc'].std()*100:.2f}",
                        "bad_edge_f1": f"{s['bad_edge_f1'].mean():.3f}" if "bad_edge_f1" in s.columns else "-",
                    })

    df = pd.DataFrame(rows)
    df.to_csv("paper_tables/noisy_edge_robustness.csv", index=False)
    print("Generated paper_tables/noisy_edge_robustness.csv")
    return df


def generate_ablation_table(results):
    """Generate ablation table."""
    if "ablation" not in results:
        return

    abl_df = results["ablation"]

    rows = []
    for ds in abl_df["dataset"].unique():
        for method in abl_df["method"].unique():
            for model in abl_df["downstream_model"].unique():
                s = abl_df[(abl_df["dataset"] == ds) & (abl_df["method"] == method) &
                          (abl_df["downstream_model"] == model)]["test_acc"]
                if len(s) > 0:
                    rows.append({
                        "dataset": ds,
                        "method": method,
                        "model": model,
                        "test_acc": f"{s.mean()*100:.2f} ± {s.std()*100:.2f}",
                    })

    df = pd.DataFrame(rows)
    df.to_csv("paper_tables/ablation.csv", index=False)
    print("Generated paper_tables/ablation.csv")
    return df


def generate_oracle_gap_table(results):
    """Generate oracle gap analysis table."""
    if "oracle" not in results or "main" not in results:
        return

    oracle_df = results["oracle"]
    main_df = results["main"]

    rows = []
    for ds in ["Cora", "CiteSeer", "PubMed"]:
        for model in ["GCN", "GAT", "GraphSAGE"]:
            oracle = oracle_df[(oracle_df["dataset"] == ds) &
                              (oracle_df["downstream_model"] == model)]["test_acc"]
            graca = main_df[(main_df["dataset"] == ds) & (main_df["method"] == "GraCA-lite") &
                           (main_df["downstream_model"] == model)]["test_acc"]
            if len(oracle) > 0 and len(graca) > 0:
                gap = oracle.mean() - graca.mean()
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "oracle_acc": f"{oracle.mean()*100:.2f}",
                    "graca_acc": f"{graca.mean()*100:.2f}",
                    "gap": f"{gap*100:.2f}",
                })

    df = pd.DataFrame(rows)
    df.to_csv("paper_tables/oracle_gap.csv", index=False)
    print("Generated paper_tables/oracle_gap.csv")
    return df


def main():
    print("=" * 60)
    print("Generating Paper Tables")
    print("=" * 60)

    results = load_results()

    generate_main_homophily(results)
    generate_main_heterophily(results)
    generate_noisy_edge_table(results)
    generate_ablation_table(results)
    generate_oracle_gap_table(results)

    print("\n✓ All tables generated in paper_tables/")


if __name__ == "__main__":
    main()
