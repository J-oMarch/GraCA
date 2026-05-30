import pandas as pd
from pathlib import Path


def aggregate_results(
    input_dir: str,
    output_path: str,
    exclude_oracle: bool = True,
):
    """Aggregate results from CSV files in input_dir.

    Groups by (dataset, method, downstream_model) and computes mean ± std.
    """
    csv_files = list(Path(input_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No valid results found")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    # Exclude oracle from main results
    if exclude_oracle and "oracle_only" in all_df.columns:
        all_df = all_df[all_df["oracle_only"] != True]
        all_df = all_df[all_df["oracle_only"] != "True"]

    # Group and aggregate
    group_cols = ["dataset", "method", "downstream_model"]
    if "proxy_model" in all_df.columns:
        group_cols_proxy = group_cols + ["proxy_model"]
    else:
        group_cols_proxy = group_cols

    numeric_cols = ["val_acc", "test_acc", "test_f1", "prune_ratio", "runtime"]
    numeric_cols = [c for c in numeric_cols if c in all_df.columns]

    summary = all_df.groupby(group_cols_proxy)[numeric_cols].agg(["mean", "std"]).reset_index()

    # Flatten column names
    summary.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in summary.columns
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")
    print(f"\nSummary:\n{summary.to_string()}")

    return summary


def print_main_table(input_dir: str, exclude_oracle: bool = True):
    """Print a formatted main experiment table."""
    csv_files = list(Path(input_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    all_df = pd.concat(dfs, ignore_index=True)

    if exclude_oracle and "oracle_only" in all_df.columns:
        all_df = all_df[all_df["oracle_only"] != True]
        all_df = all_df[all_df["oracle_only"] != "True"]

    group_cols = ["dataset", "method", "downstream_model"]
    summary = all_df.groupby(group_cols)["test_acc"].agg(["mean", "std"]).reset_index()
    summary["result"] = summary.apply(
        lambda r: f"{r['mean']*100:.2f} ± {r['std']*100:.2f}", axis=1
    )

    pivot = summary.pivot_table(
        index=["dataset", "downstream_model"],
        columns="method",
        values="result",
        aggfunc="first",
    )

    print("\n" + "=" * 80)
    print("Main Experiment Results (test_acc %)")
    print("=" * 80)
    print(pivot.to_string())
    print("=" * 80)

    return pivot
