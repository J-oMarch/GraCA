"""
Unified result writer for GraCA experiments.
All CSVs produced by new experiments use RESULT_FIELDS as the canonical schema.
"""
import csv
import os
from pathlib import Path
from datetime import datetime


# Canonical schema for ALL experiment types
RESULT_FIELDS = [
    # Identity
    "run_id",
    "timestamp",
    "experiment_type",       # clean | noisy_edge | oracle | ablation | scalability
    "seed",
    "dataset",
    "split_type",            # standard | custom
    # Method
    "method",                # Original | DropEdge | Random-Matched | ...
    "oracle_only",           # True/False
    "proxy_model",           # GCN | GAT | GraphSAGE | -
    "downstream_model",      # GCN | GAT | GraphSAGE
    # Pruning stats
    "prune_ratio_target",    # requested target (may differ from actual)
    "actual_prune_ratio",    # real fraction removed
    "num_edges_before",
    "num_edges_after",
    "isolated_nodes",
    "min_degree",
    "mean_degree",
    "largest_connected_component_ratio",
    # Homophily
    "edge_homophily_before",
    "edge_homophily_after",
    # Accuracy
    "val_acc",
    "test_acc",
    "test_f1",
    "best_epoch",
    "runtime",
    # Provenance
    "config_path",
    "graph_path",
    # Noisy-edge specific (empty for clean experiments)
    "noise_type",            # cross_class_train_safe | cross_class_oracle | low_feature_similarity | random_inter_community
    "noise_ratio",           # 0.05 | 0.10 | 0.20 | 0.30
    "num_injected_edges",
    "bad_edge_precision",
    "bad_edge_recall",
    "bad_edge_f1",
    "clean_edge_mistakenly_removed_ratio",
    # Free text
    "notes",
]


def write_result_row(result: dict, csv_path: str):
    """Append a single result row to CSV.

    Unknown keys in *result* are silently ignored so callers can pass
    extra information without breaking the schema.
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    result.setdefault("timestamp", datetime.now().isoformat())

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def read_results(csv_path: str) -> list:
    """Read all results from CSV."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)
