import csv
import os
from pathlib import Path
from datetime import datetime


RESULT_FIELDS = [
    "run_id", "timestamp", "seed", "dataset", "method", "oracle_only",
    "proxy_model", "downstream_model", "actual_prune_ratio",
    "num_edges_before", "num_edges_after", "isolated_nodes",
    "min_degree", "mean_degree", "largest_connected_component_ratio",
    "val_acc", "test_acc", "test_f1", "best_epoch", "runtime",
    "config_path", "graph_path", "checkpoint_path",
    # Noisy-edge specific fields
    "bad_edge_precision", "bad_edge_recall", "bad_edge_f1",
    "clean_edge_mistakenly_removed_ratio",
]


def write_result_row(result: dict, csv_path: str):
    """Append a single result row to CSV."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    result["timestamp"] = datetime.now().isoformat()

    file_exists = os.path.exists(csv_path)
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
