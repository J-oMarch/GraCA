"""
Validate result CSVs in results_clean/.

Checks:
1. All CSVs have consistent columns matching RESULT_FIELDS
2. No old field "prune_ratio" (must be "actual_prune_ratio")
3. Method names are unified
4. No duplicate rows (same dataset/method/model/seed/noise_ratio)
5. oracle_only=True results are flagged (should not appear in practical main table)

Usage:
    python scripts/validate_results.py --dir results_clean/
"""
import sys
import os
import argparse
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.result_writer import RESULT_FIELDS


VALID_METHODS = {
    "Original", "Original+Noise", "DropEdge",
    "Random-Matched", "DegreeAwareRandom-Matched",
    "Similarity-Pruning", "Homophily-TrainOnly",
    "GraCA-lite", "GraCA-Oracle",
    # Ablation variants
    "no_gradient_direction", "direction_only", "harmful_only", "helpful_only",
    "no_relative_strength", "no_uncertainty", "no_ema", "hard_pseudo",
    "global_threshold", "no_local_budget", "first_layer", "last_layer",
    "all_layers", "deterministic_off", "full_GraCA-lite",
}


def validate_csv(csv_path: str) -> list:
    """Validate a single CSV file. Returns list of error strings."""
    errors = []
    path = Path(csv_path)

    if not path.exists():
        return [f"File not found: {csv_path}"]

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if headers is None:
            errors.append(f"{csv_path}: empty file or no headers")
            return errors

        # Check 1: No old field "prune_ratio"
        if "prune_ratio" in headers:
            errors.append(f"{csv_path}: contains old field 'prune_ratio' (must be 'actual_prune_ratio')")

        # Check 2: All required fields present
        for field in RESULT_FIELDS:
            if field not in headers:
                errors.append(f"{csv_path}: missing required field '{field}'")

        # Check 3: No extra fields (except known legacy fields)
        allowed_extra = {"checkpoint_path", "split_type"}
        for h in headers:
            if h not in RESULT_FIELDS and h not in allowed_extra:
                errors.append(f"{csv_path}: unexpected field '{h}'")

        # Check rows
        seen_keys = defaultdict(int)
        oracle_in_practical = []

        for i, row in enumerate(reader, start=2):
            # Check method name
            method = row.get("method", "")
            if method and method not in VALID_METHODS:
                errors.append(f"{csv_path} line {i}: unknown method '{method}'")

            # Check for duplicates
            key = (
                row.get("dataset", ""),
                row.get("method", ""),
                row.get("downstream_model", ""),
                row.get("seed", ""),
                row.get("noise_ratio", ""),
            )
            seen_keys[key] += 1

            # Check oracle in practical
            if row.get("oracle_only", "") in ("True", "true", "1"):
                oracle_in_practical.append(i)

        # Report duplicates
        for key, count in seen_keys.items():
            if count > 1:
                errors.append(f"{csv_path}: duplicate rows for {key} (count={count})")

        # Warn about oracle
        if oracle_in_practical:
            errors.append(f"{csv_path}: WARNING - {len(oracle_in_practical)} oracle rows found "
                          f"(lines {oracle_in_practical[:5]}...). "
                          f"These should NOT appear in practical main tables.")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results_clean/",
                        help="Directory to scan for CSV files")
    args = parser.parse_args()

    csv_files = list(Path(args.dir).rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.dir}")
        return

    all_errors = []
    for csv_file in sorted(csv_files):
        errors = validate_csv(str(csv_file))
        if errors:
            all_errors.extend(errors)
        else:
            print(f"  OK: {csv_file}")

    if all_errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION FAILED: {len(all_errors)} errors found")
        print(f"{'='*60}")
        for err in all_errors:
            print(f"  ERROR: {err}")
        sys.exit(1)
    else:
        print(f"\nAll {len(csv_files)} CSV files validated successfully.")


if __name__ == "__main__":
    main()
