"""
EdgeBench: Binary classifier for harmful edge detection.

Two protocols:
1. EdgeBench-InGraphSupervised: Uses target graph's bad_edge_mask for training.
   DIAGNOSTIC ONLY - oracle_only=True, not for practical claims.
2. EdgeBench-Transfer: Trains on source graph units, evaluates on target graph.
   PRACTICAL - no target label leakage.

Usage:
    from src.graca.edge_bench import (
        compute_edge_bench_in_graph,
        compute_edge_bench_transfer,
        prune_by_edge_bench,
    )
"""
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_edge_bench_in_graph(
    delta_softmax: np.ndarray,
    feature_cosine: np.ndarray,
    bad_edge_mask: np.ndarray,
    test_size: float = 0.3,
    classifier: str = "random_forest",
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict:
    """EdgeBench-InGraphSupervised: Train on target graph's bad_edge_mask.

    DIAGNOSTIC ONLY - uses target graph labels. oracle_only=True.
    Not for practical claims.

    Args:
        delta_softmax: [E] delta_softmax scores
        feature_cosine: [E] feature cosine similarity
        bad_edge_mask: [E] boolean mask, True for injected bad edges
        test_size: fraction for test split
        classifier: "random_forest" or "logistic_regression"
        n_estimators: number of trees
        seed: random seed

    Returns:
        dict with scores, auc_roc, auc_pr, classifier, diagnostics
    """
    X = np.column_stack([delta_softmax, -feature_cosine])
    y = bad_edge_mask.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = _train_classifier(classifier, n_estimators, seed)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X)[:, 1]
    auc_roc, auc_pr = _evaluate(clf, X_test, y_test)

    diagnostics = {
        "protocol": "in_graph_supervised",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_pos_frac": float(y_train.mean()),
        "test_pos_frac": float(y_test.mean()),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "classifier_type": classifier,
    }

    logger.info(f"EdgeBench-InGraphSupervised: AUC-ROC={auc_roc:.4f}")

    return {
        "scores": proba,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "classifier": clf,
        "diagnostics": diagnostics,
    }


def compute_edge_bench_transfer(
    target_delta_softmax: np.ndarray,
    target_feature_cosine: np.ndarray,
    source_features_list: List[Dict],
    classifier: str = "random_forest",
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict:
    """EdgeBench-Transfer: Train on source graph units, predict on target.

    PRACTICAL - no target label leakage.

    Args:
        target_delta_softmax: [E_target] target graph delta_softmax scores
        target_feature_cosine: [E_target] target graph feature cosine
        source_features_list: List of source graph feature dicts, each with:
            - delta_softmax: [E_source]
            - feature_cosine: [E_source]
            - bad_edge_mask: [E_source]
        classifier: classifier type
        n_estimators: number of trees
        seed: random seed

    Returns:
        dict with scores, diagnostics
    """
    # Aggregate source training data
    X_sources = []
    y_sources = []
    for src in source_features_list:
        X_src = np.column_stack([src["delta_softmax"], -src["feature_cosine"]])
        y_src = src["bad_edge_mask"].astype(int)
        X_sources.append(X_src)
        y_sources.append(y_src)

    X_train = np.concatenate(X_sources, axis=0)
    y_train = np.concatenate(y_sources, axis=0)

    # Train on source data
    clf = _train_classifier(classifier, n_estimators, seed)
    clf.fit(X_train, y_train)

    # Predict on target
    X_target = np.column_stack([target_delta_softmax, -target_feature_cosine])
    proba = clf.predict_proba(X_target)[:, 1]

    diagnostics = {
        "protocol": "transfer",
        "train_size": len(X_train),
        "train_pos_frac": float(y_train.mean()),
        "n_sources": len(source_features_list),
        "classifier_type": classifier,
    }

    logger.info(f"EdgeBench-Transfer: trained on {len(source_features_list)} sources, "
                f"{len(X_train)} edges")

    return {
        "scores": proba,
        "classifier": clf,
        "diagnostics": diagnostics,
    }


def compute_edge_bench_transfer_leave_one_out(
    experiments_data: Dict[str, Dict],
    target_key: str,
    leave_out: str = "seed",
    classifier: str = "random_forest",
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict:
    """EdgeBench-Transfer with leave-one-out protocol.

    Args:
        experiments_data: Dict mapping (dataset, noise_type, seed) -> feature dict
            Each feature dict has: delta_softmax, feature_cosine, bad_edge_mask
        target_key: key of target experiment (e.g., "Cora_cross_class_oracle_0")
        leave_out: "seed" (leave-one-seed-out) or "dataset" (leave-one-dataset-out)
        classifier: classifier type
        n_estimators: number of trees
        seed: random seed

    Returns:
        dict with scores, diagnostics
    """
    # Parse target key: dataset_noise_type_seed
    parts = target_key.rsplit("_", 2)
    target_dataset = parts[0]
    target_noise = parts[1]
    target_seed = parts[2]

    # Select source experiments
    source_features = []
    for key, data in experiments_data.items():
        if key == target_key:
            continue

        k_parts = key.rsplit("_", 2)
        k_dataset = k_parts[0]
        k_noise = k_parts[1]
        k_seed = k_parts[2]

        if leave_out == "seed":
            # Same dataset and noise, different seed
            if k_dataset == target_dataset and k_noise == target_noise and k_seed != target_seed:
                source_features.append(data)
        elif leave_out == "dataset":
            # Different dataset, any noise/seed
            if k_dataset != target_dataset:
                source_features.append(data)

    if not source_features:
        logger.warning(f"No source experiments found for {target_key} with leave_out={leave_out}")
        # Fallback: use all other experiments
        for key, data in experiments_data.items():
            if key != target_key:
                source_features.append(data)

    target_data = experiments_data[target_key]
    return compute_edge_bench_transfer(
        target_delta_softmax=target_data["delta_softmax"],
        target_feature_cosine=target_data["feature_cosine"],
        source_features_list=source_features,
        classifier=classifier,
        n_estimators=n_estimators,
        seed=seed,
    )


def prune_by_edge_bench(
    edge_index: torch.Tensor,
    scores: np.ndarray,
    num_nodes: int,
    prune_ratio: float = 0.20,
    undirected: bool = True,
    protect_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Prune graph using EdgeBench scores."""
    from src.graca.pruning import prune_graph

    risk_score = torch.from_numpy(scores).float()

    pruned_ei, prune_mask, stats = prune_graph(
        edge_index=edge_index,
        risk_score=risk_score,
        num_nodes=num_nodes,
        beta=0.2,
        min_degree=1,
        undirected=undirected,
        protect_self_loops=protect_self_loops,
        target_prune_ratio=prune_ratio,
    )

    return pruned_ei, prune_mask, stats


def compute_edge_bench_scores_simple(
    delta_softmax: np.ndarray,
    feature_cosine: np.ndarray,
) -> np.ndarray:
    """Simple z-score combination fallback."""
    from scipy.stats import zscore as sp_zscore

    z_ds = sp_zscore(delta_softmax)
    z_cos = sp_zscore(-feature_cosine)
    z_ds = np.nan_to_num(z_ds, nan=0.0)
    z_cos = np.nan_to_num(z_cos, nan=0.0)
    return z_ds + z_cos


def _train_classifier(classifier: str, n_estimators: int, seed: int):
    """Train a classifier."""
    if classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=5,
            random_state=seed,
            class_weight="balanced"
        )
    elif classifier == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


def _evaluate(clf, X_test, y_test):
    """Evaluate classifier on test set."""
    test_proba = clf.predict_proba(X_test)[:, 1]
    try:
        auc_roc = roc_auc_score(y_test, test_proba)
    except ValueError:
        auc_roc = 0.5
    try:
        precision, recall, _ = precision_recall_curve(y_test, test_proba)
        auc_pr = auc(recall, precision)
    except ValueError:
        auc_pr = 0.0
    return auc_roc, auc_pr
