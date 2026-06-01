"""
EdgeBench: Binary classifier for harmful edge detection.

Uses edge features (delta_softmax + feature_cosine) to train a classifier
that detects harmful edges. The training signal comes from injected noise
edges (from noise_injection.py), not from teacher pseudo labels.

Key insight: EdgeBench uses the noise injection mechanism itself as the
training signal, making it more practical than EdgeInfluence-Pseudo.

Usage:
    from src.graca.edge_bench import compute_edge_bench_scores, prune_by_edge_bench

    # After injecting noise and computing edge features:
    scores = compute_edge_bench_scores(
        delta_softmax=delta_softmax,
        feature_cosine=feature_cosine,
        bad_edge_mask=bad_edge_mask,
        train_mask=train_mask,
        seed=42
    )

    pruned_ei, mask, stats = prune_by_edge_bench(
        edge_index=edge_index,
        scores=scores,
        num_nodes=num_nodes,
        prune_ratio=0.20,
        undirected=True
    )
"""
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_edge_bench_scores(
    delta_softmax: np.ndarray,
    feature_cosine: np.ndarray,
    bad_edge_mask: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    test_size: float = 0.3,
    classifier: str = "random_forest",
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict:
    """Compute EdgeBench scores using binary classifier.

    Args:
        delta_softmax: [E] delta_softmax scores (higher = more harmful)
        feature_cosine: [E] feature cosine similarity (lower = more harmful)
        bad_edge_mask: [E] boolean mask, True for injected bad edges
        train_mask: [E] optional boolean mask for training subset
        test_size: fraction of data for testing (if train_mask is None)
        classifier: "random_forest" or "logistic_regression"
        n_estimators: number of trees for RandomForest
        seed: random seed

    Returns:
        dict with keys:
            scores: [E] EdgeBench probability scores (higher = more harmful)
            auc_roc: ROC-AUC on test set
            auc_pr: PR-AUC on test set
            classifier: trained classifier object
            feature_importance: [2] feature importance (delta_softmax, cosine)
            diagnostics: dict with training statistics
    """
    # Build feature matrix: [E, 2]
    # Feature 1: delta_softmax (higher = more harmful)
    # Feature 2: -feature_cosine (higher = more harmful, i.e., lower cosine)
    X = np.column_stack([delta_softmax, -feature_cosine])
    y = bad_edge_mask.astype(int)

    # Split into train/test
    if train_mask is not None:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    # Train classifier
    if classifier == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=5,
            random_state=seed,
            class_weight="balanced"
        )
    elif classifier == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X_train, y_train)

    # Predict probabilities for all edges
    proba = clf.predict_proba(X)[:, 1]  # P(bad_edge)

    # Evaluate on test set
    test_proba = clf.predict_proba(X_test)[:, 1]
    try:
        auc_roc = roc_auc_score(y_test, test_proba)
    except ValueError:
        auc_roc = 0.5  # Only one class present

    try:
        precision, recall, _ = precision_recall_curve(y_test, test_proba)
        auc_pr = auc(recall, precision)
    except ValueError:
        auc_pr = 0.0

    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        feat_imp = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        feat_imp = np.abs(clf.coef_[0])
    else:
        feat_imp = np.array([0.5, 0.5])

    diagnostics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_pos_frac": float(y_train.mean()),
        "test_pos_frac": float(y_test.mean()),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "classifier_type": classifier,
    }

    logger.info(f"EdgeBench: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")
    logger.info(f"  Feature importance: delta_softmax={feat_imp[0]:.3f}, cosine={feat_imp[1]:.3f}")

    return {
        "scores": proba,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "classifier": clf,
        "feature_importance": feat_imp,
        "diagnostics": diagnostics,
    }


def prune_by_edge_bench(
    edge_index: torch.Tensor,
    scores: np.ndarray,
    num_nodes: int,
    prune_ratio: float = 0.20,
    undirected: bool = True,
    protect_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Prune graph using EdgeBench scores.

    Args:
        edge_index: [2, E] edge index
        scores: [E] EdgeBench probability scores (higher = more harmful)
        num_nodes: number of nodes
        prune_ratio: fraction of edges to remove
        undirected: whether graph is undirected
        protect_self_loops: whether to protect self-loops

    Returns:
        pruned_edge_index: [2, E'] pruned edge index
        prune_mask: [E] boolean mask, True for pruned edges
        stats: dict with pruning statistics
    """
    from src.graca.pruning import prune_graph

    # Convert scores to risk_score tensor
    risk_score = torch.from_numpy(scores).float()

    # Use existing prune_graph with target_prune_ratio
    pruned_ei, prune_mask, stats = prune_graph(
        edge_index=edge_index,
        risk_score=risk_score,
        num_nodes=num_nodes,
        beta=0.2,  # Not used when target_prune_ratio is set
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
    """Simple EdgeBench score without training (z-score combination).

    This is the fallback when no noise injection labels are available.
    Uses z-score normalization: zscore(delta_softmax) + zscore(-cosine).

    Args:
        delta_softmax: [E] delta_softmax scores
        feature_cosine: [E] feature cosine similarity

    Returns:
        scores: [E] combined risk scores (higher = more harmful)
    """
    from scipy.stats import zscore as sp_zscore

    z_ds = sp_zscore(delta_softmax)
    z_cos = sp_zscore(-feature_cosine)

    # Handle NaN from constant arrays
    z_ds = np.nan_to_num(z_ds, nan=0.0)
    z_cos = np.nan_to_num(z_cos, nan=0.0)

    return z_ds + z_cos
