"""
Paired statistics and residual diagnostics for GraGE experiments.

Provides:
- Paired t-test and Wilcoxon signed-rank test
- Cohen's d effect size
- Win rate
- Runtime ratio
- Comprehensive paired_stats() combining all of the above
- Residual diagnostics: correlation, projection ratio, AUCs
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired samples.

    d = mean(x - y) / std(x - y)

    Args:
        x: first sample.
        y: second sample (same length as x).

    Returns:
        Cohen's d (positive means x > y).
    """
    diff = x - y
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(diff.mean() / sd)


def win_rate(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of pairs where x > y.

    Args:
        x: first sample.
        y: second sample.

    Returns:
        Win rate in [0, 1].
    """
    return float((x > y).mean())


def paired_t_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Paired t-test: t-statistic and two-sided p-value.

    Args:
        x: first sample.
        y: second sample.

    Returns:
        (t_stat, p_value).
    """
    from scipy import stats as sp_stats
    diff = x - y
    n = len(diff)
    if n < 2:
        return 0.0, 1.0
    t_stat, p_value = sp_stats.ttest_rel(x, y)
    return float(t_stat), float(p_value)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test: statistic and two-sided p-value.

    Args:
        x: first sample.
        y: second sample.

    Returns:
        (statistic, p_value). Returns (0.0, 1.0) if all differences are zero.
    """
    from scipy import stats as sp_stats
    diff = x - y
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p_value = sp_stats.wilcoxon(x, y, alternative='two-sided')
        return float(stat), float(p_value)
    except ValueError:
        return 0.0, 1.0


def paired_stats(
    treatment: np.ndarray,
    baseline: np.ndarray,
    treatment_runtimes: Optional[np.ndarray] = None,
    baseline_runtimes: Optional[np.ndarray] = None,
) -> Dict:
    """Compute comprehensive paired statistics.

    Args:
        treatment: accuracy values for the treatment method.
        baseline: accuracy values for the baseline method.
        treatment_runtimes: optional runtime values for treatment.
        baseline_runtimes: optional runtime values for baseline.

    Returns:
        dict with mean, std, delta, p_value, win_rate, cohens_d, runtime info.
    """
    delta_pp = float((treatment.mean() - baseline.mean()) * 100)
    t_stat, t_pval = paired_t_test(treatment, baseline)
    w_stat, w_pval = wilcoxon_test(treatment, baseline)
    d = cohens_d(treatment, baseline)
    wr = win_rate(treatment, baseline)

    result = {
        "treatment_mean": float(treatment.mean()),
        "treatment_std": float(treatment.std(ddof=1)) if len(treatment) > 1 else 0.0,
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0,
        "delta_pp": delta_pp,
        "paired_t_stat": t_stat,
        "paired_t_pvalue": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval,
        "cohens_d": d,
        "win_rate": wr,
        "n_pairs": len(treatment),
    }

    if treatment_runtimes is not None and baseline_runtimes is not None:
        result["treatment_runtime_mean"] = float(treatment_runtimes.mean())
        result["baseline_runtime_mean"] = float(baseline_runtimes.mean())
        if baseline_runtimes.mean() > 1e-8:
            result["runtime_ratio"] = float(treatment_runtimes.mean() / baseline_runtimes.mean())
        else:
            result["runtime_ratio"] = float('inf')

    return result


def compute_residual_diagnostics(
    stability_score: np.ndarray,
    feature_risk: np.ndarray,
    feature_similarity: np.ndarray,
    bad_edge_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Compute residual diagnostics for a stability score.

    Args:
        stability_score: [E] raw stability edge score.
        feature_risk: [E] 1 - cosine similarity.
        feature_similarity: [E] cosine similarity.
        bad_edge_mask: [E] optional binary mask for AUC computation.

    Returns:
        dict with residual-feature correlation, projection ratio,
        residual AUC, raw stability AUC, feature risk AUC.
    """
    from sklearn.metrics import roc_auc_score

    def _rank_normalize(x):
        sorted_indices = np.argsort(x)
        ranks = np.zeros_like(x, dtype=float)
        ranks[sorted_indices] = np.arange(len(x), dtype=float)
        return ranks / max(len(x) - 1, 1)

    R_stab = _rank_normalize(stability_score)
    R_feat = _rank_normalize(feature_risk)
    R_sim = _rank_normalize(feature_similarity)

    # Projection: beta = cov(stab, feat) / var(feat)
    stab_mean = R_stab.mean()
    feat_mean = R_feat.mean()
    cov = ((R_stab - stab_mean) * (R_feat - feat_mean)).mean()
    feat_var = ((R_feat - feat_mean) ** 2).mean()
    beta = cov / max(feat_var, 1e-8)

    # Residual
    residual = R_stab - beta * R_feat
    residual = _rank_normalize(residual)

    # Projection ratio
    proj_ratio = float((beta ** 2 * feat_var) / max(R_stab.var(), 1e-8))

    # Residual-feature-similarity correlation
    resid_corr = float(np.corrcoef(residual, R_sim)[0, 1]) if len(residual) > 1 else 0.0

    result = {
        "projection_beta": float(beta),
        "projection_ratio": proj_ratio,
        "residual_feature_sim_corr": resid_corr,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
    }

    # AUCs (diagnostic only, using bad_edge_mask)
    if bad_edge_mask is not None and len(np.unique(bad_edge_mask)) > 1:
        try:
            result["residual_auc"] = float(roc_auc_score(bad_edge_mask, residual))
        except ValueError:
            result["residual_auc"] = 0.5
        try:
            result["raw_stability_auc"] = float(roc_auc_score(bad_edge_mask, stability_score))
        except ValueError:
            result["raw_stability_auc"] = 0.5
        try:
            result["feature_risk_auc"] = float(roc_auc_score(bad_edge_mask, feature_risk))
        except ValueError:
            result["feature_risk_auc"] = 0.5
        try:
            result["combined_score_auc"] = float(roc_auc_score(
                bad_edge_mask,
                _rank_normalize(feature_risk) + 0.5 * residual,
            ))
        except ValueError:
            result["combined_score_auc"] = 0.5

    return result


def summarize_results_by_method(
    df,
    value_col: str = "test_acc",
    group_cols: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Summarize results grouped by method.

    Args:
        df: pandas DataFrame with experiment results.
        value_col: column to summarize.
        group_cols: additional grouping columns (e.g., ['dataset', 'noise_type']).

    Returns:
        dict mapping method name -> {mean, std, count}.
    """
    if group_cols is None:
        group_cols = []

    all_group_cols = ["method"] + group_cols
    summary = {}
    for name, group in df.groupby(all_group_cols):
        vals = group[value_col].dropna().values
        if len(vals) == 0:
            continue
        key = name if isinstance(name, str) else str(name)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "count": len(vals),
            "values": vals.tolist(),
        }
    return summary


def compute_all_paired_stats(
    df,
    treatment_name: str,
    baseline_name: str = "Feature-only",
    value_col: str = "test_acc",
    runtime_col: str = "runtime",
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """Compute paired statistics between two methods across all groups.

    Args:
        df: pandas DataFrame.
        treatment_name: name of treatment method.
        baseline_name: name of baseline method.
        value_col: column with accuracy values.
        runtime_col: column with runtime values.
        group_cols: columns to group by before pairing.

    Returns:
        dict with overall and per-group paired stats.
    """
    if group_cols is None:
        group_cols = []

    treatment_df = df[df["method"] == treatment_name]
    baseline_df = df[df["method"] == baseline_name]

    if len(treatment_df) == 0 or len(baseline_df) == 0:
        return {"error": f"Missing data for {treatment_name} or {baseline_name}"}

    # Overall paired stats (match by seed + dataset + noise_type)
    merge_cols = ["dataset", "noise_type", "noise_ratio", "seed"]
    merged = treatment_df.merge(
        baseline_df,
        on=merge_cols,
        suffixes=("_treat", "_base"),
    )

    if len(merged) == 0:
        return {"error": "No matching pairs found"}

    overall = paired_stats(
        treatment=merged[f"{value_col}_treat"].values,
        baseline=merged[f"{value_col}_base"].values,
        treatment_runtimes=merged[f"{runtime_col}_treat"].values if f"{runtime_col}_treat" in merged else None,
        baseline_runtimes=merged[f"{runtime_col}_base"].values if f"{runtime_col}_base" in merged else None,
    )

    result = {"overall": overall}

    # Per-group stats
    for group_col in group_cols:
        if group_col not in merged.columns:
            continue
        result[f"by_{group_col}"] = {}
        for group_val, group_df in merged.groupby(f"{group_col}_treat"):
            t_vals = group_df[f"{value_col}_treat"].values
            b_vals = group_df[f"{value_col}_base"].values
            if len(t_vals) >= 2:
                result[f"by_{group_col}"][str(group_val)] = paired_stats(t_vals, b_vals)

    return result
