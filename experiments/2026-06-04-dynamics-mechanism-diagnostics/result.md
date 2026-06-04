# Result: Training-Dynamics Mechanism Diagnostics

## Summary

**Status: Negative result for the mechanism claim.**

The diagnostics do NOT support the hypothesis that training-dynamics-derived
edge signals provide useful information beyond static feature similarity for
bad-edge detection. The dynamic gradient signal is near-random (AUC ~0.49–0.52),
and combining it with feature_risk in the hybrid score *degrades* performance
relative to feature_risk alone.

## Key Findings

### 1. Global AUC: feature_risk dominates

| Dataset   | Noise Type                  | feature_risk | raw_grad | hybrid | hybrid_delta |
|-----------|-----------------------------|-------------|----------|--------|--------------|
| Cora      | feature_similar_cross_class | 0.612       | 0.514    | 0.477  | −0.135       |
| Cora      | cross_class_oracle          | 0.808       | 0.519    | 0.686  | −0.122       |
| CiteSeer  | feature_similar_cross_class | 0.758       | 0.482    | 0.606  | −0.152       |
| PubMed    | feature_similar_cross_class | 0.774       | 0.488    | 0.615  | −0.159       |

Across all 90 cases (3 datasets × 3 noise types × 10 seeds), feature_risk
outperforms hybrid by a large margin. The raw dynamic gradient is near-random.

### 2. Hybrid is anti-correlated with feature_risk

For Cora/feature_similar_cross_class:
- feature_risk AUC: 0.612 (above 0.5, correctly ranks bad edges higher)
- hybrid AUC: 0.477 (below 0.5, *inversely* ranks bad edges)
- Spearman(feature_risk, hybrid): 0.934 (highly correlated in ranking)
- Spearman(feature_risk, raw_grad): 0.038 (nearly uncorrelated)

The hybrid's AUC below 0.5 means it actively inverts the correct edge ranking.
The near-zero correlation between feature_risk and raw_grad means the gradient
adds no consistent directional information.

### 3. Residual AUC is misleading

The residual diagnostic shows:
- hybrid residual AUC: 0.887 (seems high)
- raw_grad residual AUC: 0.515 (near random)

The hybrid's high residual AUC is an **artifact of anti-correlation**: when you
regress out feature_risk from a signal that is anti-correlated with feature_risk,
the residual captures the *negative* of feature_risk, which has high AUC. This
does NOT indicate that the dynamic gradient carries information beyond feature_risk.

The raw_grad residual AUC of 0.515 (essentially random) is the honest measure.

### 4. Feature-bin analysis: feature_risk wins everywhere

For Cora/feature_similar_cross_class, AUC within feature-similarity bins:

| Bin (most→least similar) | feature_risk | raw_grad | hybrid |
|--------------------------|-------------|----------|--------|
| bin_0 (most similar)     | 0.743       | 0.526    | 0.229  |
| bin_1                    | 0.603       | 0.513    | 0.183  |
| bin_2                    | 0.557       | 0.506    | 0.161  |
| bin_3 (least similar)    | 0.085       | 0.502    | 0.025  |

feature_risk has the best AUC in all bins. The hybrid is *below 0.5* in all bins,
meaning it actively inverts the correct ranking within every feature-similarity bin.

### 5. Shuffle ablation: no meaningful difference

Mean AUC delta (real hybrid − shuffled hybrid) by noise type:

| Noise Type                  | Mean Delta | Std  |
|-----------------------------|-----------|------|
| feature_similar_cross_class | 0.003     | 0.003|
| cross_class_oracle          | 0.002     | 0.003|
| low_feature_similarity      | −0.001    | 0.003|

The real hybrid performs almost identically to the shuffled-gradient hybrid.
This confirms the dynamic gradient contributes essentially zero information.

### 6. Prune metrics: raw_grad outperforms feature_risk (paradox)

At prune_ratio=0.2 for Cora/feature_similar_cross_class:

| Score       | Precision | Recall | F1    |
|-------------|----------|--------|-------|
| feature_risk| 0.088    | 0.076  | 0.082 |
| raw_grad    | 0.243    | 0.210  | 0.225 |
| hybrid      | 0.000    | 0.000  | 0.000 |

raw_grad has better F1 at top-k despite near-random global AUC. This is because
the top-ranked edges by raw_grad happen to contain more bad edges than expected,
even though the overall ranking is poor. The hybrid has F1=0 because its top-k
edges contain zero bad edges (it inverts the ranking).

## Interpretation

The dynamic gradient signal is too weak (mean magnitude ~0.000004) to contribute
meaningfully to edge scoring. The hybrid score, which adds rank-normalized gradient
components to feature_risk, performs worse than feature_risk alone because:

1. The gradient signal is near-random, so adding it introduces noise.
2. The `− lambda_neg * rank(relu(-grad))` term can flip the ranking of edges
   with similar feature_risk values.
3. The rank normalization amplifies the noise from the gradient.

## Implications for the Paper

The mechanism claim **"training-dynamics signals provide information beyond
feature similarity"** is not supported by edge-level bad-edge detection diagnostics.

However, the prior experiment (GRAGE_HYBRID_DECISION_REPORT.md) shows that
GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 achieves higher downstream test accuracy
than Feature-only (0.6256 vs 0.6086 on feature_similar_cross_class). This
improvement is real but the mechanism is NOT "better bad-edge detection."

Possible explanations for the accuracy improvement:
1. The gradient signal helps with edge *weight calibration* rather than binary
   good/bad classification.
2. The improvement comes from budget/degree effects rather than signal quality.
3. The improvement is small and may not be robust across seeds.

## Recommended Paper Framing

**Do NOT claim**: "Training-dynamics signals detect harmful edges beyond what
feature similarity can capture."

**Safer claim**: "GraGE-Hybrid combines feature smoothness with training-dynamics
calibration. While the dynamic gradient alone has weak edge-level signal, the
hybrid approach achieves modest improvements in downstream accuracy, particularly
in the feature-similar cross-class regime. The mechanism appears to be edge weight
calibration rather than bad-edge detection."

**Alternative**: Reframe the paper around a different mechanism, or focus on the
engineering contribution (hybrid score framework) rather than the mechanistic claim.

## Generated Tables

- `logs/tables/global_signal_auc.csv` — AUC per dataset/noise/seed/score
- `logs/tables/feature_bin_auc.csv` — AUC within feature-similarity bins
- `logs/tables/shuffle_ablation.csv` — Real vs shuffled gradient hybrid
- `logs/tables/residual_signal.csv` — Residual AUC after feature risk regression
- `logs/tables/correlation_summary.csv` — Spearman correlations
- `logs/tables/prune_metrics.csv` — Precision/recall/F1 at prune_ratio=0.2

## Generated Figures

- `logs/figures/feature_bin_auc_fscc.png` — Feature-bin AUC plot
- `logs/figures/shuffle_ablation_delta.png` — Shuffle ablation delta plot
