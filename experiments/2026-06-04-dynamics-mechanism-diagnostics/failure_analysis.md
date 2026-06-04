# Failure Analysis: Mechanism Diagnostics

## Overview

This document describes the negative and ambiguous mechanism evidence from the
training-dynamics diagnostics experiment. The experiment tested whether dynamic
edge-gate gradients contain residual information beyond feature cosine similarity
for bad-edge detection.

**Conclusion: The mechanism claim is not supported.**

## Failure Mode 1: Near-Random Dynamic Gradient

### Evidence

The raw dynamic gradient has AUC ~0.49–0.52 against `bad_edge_mask` across all
datasets and noise types. This is essentially random:

| Dataset   | Noise Type                  | raw_grad AUC | pos_grad AUC |
|-----------|-----------------------------|-------------|--------------|
| Cora      | feature_similar_cross_class | 0.514       | 0.506        |
| CiteSeer  | feature_similar_cross_class | 0.482       | 0.472        |
| PubMed    | feature_similar_cross_class | 0.488       | 0.477        |

The gradient mean is ~0.000004, which is near the numerical precision limit.

### Why This Happens

The first-order edge-gate gradient `d L_score / d m_e` measures how much each
edge contributes to the score loss. When the model is well-trained, most edges
contribute similarly (the loss landscape is flat with respect to individual edge
gates), resulting in near-zero gradients for all edges.

The gradient signal is too weak to distinguish harmful from helpful edges.

### Impact

Since the gradient is near-random, any score that combines it with feature_risk
will be degraded by the noise injection. This explains why the hybrid performs
worse than feature_risk alone.

## Failure Mode 2: Hybrid Inverts Edge Ranking

### Evidence

For Cora/feature_similar_cross_class:
- feature_risk AUC: 0.612 (correctly ranks bad edges higher)
- hybrid AUC: 0.477 (inverts ranking — bad edges ranked *lower*)
- The hybrid is BELOW 0.5, meaning it actively harms detection.

This pattern holds across all datasets:

| Dataset   | feature_risk AUC | hybrid AUC | delta    |
|-----------|-----------------|-----------|----------|
| Cora      | 0.612           | 0.477     | −0.135   |
| CiteSeer  | 0.758           | 0.606     | −0.152   |
| PubMed    | 0.774           | 0.615     | −0.159   |

### Why This Happens

The hybrid score is:
```
hybrid = rank(feature_risk) + 0.1 * rank(pos_grad) - 0.5 * rank(neg_grad)
```

The `−0.5 * rank(neg_grad)` term subtracts the rank of protective edges
(negative gradient). When `neg_grad` is near-random, this subtraction introduces
noise that can flip the relative ordering of edges with similar feature_risk.

Specifically, if two edges have similar feature_risk but the one that is actually
*harmful* happens to have a slightly higher `neg_grad` (by random chance), its
hybrid score will be *lower* (because we subtract `neg_grad`). This inverts the
correct ranking.

### Impact

The hybrid actively degrades edge detection relative to feature_risk alone. This
is not a "no improvement" failure — it is a "makes things worse" failure.

## Failure Mode 3: Misleading Residual AUC

### Evidence

The residual diagnostic shows:
- hybrid residual AUC: 0.887 (seems to support the claim)
- raw_grad residual AUC: 0.515 (does NOT support the claim)

The hybrid's residual Spearman correlation with feature_risk is −0.56, indicating
strong anti-correlation.

### Why This Is Misleading

The residual diagnostic works by:
1. Regressing the dynamic signal on feature_risk
2. Testing whether the residual still detects bad edges

For the hybrid, step 1 removes the feature_risk component. Since the hybrid is
anti-correlated with feature_risk (due to the `−neg_grad` term), the residual
is essentially `−rank(feature_risk)`, which has high AUC because feature_risk
itself has high AUC.

This is a **circular artifact**: the residual captures the *inversion* of
feature_risk, not independent signal from the gradient.

The raw_grad residual AUC of 0.515 is the honest measure: after removing
feature_risk, the gradient has essentially no remaining signal.

### Impact

Any paper that cites the hybrid's residual AUC as evidence for the mechanism
claim would be misleading. The correct interpretation is that the gradient
adds no information beyond feature_risk.

## Failure Mode 4: Shuffle Ablation Shows No Advantage

### Evidence

Mean AUC delta (real hybrid − shuffled hybrid):

| Noise Type                  | Mean Delta | Std  |
|-----------------------------|-----------|------|
| feature_similar_cross_class | 0.003     | 0.003|
| cross_class_oracle          | 0.002     | 0.003|
| low_feature_similarity      | −0.001    | 0.003|

The real hybrid performs almost identically to the shuffled-gradient hybrid.

### Why This Happens

Since the gradient is near-random, shuffling it does not change its statistical
properties. The shuffled gradient is just as (un)informative as the real gradient.

### Impact

This is the most direct evidence against the mechanism claim. If the gradient
contained useful graph-channel information, shuffling it would degrade performance.
The fact that shuffling has no effect proves the gradient is noise.

## Failure Mode 5: Feature-Bin Analysis Contradicts Claim

### Evidence

The experiment's central claim is that dynamic signals help in the
"feature-ambiguous region" (most feature-similar edges). The feature-bin AUC
for Cora/feature_similar_cross_class:

| Bin (most→least similar) | feature_risk | raw_grad | hybrid |
|--------------------------|-------------|----------|--------|
| bin_0 (most similar)     | 0.743       | 0.526    | 0.229  |
| bin_1                    | 0.603       | 0.513    | 0.183  |
| bin_2                    | 0.557       | 0.506    | 0.161  |
| bin_3 (least similar)    | 0.085       | 0.502    | 0.025  |

- feature_risk has AUC 0.743 in bin_0 (best performance, most feature-similar)
- raw_grad has AUC 0.526 in bin_0 (near random)
- hybrid has AUC 0.229 in bin_0 (actively harmful)

### Why This Is Problematic

The claim was that dynamic signals would help specifically in the feature-similar
bin where feature_risk struggles. Instead:
1. feature_risk is *strongest* in bin_0 (AUC 0.743)
2. The gradient is random in bin_0 (AUC 0.526)
3. The hybrid is *worst* in bin_0 (AUC 0.229)

The feature-similar cross-class noise type was designed to create edges where
feature_risk fails. But feature_risk does NOT fail — it has its highest AUC
precisely in the most feature-similar bin.

### Why Feature_risk Works in bin_0

The `feature_similar_cross_class` noise type injects edges between nodes with
*high* feature similarity but *different* classes. After injection, the noisy
graph has:
- Original edges: high feature similarity, same class (clean)
- Injected edges: high feature similarity, different class (bad)

Within bin_0 (most feature-similar), the original edges have slightly higher
feature similarity than the injected edges. feature_risk captures this subtle
difference, achieving AUC 0.743.

The gradient cannot improve on this because it is near-random.

## Failure Mode 6: Prune Metrics Paradox

### Evidence

At prune_ratio=0.2 for Cora/feature_similar_cross_class:

| Score       | Precision | Recall | F1    |
|-------------|----------|--------|-------|
| feature_risk| 0.088    | 0.076  | 0.082 |
| raw_grad    | 0.243    | 0.210  | 0.225 |
| hybrid      | 0.000    | 0.000  | 0.000 |

raw_grad has F1=0.225 (better than feature_risk's 0.082) despite near-random
global AUC.

### Why This Happens

The top 20% of edges by raw_grad score happen to contain more bad edges than
the top 20% by feature_risk. This is a statistical accident of the specific
ranking, not evidence of systematic signal.

With near-random AUC, the top-k selection is essentially random, and by chance
some random orderings will have better F1 than others. This is not robust across
seeds (the std is high).

### Impact

The raw_grad's better F1 at one prune ratio is not reliable evidence. The global
AUC (which averages over all thresholds) is the correct metric, and it shows
raw_grad is near-random.

## Summary of Failure Modes

| Failure Mode | Severity | Implication |
|-------------|----------|-------------|
| Near-random gradient | Critical | Dynamic signal has no edge-level content |
| Hybrid inverts ranking | Critical | Hybrid actively degrades performance |
| Misleading residual AUC | High | Cannot cite residual as mechanism evidence |
| Shuffle ablation = 0 | High | Direct proof gradient is noise |
| Feature-bin contradicts claim | High | feature_risk works in the target regime |
| Prune metrics paradox | Medium | Raw gradient F1 is not robust |

## Recommendations

1. **Do not claim** "training-dynamics signals detect harmful edges beyond
   feature similarity" based on this evidence.

2. **Reframe the paper** around the engineering contribution (hybrid score
   framework) rather than the mechanistic claim.

3. **Investigate why** the prior experiment showed accuracy improvement:
   - Is it from edge weight calibration rather than bad-edge detection?
   - Is it from budget/degree effects?
   - Is it robust across seeds?

4. **Consider alternative mechanisms**:
   - The gradient might help with edge *weight calibration* (soft pruning)
     rather than binary good/bad classification.
   - The gradient might help with training stability rather than edge quality.
   - The gradient might be useful at a different training stage (early vs late).

5. **If pursuing the paper direction**, focus on:
   - The downstream accuracy improvement (0.6256 vs 0.6086) as the main result
   - Frame as "training-dynamics calibration improves graph structure" without
     claiming the mechanism is bad-edge detection
   - Acknowledge the gradient signal is weak and the mechanism is unclear
