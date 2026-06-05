# StabilityResidual Heterophily and Regime Test — Decision Report

## Executive Summary

**The heterophily claim is NOT supported.** StabilityResidual-GraGE loses to
Feature-only on all three heterophily datasets (Texas, Wisconsin, Actor) by
**−1.14 pp** overall (p=0.0133, win rate 0.31, Cohen's d −0.07). Every tested
method—including GCN-Jaccard, MCGC, GraGE-Hybrid, and Random-Matched—also loses
to Feature-only on these datasets. The paper must restrict its claim to
homophilic or feature-ambiguous citation regimes.

## Method and Dataset Details

### Selected Candidate

`StabilityResidual-v5-dp0.15-grad-frozen` from
`2026-06-04-stability-channel-rebuild` and
`2026-06-04-stability-ablation-confirmation`.

### Heterophily Datasets

| Dataset | Nodes | Edges | Features | Classes | Edge Homophily |
|---------|------:|------:|---------:|--------:|---------------:|
| Texas | 183 | 574 | 1,703 | 5 | 0.087 |
| Wisconsin | 251 | 916 | 1,703 | 5 | 0.192 |
| Actor | 7,600 | 53,411 | 932 | 5 | 0.218 |

All three datasets have edge homophily well below 0.5, confirming they are
genuinely heterophilic.

### Noise/Regime Matrix

- `feature_similar_cross_class` (noise_ratio=0.3)
- `low_feature_similarity` (noise_ratio=0.3)
- `degree_aligned_random` (noise_ratio=0.3)
- `clean` (no noise)

### Seeds

10 seeds per (dataset × noise × method) combination.

### Methods Compared

1. Feature-only
2. GCN-Jaccard
3. Random-Matched
4. DegreeAwareRandom
5. MCGC (cw=3.0, lp=0.1, ln=0.5)
6. GraGE-Hybrid (FO-posneg, lp=0.1, ln=0.5)
7. StabilityResidual-v5-dp0.15-grad-frozen

## Overall Results

### Mean Test Accuracy by Method

| Method | Mean Acc | Std | Count |
|--------|---------:|----:|------:|
| **Feature-only** | **0.4975** | 0.1557 | 120 |
| GraGE-Hybrid | 0.4904 | 0.1541 | 120 |
| MCGC | 0.4867 | 0.1551 | 120 |
| StabilityResidual | 0.4861 | 0.1536 | 120 |
| GCN-Jaccard | 0.4790 | 0.1426 | 120 |
| Random-Matched | 0.4658 | 0.1433 | 120 |
| DegreeAwareRandom | 0.4600 | 0.1403 | 120 |

Feature-only is the strongest method on heterophily datasets.

### Paired Delta vs Feature-only (Overall)

| Method | Delta (pp) | p-value | Win Rate | Sig |
|--------|----------:|--------:|---------:|-----|
| GraGE-Hybrid | −0.71 | 0.0852 | 0.31 | |
| StabilityResidual | **−1.14** | **0.0133** | **0.31** | * |
| MCGC | −1.08 | 0.0096 | 0.29 | ** |
| GCN-Jaccard | −1.85 | <0.0001 | 0.31 | *** |
| Random-Matched | −3.18 | <0.0001 | 0.19 | *** |
| DegreeAwareRandom | −3.76 | <0.0001 | 0.18 | *** |

All methods lose to Feature-only. StabilityResidual loses by −1.14 pp with
p=0.0133 and win rate 0.31.

## Per-Dataset Detail

### Paired Delta vs Feature-only by Dataset

| Dataset | Method | Delta (pp) | p-value | Win Rate | Sig |
|---------|--------|----------:|--------:|---------:|-----|
| Texas | StabilityResidual | −1.35 | 0.2227 | 0.38 | |
| Texas | GraGE-Hybrid | −0.61 | 0.5261 | 0.38 | |
| Texas | MCGC | −1.69 | 0.0775 | 0.33 | |
| Texas | GCN-Jaccard | −1.96 | 0.0194 | 0.28 | * |
| Texas | Random-Matched | −4.86 | 0.0011 | 0.20 | ** |
| Texas | DegreeAwareRandom | −4.46 | 0.0007 | 0.20 | *** |
| Wisconsin | StabilityResidual | −1.27 | 0.1270 | 0.35 | |
| Wisconsin | GraGE-Hybrid | −0.93 | 0.2444 | 0.33 | |
| Wisconsin | MCGC | −0.69 | 0.3902 | 0.38 | |
| Wisconsin | GCN-Jaccard | −3.48 | 0.0001 | 0.23 | *** |
| Wisconsin | Random-Matched | −3.87 | 0.0002 | 0.17 | *** |
| Wisconsin | DegreeAwareRandom | −5.98 | <0.0001 | 0.07 | *** |
| Actor | StabilityResidual | −0.80 | <0.0001 | 0.20 | *** |
| Actor | GraGE-Hybrid | −0.60 | 0.0001 | 0.23 | *** |
| Actor | MCGC | −0.86 | <0.0001 | 0.17 | *** |
| Actor | GCN-Jaccard | −0.10 | 0.3333 | 0.42 | |
| Actor | Random-Matched | −0.79 | <0.0001 | 0.20 | *** |
| Actor | DegreeAwareRandom | −0.83 | <0.0001 | 0.28 | *** |

StabilityResidual loses on all three datasets. The loss is significant on Actor
(p<0.001) but not individually significant on Texas or Wisconsin.

### Mean Test Accuracy by Dataset × Noise

| Dataset | Noise | FO | StabRes | Delta (pp) |
|---------|-------|---:|--------:|-----------:|
| Texas | feature_similar_cross_class | 0.668 | 0.624 | −4.32 |
| Texas | low_feature_similarity | 0.584 | 0.559 | −2.43 |
| Texas | degree_aligned_random | 0.592 | 0.611 | +1.89 |
| Texas | clean | 0.614 | 0.608 | −0.54 |
| Wisconsin | feature_similar_cross_class | 0.639 | 0.604 | −3.53 |
| Wisconsin | low_feature_similarity | 0.547 | 0.549 | +0.20 |
| Wisconsin | degree_aligned_random | 0.598 | 0.582 | −1.57 |
| Wisconsin | clean | 0.582 | 0.580 | −0.20 |
| Actor | feature_similar_cross_class | 0.277 | 0.269 | −0.80 |
| Actor | low_feature_similarity | 0.282 | 0.280 | −0.26 |
| Actor | degree_aligned_random | 0.296 | 0.284 | −1.22 |
| Actor | clean | 0.292 | 0.283 | −0.93 |

StabilityResidual loses on most (dataset, noise) combinations. The only positive
slices are Texas/degree_aligned_random (+1.89 pp) and
Wisconsin/low_feature_similarity (+0.20 pp).

## Paired Delta by Noise Type

| Noise Type | StabRes Delta (pp) | p-value | Win Rate | Sig |
|------------|-------------------:|--------:|---------:|-----|
| feature_similar_cross_class | −2.89 | 0.0035 | 0.17 | ** |
| low_feature_similarity | −0.83 | 0.4207 | 0.40 | |
| degree_aligned_random | −0.30 | 0.7449 | 0.30 | |
| clean | −0.56 | 0.4645 | 0.37 | |

The worst degradation is on `feature_similar_cross_class` (−2.89 pp, p=0.0035),
which is the same noise type where StabilityResidual showed its strongest gains
on homophilic citation datasets.

## Graph-Regime Diagnostics

### Edge Homophily Before/After Pruning

| Dataset | Method | Before | After | Delta |
|---------|--------|-------:|------:|------:|
| Texas | Feature-only | 0.099 | 0.100 | +0.000 |
| Texas | StabilityResidual | 0.099 | 0.111 | +0.011 |
| Wisconsin | Feature-only | 0.179 | 0.192 | +0.013 |
| Wisconsin | StabilityResidual | 0.179 | 0.188 | +0.009 |
| Actor | Feature-only | 0.205 | 0.203 | −0.001 |
| Actor | StabilityResidual | 0.205 | 0.205 | +0.001 |

StabilityResidual slightly increases homophily after pruning (by removing
cross-class edges), but less effectively than Feature-only on Texas/Wisconsin.

### Feature-Risk AUC (Diagnostic, bad_edge_mask Used for Evaluation Only)

| Dataset | Feature-Risk AUC | Residual AUC | Raw Stability AUC |
|---------|----------------:|-------------:|------------------:|
| Texas | 0.616 | 0.517 | 0.507 |
| Wisconsin | 0.684 | 0.533 | 0.549 |
| Actor | 0.468 | 0.532 | 0.504 |

Feature risk AUC is moderate (0.62–0.68 on Texas/Wisconsin), meaning features
already capture significant edge quality information. Residual AUC is near random
(0.50–0.53), confirming the stability signal adds little on heterophily datasets.

### StabilityResidual Internal Diagnostics

| Metric | Texas | Wisconsin | Actor |
|--------|------:|----------:|------:|
| feature_risk_mean | 0.646 | 0.663 | 0.834 |
| feature_sim_mean | 0.354 | 0.337 | 0.166 |
| projection_ratio | 0.022 | 0.010 | 0.083 |
| residual_feature_sim_corr | 0.006 | −0.006 | 0.075 |
| residual_auc | 0.517 | 0.533 | 0.532 |
| abstention_fraction | 0.101 | 0.100 | 0.100 |

Key observations:

1. **Feature risk mean is high** (0.65–0.83): most edges in heterophily datasets
   have dissimilar features, so feature-based pruning is already well-calibrated.
2. **Feature similarity is low** (0.17–0.35): edges connect dissimilar nodes,
   leaving little room for the "ambiguous edge" regime where stability adds value.
3. **Residual AUC ≈ 0.52**: the stability residual is nearly random for bad-edge
   detection on heterophily graphs.
4. **Projection ratio is low** (0.01–0.08): the residual is independent of
   features, but independence alone does not guarantee useful signal.

### Runtime

| Method | Mean Runtime | Ratio vs FO |
|--------|-------------:|------------:|
| Feature-only | 1.1s | 1.0× |
| GCN-Jaccard | 0.8s | 0.7× |
| Random-Matched | 1.1s | 1.0× |
| DegreeAwareRandom | 0.7s | 0.6× |
| GraGE-Hybrid | 1.8s | 1.6× |
| MCGC | 2.9s | 2.6× |
| StabilityResidual | 4.1s | 3.6× |

Total experiment wall time: 0.4 hours (all methods, 840 experiments).

## Paper-Facing Decision

### Decision Rule Outcome

> If StabilityResidual beats Feature-only by >= +0.5 pp on heterophily with
> stable win rate and no major degradation, the paper can claim broader
> applicability.

**Outcome: FAIL.** StabilityResidual loses by −1.14 pp (p=0.0133, win rate
0.31). It loses on all three datasets and on the `feature_similar_cross_class`
regime where it was strongest on citation graphs.

### Supported Claim

The paper claim must be restricted:

```text
Prediction stability under stochastic graph perturbations provides a
training-dynamics-derived edge signal that improves matched-budget graph
evolution in homophilic, feature-ambiguous regimes (Cora/CiteSeer/PubMed).
On heterophilic graphs (Texas, Wisconsin, Actor), feature-only pruning is
already near-optimal, and the stability signal does not add value.
```

### Why Heterophily Fails

1. **Features are already informative.** On heterophily datasets, feature
   dissimilarity is a strong signal for cross-class edges (feature-risk AUC
   0.62–0.68). Feature-only pruning captures this directly.

2. **Low feature similarity means no "ambiguous edge" regime.** The stability
   signal adds value when features are ambiguous (similar features, different
   classes). On heterophily graphs, features are generally dissimilar, so there
   are few ambiguous edges to benefit from stability information.

3. **The residual signal is near-random.** Residual AUC ≈ 0.52 means the
   stability residual barely distinguishes good from bad edges on heterophily
   graphs. The training dynamics under graph perturbation do not carry useful
   edge-level signal when the graph structure is already heterophilic.

4. **Budget/degree effects dominate on small graphs.** Texas (183 nodes) and
   Wisconsin (251 nodes) are very small. Pruning 20% of edges on these graphs
   has outsized effects, and any method that doesn't perfectly preserve
   structure degrades performance.

### Recommendation

- Restrict the paper claim to homophilic citation regimes.
- Use this experiment as honest failure-mode evidence.
- Do NOT claim that StabilityResidual is a universal graph evolution method.
- Frame the contribution as: "In the feature-ambiguous homophilic regime,
  prediction stability provides residual edge information beyond static
  features."
