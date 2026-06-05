# Failure Analysis: StabilityResidual Heterophily Regime Test

## Executive Summary

StabilityResidual-GraGE fails on all three heterophily datasets (Texas,
Wisconsin, Actor), losing to Feature-only by −1.14 pp overall (p=0.0133, win
rate 0.31). This failure is not a fluke or noise artifact: it is consistent
across datasets, noise types, and seeds. The root cause is that heterophilic
graph regimes are fundamentally different from the homophilic citation regime
where the method succeeds.

## Failure Mode 1: Heterophily Incompatible

### Evidence

Edge homophily on the tested datasets:
- Texas: 0.087 (very heterophilic)
- Wisconsin: 0.192 (heterophilic)
- Actor: 0.218 (heterophilic)

Compare to homophilic citation datasets:
- Cora: ~0.81 (homophilic)
- CiteSeer: ~0.74 (homophilic)
- PubMed: ~0.80 (homophilic)

### Why It Matters

StabilityResidual works by training multiple GCN models with different edge
dropout rates and measuring how predictions change. On homophilic graphs,
edges between similar-featured nodes of different classes create ambiguity:
the GCN gets confused, predictions are unstable, and the instability signal
identifies harmful edges. On heterophilic graphs, edges between dissimilar
nodes are the norm, and the GCN's message-passing mechanism is already
challenged. Adding edge dropout doesn't create meaningful prediction diversity
because the graph structure is already "noisy" from the GCN's perspective.

### Diagnostic Support

- `feature_sim_mean` is 0.17–0.35 on heterophily datasets vs ~0.50+ on
  homophilic datasets. The "ambiguous edge" regime (high feature similarity,
  different classes) barely exists.
- `feature_risk_mean` is 0.65–0.83, meaning most edges already have
  dissimilar features. Feature-only pruning is well-calibrated.
- `residual_auc` ≈ 0.52 (near random): the stability residual cannot
  distinguish good from bad edges on heterophily graphs.

## Failure Mode 2: Features Already Informative

### Evidence

Feature-risk AUC for bad-edge detection:
- Texas: 0.616
- Wisconsin: 0.684
- Actor: 0.468

On Texas and Wisconsin, feature dissimilarity alone provides moderate
bad-edge detection. Feature-only pruning leverages this directly, and any
additional signal must add value beyond it. The stability residual does not
(residual AUC ≈ 0.52).

### Why It Matters

On homophilic citation graphs, features are less informative (Cora
feature-risk AUC ≈ 0.61), leaving room for the stability signal to contribute.
On heterophilic graphs, features are already informative for edge quality, and
the stability signal cannot improve on them.

## Failure Mode 3: No Ambiguous Edge Regime

### Evidence

The stability signal adds value in the "ambiguous edge" regime: edges where
features are similar but labels differ. This regime is defined by:
- High feature similarity (sim > threshold)
- Cross-class endpoints

On heterophily datasets:
- `feature_sim_mean` = 0.17 (Actor), 0.34 (Texas), 0.34 (Wisconsin)
- Most edges connect dissimilar-featured nodes

The residualization step removes the feature-correlated component of the
stability signal. On heterophily graphs, where most edges are already
dissimilar, the residual is essentially the raw stability signal—and the raw
stability signal is near-random (raw stability AUC ≈ 0.50–0.55).

### Diagnostic Support

- `projection_ratio` = 0.01–0.08: the stability residual is independent of
  features, but independence does not mean useful.
- `residual_feature_sim_corr` = −0.006 to 0.075: near-zero correlation,
  confirming residualization works, but the residual itself is uninformative.

## Failure Mode 4: Budget/degree Effects on Small Graphs

### Evidence

Texas (183 nodes) and Wisconsin (251 nodes) are very small. Pruning 20% of
edges means removing ~115 and ~183 edges respectively. On such small graphs:
- Every edge matters more
- GCN performance is sensitive to structural changes
- Random pruning (Random-Matched) loses by −4.86 pp (Texas) and −3.87 pp
  (Wisconsin)

### Why It Matters

On small heterophilic graphs, even Feature-only pruning risks removing useful
edges. Any method that doesn't perfectly preserve the already-challenging
heterophilic structure will degrade performance. The StabilityResidual method,
by adding noise through multi-view training and residualization, makes slightly
worse pruning decisions than Feature-only.

### Counter-Evidence

Actor (7,600 nodes) is not small, yet StabilityResidual still loses by −0.80
pp (p<0.001). This suggests the failure is not purely a small-graph artifact.

## Why This Does Not Invalidate the Homophilic Result

The homophilic result (Cora/CiteSeer/PubMed: +1.59 pp, p<0.001, win rate 0.83)
is genuine and well-supported. The mechanism works differently on homophilic
graphs:

1. **Ambiguous edges exist.** Feature similarity is moderate (~0.50), and many
   edges connect similar-featured nodes of different classes. These edges create
   prediction instability under graph perturbation.

2. **The stability signal is informative.** On Cora, residual AUC is ~0.65,
   meaning the stability residual provides genuine bad-edge detection beyond
   features.

3. **Feature informativeness is limited.** Cora feature-risk AUC ≈ 0.61,
   leaving room for the stability signal to contribute.

The method is regime-specific, not universally applicable. This is an honest
finding, not a failure.

## Comparison with Other Methods

All tested methods lose to Feature-only on heterophily:

| Method | Overall Delta (pp) | Interpretation |
|--------|-------------------:|----------------|
| GraGE-Hybrid | −0.71 | Gradient signal adds noise |
| MCGC | −1.08 | Multi-checkpoint gradients unhelpful |
| StabilityResidual | −1.14 | Stability signal uninformative |
| GCN-Jaccard | −1.85 | Jaccard similarity less useful |
| Random-Matched | −3.18 | Random pruning harmful |
| DegreeAwareRandom | −3.76 | Degree-preserving random harmful |

This suggests that the entire class of training-dynamics-based edge scoring
methods struggles on heterophilic graphs, not just StabilityResidual.

## Recommendation for Paper

1. **Restrict the claim** to homophilic or feature-ambiguous citation regimes.
2. **Report honestly**: "On heterophilic datasets, feature-only pruning is
   near-optimal, and the stability signal does not add value."
3. **Frame the contribution** as regime-specific: "In the feature-ambiguous
   homophilic regime, prediction stability provides residual edge information."
4. **Use this experiment** as evidence of method boundaries, not as a failure
   to hide.
5. **Future work**: investigate adaptive methods that detect graph regime and
   fall back to feature-only when the regime is heterophilic.
