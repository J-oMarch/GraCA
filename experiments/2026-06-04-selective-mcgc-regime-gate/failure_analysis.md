# Failure Analysis: Selective MCGC Regime Gate

## Executive Summary

The selective MCGC gate **succeeds as a regularizer** (prevents MCGC degradation
on LFS/DAR) but **fails as a feature-ambiguous regime detector** (cannot improve
FSCC beyond Feature-only). The bottleneck is gradient signal quality, not
threshold selection or gate design.

## What the Selective Gate Achieves

### Prevents MCGC Degradation

| Noise Type | MCGC vs FO | Selective vs FO | Selective vs MCGC |
|-----------|-----------|-----------------|-------------------|
| feature_similar_cross_class | -0.51pp | +0.09pp | +0.60pp |
| low_feature_similarity | **-2.46pp** | **+1.90pp** | **+4.36pp** |
| degree_aligned_random | **-1.91pp** | +0.09pp | +1.99pp |

The selective gate converts MCGC's statistically significant degradation on LFS
(p=0.012) and DAR (p=0.001) into neutral or positive outcomes. This is a real
engineering contribution.

### Best Overall Method

Across all conditions, the selective variant achieves the highest mean accuracy
(0.6787 vs 0.6718 for Feature-only), driven primarily by LFS improvements.

## What the Selective Gate Fails to Achieve

### FSCC Gains Are Within Noise

- Delta: +0.09pp (essentially zero)
- p-value: 0.575 (not significant)
- Win rate: 0.47 (below 0.6 threshold)
- Per-dataset: Cora +0.26pp, CiteSeer -0.08pp, PubMed +0.10pp

The selective gate does not meaningfully improve upon Feature-only on
feature_similar_cross_class. This was the primary target noise type.

### Gate Is Not Regime-Aware

With tau at the median feature similarity (quantile 0.5), the gate activates for
exactly 50% of edges. This is not "regime-aware gating" — it is closer to random
edge selection. A truly regime-aware gate would show:
- Different activation rates across noise types
- Higher activation where features are genuinely ambiguous
- Clear separation between "feature-clear" and "feature-ambiguous" edges

The current gate shows none of these properties because the gradient signal it
gates is too weak to matter.

## Root Cause Analysis

### Primary Bottleneck: Gradient Signal Quality

The edge-gate gradient signal has:
- Mean magnitude: ~0.000003 (near zero)
- Direction: near-random (AUC ~0.50 for bad-edge detection)
- Consistency: high (~0.91) but meaningless (consistent near-zero is still
  near-zero)

Even with multi-checkpoint consistency weighting, the signal cannot meaningfully
alter the edge ranking produced by feature risk alone.

### Secondary Issue: Gate Design Limitation

The hard gate at quantile 0.5 does not implement the intended "feature-ambiguous
regime" concept. Feature-ambiguous edges are those where feature similarity is
HIGH (features don't help distinguish endpoints). The gate should activate for
high-similarity edges, but the median threshold activates for half of all edges
regardless of regime.

A more principled approach would use a higher threshold (e.g., quantile 0.75 or
0.9) to target only the most feature-ambiguous edges. However, search results
show that higher thresholds produce even smaller FSCC gains because fewer edges
receive the (already weak) gradient signal.

### Tertiary Issue: Zero-Gate Contamination

The zero-gate control (tau=2.0, gate always off) should produce results identical
to Feature-only. Instead, it shows +3.2pp improvement on LFS in the search
results. This is caused by the MCGC pipeline consuming random numbers during
checkpoint gradient collection, changing the downstream model initialization.
This contamination makes the zero-gate control unreliable for attributing gains
to the gate mechanism.

## Per-Candidate Analysis (Search Phase)

### Selective-MCGC-hard-q0.5-lp0.1-ln0.5 (Best)
- FSCC delta: +0.0030 (search), +0.0009 (validation)
- LFS delta: +0.0317 (search), +0.0190 (validation)
- Status: Prevents degradation, marginal FSCC gains

### Selective-MCGC-hard-q0.75-lp0.1-ln0.5
- FSCC delta: -0.0018 (search)
- Status: Loses to FO on FSCC

### Selective-MCGC-soft-q0.75-lp0.1-ln0.5
- FSCC delta: -0.0005 (search)
- Status: Marginal underperformance

### Selective-MCGC-hard-q0.9-lp0.1-ln0.5
- FSCC delta: -0.0020 (search)
- Status: Higher threshold = fewer active edges = worse FSCC

### Common Pattern
All selective variants cluster around Feature-only on FSCC (within ±0.5pp). The
gate threshold and type (hard/soft) do not meaningfully affect FSCC performance.
This confirms the bottleneck is signal quality, not gate design.

## Implications for Paper

### What Can Be Claimed
1. "A selective dynamics gate prevents MCGC degradation on low-feature-similarity
   and degree-aligned-random noise types."
2. "The selective variant achieves the highest overall accuracy across all
   conditions."
3. "Training-dynamics signals, when used without selective gating, degrade
   performance on noise types where features are already informative."

### What Cannot Be Claimed
1. "Training-dynamics signals provide useful information beyond feature similarity
   in the feature-ambiguous regime." (FSCC gains within noise)
2. "The selective gate detects feature-ambiguous edges and applies dynamics
   selectively." (Gate is not regime-aware)
3. "The method beats Feature-only on feature_similar_cross_class." (Delta too
   small, not significant)

### Recommended Paper Direction
Option A: **Reframe as gate-as-regularizer**
- Story: "MCGC provides useful consistency signals but adds noise on some
  regimes. A selective gate regularizes MCGC by suppressing dynamics where
  features are informative."
- Strength: Honest, supported by data
- Weakness: Not a strong novelty claim

Option B: **Abandon training-dynamics claim**
- Focus on the engineering contribution (selective gating framework)
- Position Feature-only as the practical method
- Use MCGC/selective as ablation evidence

Option C: **Seek stronger gradient signals**
- The current gradient magnitude (~0.000003) is too weak
- Consider: larger learning rate for edge gates, different loss formulation,
  or alternative gradient computation methods
- This is a research direction, not a paper-ready result

## Conclusion

The selective MCGC regime gate experiment is a **partial success**: it
demonstrates that selective gating can prevent MCGC degradation, but it does not
support the stronger claim that training-dynamics signals improve edge scoring in
feature-ambiguous regions. The bottleneck is fundamental (gradient signal quality)
rather than engineering (gate design, threshold selection).

**Candidate selected for confirmation: No.**
