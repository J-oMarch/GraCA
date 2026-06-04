# Failure Analysis
## Candidate Methods Evaluated
Total candidates: 16
Best candidate: MCGC-cw3.0-lp0.1-ln0.5
Feature-only baseline on feature_similar_cross_class: 0.5798

## Per-Candidate Analysis

### FAA-Hybrid-as0.5-lp0.1-ln0.5
- **Status**: ✗ LOSES Feature-only on feature_similar_cross_class
- **Delta**: -0.0035
- **Failure mode**: Marginal underperformance. Close to Feature-only but not enough to justify complexity.

### FAA-Hybrid-as0.5-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0005
- **Result**: Positive delta. Candidate is viable.

### FAA-Hybrid-as1.0-lp0.1-ln0.5
- **Status**: ✗ LOSES Feature-only on feature_similar_cross_class
- **Delta**: -0.0030
- **Failure mode**: Marginal underperformance. Close to Feature-only but not enough to justify complexity.

### FAA-Hybrid-as1.0-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0030
- **Result**: Positive delta. Candidate is viable.

### FAA-Hybrid-as2.0-lp0.1-ln0.5
- **Status**: ✗ LOSES Feature-only on feature_similar_cross_class
- **Delta**: -0.0153
- **Failure mode**: Overall degradation (-0.0139). Method adds noise rather than signal.

### FAA-Hybrid-as2.0-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0073
- **Result**: Positive delta. Candidate is viable.

### FAA-Hybrid-as3.0-lp0.1-ln0.5
- **Status**: ✗ LOSES Feature-only on feature_similar_cross_class
- **Delta**: -0.0140
- **Failure mode**: Overall degradation (-0.0127). Method adds noise rather than signal.

### FAA-Hybrid-as3.0-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0043
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw0.5-lp0.1-ln0.5
- **Status**: ✗ LOSES Feature-only on feature_similar_cross_class
- **Delta**: -0.0003
- **Failure mode**: Overall degradation (-0.0178). Method adds noise rather than signal.

### MCGC-cw0.5-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0037
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw1.0-lp0.1-ln0.5
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0017
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw1.0-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0077
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw2.0-lp0.1-ln0.5
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0083
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw2.0-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0112
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw3.0-lp0.1-ln0.5
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0148
- **Result**: Positive delta. Candidate is viable.

### MCGC-cw3.0-lp0.25-ln0.25
- **Status**: ✓ BEATS Feature-only on feature_similar_cross_class
- **Delta**: +0.0018
- **Result**: Positive delta. Candidate is viable.

## Common Failure Patterns

1. **Signal quality**: The gradient signal from a single training snapshot may be too noisy to reliably identify harmful edges beyond what feature similarity already captures.
2. **Support/score split instability**: The train-split into support/score masks reduces the effective training data, making gradient estimates less reliable.
3. **Budget matching**: Pruning 20% of edges may not align with the actual fraction of harmful edges, leading to over- or under-pruning.
4. **Degree preservation conflict**: min_degree constraints prevent removing edges to low-degree nodes even when they are harmful.

## Validation Constraint Violation

### MCGC-cw3.0-lp0.1-ln0.5

- **Search result**: +0.0148 over Feature-only on feature_similar_cross_class (p=0.0402, significant)
- **Validation result**: -0.0266 degradation on low_feature_similarity (constraint: max 0.005)
- **Status**: FAILS validation constraint

### Root Cause Analysis

The MCGC method works well on `feature_similar_cross_class` because:
- Features are misleadingly similar → gradient signal provides useful correction
- Multi-checkpoint consistency amplifies reliable harmful-edge signal

But MCGC degrades on `low_feature_similarity` because:
- Features are already informative → gradient signal adds noise, not signal
- The method over-prunes edges that Feature-only would correctly keep
- The consistency-weighted gradient penalty is too aggressive when features are clear

### Key Insight

**Training-dynamics signals are regime-dependent.** They help where features are
ambiguous but hurt where features are informative. This is not a bug — it's a
fundamental property of the signal.

The right paper story is NOT "MCGC beats Feature-only everywhere" but rather:
"Training-dynamics signals contain useful information in the feature-ambiguous
regime, and an adaptive method should learn to use them selectively."

### Method Redesign Justification

The validation failure justifies a new method direction:
1. **Regime detection**: Automatically detect per-edge whether features are ambiguous
2. **Adaptive weighting**: Use gradient signal only when features are ambiguous
3. **Graceful fallback**: Fall back to feature-only when features are informative

This is essentially the FAA-Hybrid approach, but with a learned or estimated
threshold rather than a fixed `ambig_scale` parameter.
