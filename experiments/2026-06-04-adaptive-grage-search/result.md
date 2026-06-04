# Adaptive GraGE Search — Decision Report

## Summary

**Best candidate**: `MCGC-cw3.0-lp0.1-ln0.5`
**Method type**: `mcgc`
**Candidate family**: Multi-Checkpoint Gradient Consistency

## Overall Performance

| Method | Mean Test Acc |
|--------|--------------|
| Feature-only | 0.6136 |
| GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 | 0.5952 |
| MCGC-cw3.0-lp0.1-ln0.5 | 0.6088 |

## Paired Deltas vs Feature-only

| Method | Noise Type | Delta vs Feature-only | p-value | Significant |
|--------|-----------|----------------------|---------|-------------|
| MCGC-cw3.0-lp0.1-ln0.5 | feature_similar_cross_class | +0.0148 ± 0.0132 | 0.0402 | Yes |
| MCGC-cw3.0-lp0.1-ln0.5 | cross_class_oracle | -0.0245 ± 0.0234 | 0.0506 | No |

## Key Result: feature_similar_cross_class

- Feature-only mean: 0.5798
- MCGC-cw3.0-lp0.1-ln0.5 mean: 0.5947
- Delta: +0.0148 ± 0.0132
- p-value: 0.0402
- Significant: Yes

## vs Current Best Hybrid (feature_similar_cross_class)

- Hybrid mean: 0.5710
- MCGC-cw3.0-lp0.1-ln0.5 mean: 0.5947
- Delta: +0.0237

## Algorithmic Contribution

The MCGC method uses gradient sign consistency across multiple training
checkpoints as a confidence signal. Edges whose harmful gradient is consistent
across training stages are more reliably harmful than edges with unstable
gradients.

**Key insight**: A single training snapshot may give noisy signals. Consistency
across checkpoints indicates reliable edge-level information.

## Validation Results (5 seeds × 3 datasets × 3 noise types)

### Overall Validation Performance

| Method | Mean Test Acc | Std |
|--------|--------------|-----|
| Feature-only | 0.6711 | 0.0681 |
| MCGC-cw3.0-lp0.1-ln0.5 | 0.6568 | 0.0618 |
| GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 | 0.6562 | 0.0736 |
| Random-Matched | 0.6465 | 0.0666 |

### Validation Constraint Check

- **Constraint**: No more than 0.005 mean degradation vs Feature-only on `low_feature_similarity`
- **Observed degradation**: -0.0266
- **Constraint VIOLATED**: MCGC loses to Feature-only by 0.0266 on `low_feature_similarity`

### Interpretation

MCGC-cw3.0-lp0.1-ln0.5 shows genuine gains on `feature_similar_cross_class` (+0.0148)
where feature similarity is misleading. However, it degrades on `low_feature_similarity`
where features are already informative and the gradient signal adds noise rather than
signal.

This is a meaningful finding: **the gradient signal is useful precisely where features
are ambiguous, but harmful where features are already informative**. This suggests the
method needs an automatic regime detector — use gradient signal when features are
ambiguous, fall back to feature-only when features are clear.

## Candidate Selected for Confirmation

**No** — while MCGC shows +0.0148 on feature_similar_cross_class, it violates the
low_feature_similarity degradation constraint (-0.0266 vs allowed -0.005).

## Recommended Paper Framing

The results support a nuanced paper story:

1. **Training-dynamics signals DO contain useful information** beyond static feature
   similarity, specifically in the feature-ambiguous regime.
2. **But raw gradient signals are noisy** and can degrade performance when features
   are already informative.
3. **The right approach is adaptive**: use gradient signal where features are
   ambiguous, trust features where they are clear.

## Next Experiment

Design an adaptive method that automatically detects the feature regime per-edge and
adjusts gradient weighting accordingly — essentially combining FAA-Hybrid's regime
detection with MCGC's consistency signal. The key innovation would be:

```
if feature_similarity > threshold:
    use gradient-weighted score (trust dynamics)
else:
    use feature-only score (trust features)
```

This would require learning or estimating the threshold from training data without
label leakage.
