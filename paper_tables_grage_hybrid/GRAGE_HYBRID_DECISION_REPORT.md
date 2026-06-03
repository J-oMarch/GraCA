# GraGE-Hybrid Decision Report

## Summary

### 1. Does GraGE-Hybrid exceed Feature-only?

Feature-only mean accuracy: 0.6751

Best hybrid method: GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 (0.6791)

**Result: GraGE-Hybrid DOES exceed Feature-only** (delta = +0.0040)

### 2. Improvement by noise type

- cross_class_oracle: Feature-only=0.6657, Best hybrid=0.6681, delta=+0.0024
- feature_similar_cross_class: Feature-only=0.6086, Best hybrid=0.6256, delta=+0.0170
- degree_aligned_random: Feature-only=0.7080, Best hybrid=0.7092, delta=+0.0012
- random_inter_community: Feature-only=0.6993, Best hybrid=0.7031, delta=+0.0038
- low_feature_similarity: Feature-only=0.6940, Best hybrid=0.6973, delta=+0.0033

### 3. feature_similar_cross_class analysis

Feature-only: 0.6086
Best hybrid: 0.6256
**GraGE-Hybrid exceeds Feature-only on feature_similar_cross_class**

### 4. low_feature_similarity analysis (sanity check)

Feature-only: 0.6940
Best hybrid: 0.6973
Delta: +0.0033

### 5. Positive/negative gradient effectiveness

Methods with positive gradient: mean acc = 0.6736
Methods with negative gradient: mean acc = 0.6627

### 6. Degree normalization effect

### 7. Final Conclusion

**GraGE-Hybrid provides gains over Feature-only.**
The training-dynamics calibration successfully improves upon static feature smoothness.
