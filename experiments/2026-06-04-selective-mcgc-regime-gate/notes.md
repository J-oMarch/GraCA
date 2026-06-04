# Notes

First-batch evidence:

- MCGC gained +1.48 pp over Feature-only in search on
  `feature_similar_cross_class`.
- MCGC failed validation because it degraded `low_feature_similarity` by
  -2.66 pp.
- Mechanism diagnostics show raw edge-gate gradients are not reliable bad-edge
  detectors and shuffled-gradient controls show almost no difference.

The next method must therefore be selective: use dynamics only where feature
evidence is ambiguous, and fall back to Feature-only where feature risk is
reliable.

Do not submit until explicitly requested.

