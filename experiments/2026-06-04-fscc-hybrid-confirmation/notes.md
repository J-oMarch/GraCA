# Notes

This is the main confirmation experiment for the AAAI direction.

Current evidence shows only a small overall GraGE-Hybrid gain over Feature-only,
but a stronger and statistically significant gain on `feature_similar_cross_class`.
This experiment should test that focused claim with enough seeds and paired
statistics to support or falsify a paper-facing result.

Do not use validation labels or test labels for edge scoring. `bad_edge_mask` is
evaluation-only.

Preferred final claim if supported:

> Training-dynamics-derived edge signals provide residual task information beyond
> static feature similarity in the feature-similar harmful-edge regime.
