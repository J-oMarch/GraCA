# Notes: 2026-06-05 Ambiguity Stability Evidence

## Purpose

This experiment is for AAAI evidence construction, not method search. It should
produce P0/P1 evidence for the current StabilityResidual-GraGE claim:

```text
Prediction stability provides complementary edge-quality evidence beyond
feature similarity in homophilic feature-ambiguous citation regimes.
```

## Non-Negotiable Constraints

- The only main method is `StabilityResidual-v5-dp0.15-grad-frozen`.
- Do not restart GraGE-Hybrid, MCGC, or Selective-MCGC.
- Do not create a new main method.
- Do not run open-ended novelty search.
- Do not run new large heterophily or GSL experiments.
- Do not rely on untracked local scripts.

## P0 Interpretation

High-Ambiguity must be defined only by feature-derived signals. The intended
definition is closeness to the Feature-only pruning decision boundary, not high
feature similarity by itself.

Labels and `bad_edge_mask` are allowed only after buckets are assigned, for
diagnostics such as AUC, precision, recall, F1, and changed-prune attribution.

The key question is whether Feature-only to StabilityResidual gains come mainly
from High-Ambiguity bucket changes.

## P1 Interpretation

The alignment destruction controls are essential:

- `Feature+Shuffled Stability` destroys edge-level alignment.
- `Feature+Permuted Stability` destroys node-to-stability alignment while
  preserving the node-stability value distribution.

If either control is competitive with real aligned stability, write that as a
reviewer risk instead of overclaiming.

## Paper Direction After Results

Positive evidence should update the paper toward:

- feature-defined ambiguity region;
- stability residual as aligned complementary evidence;
- honest heterophily failure boundary;
- GSL-inspired proxy competitiveness, not GSL superiority.

Negative or mixed evidence should shrink the claim and update failure analysis,
limitations, and rebuttal risks.

