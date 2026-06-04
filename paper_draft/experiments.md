# Experiments Plan

## Current Evidence

Historical GraGE-Hybrid tables report:

- Overall Feature-only mean accuracy: `0.6751`.
- Best hybrid mean accuracy: `0.6791`.
- Overall delta: `+0.0040`.
- Feature-similar cross-class delta for
  `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`: `+0.0170` over 9 paired cases.
- Earlier raw edge-gate and unrolled scores did not beat Feature-only, so the
  current method claim must focus on hybrid dynamics and residual diagnostics.

These results are promising but not AAAI-ready until confirmed with more seeds,
effect sizes, win rates, heterophily failure analysis, and no-leak mechanism
evidence.

## First Batch

1. `2026-06-04-fscc-hybrid-confirmation`
   confirms matched-budget GraGE-Hybrid vs Feature-only on Cora/CiteSeer/PubMed
   with 20 seeds and a small heterophily slice.
2. `2026-06-04-dynamics-mechanism-diagnostics`
   tests whether dynamic edge signals remain useful after controlling for
   feature cosine.
3. `2026-06-04-adaptive-grage-search`
   searches for a stronger no-leak training-dynamics mechanism suitable for a
   larger confirmation run.

## Required Reporting

Every experiment must report mean, standard deviation, paired delta vs
Feature-only, effect size, p-value, win rate, runtime, and failure modes. Oracle
results are diagnostic only.

