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

These results are no longer sufficient as paper evidence. The first automated
batch found that the confirmation prompt failed operationally, the mechanism
diagnostics contradict the edge-level residual-signal claim, and the adaptive
search found only regime-dependent gains.

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

## First-Batch Outcome

- `2026-06-04-fscc-hybrid-confirmation` failed operationally: Claude exited
  without producing real metrics, so the matched-budget confirmation must be
  rerun with a tighter prompt or direct runner.
- `2026-06-04-dynamics-mechanism-diagnostics` completed 90 cases. It does not
  support the claim that raw edge-gate gradients provide residual bad-edge
  detection signal beyond feature risk: feature risk dominates global AUC, raw
  gradients are near random, and real-vs-shuffled hybrid deltas are around
  `0.003`.
- `2026-06-04-adaptive-grage-search` implemented FAA-Hybrid and MCGC. MCGC
  gained `+1.48 pp` over Feature-only in the search feature-similar cross-class
  regime (`p=0.0402`, win rate `83.3%`), but failed validation overall and
  degraded `low_feature_similarity` by `-2.66 pp`, violating the allowed
  degradation constraint.

## Current Decision

The AAAI claim should shift from "edge gradients detect bad edges beyond feature
similarity" to "training dynamics are regime-dependent and may help when static
features are ambiguous, but must be gated by feature-regime detection." The next
experiment should test a selective dynamics gate that falls back to Feature-only
when feature risk is already reliable.

## Second Batch Prepared

1. `2026-06-04-selective-mcgc-regime-gate`
   tests hard and soft feature-regime gates for MCGC, with no-leak threshold
   selection, shuffled/frozen dynamic controls, zero-gate fallback, and
   threshold sensitivity. Success requires a positive feature-similar
   cross-class delta without more than `-0.5 pp` degradation in
   low-feature-similarity regimes.
2. `2026-06-04-fscc-confirmation-rerun`
   reruns the failed matched-budget confirmation with a direct auditable matrix
   over Cora/CiteSeer/PubMed, 20 FSCC seeds, control regimes, and a small
   heterophily slice.

## Second-Batch Outcome To Date

- `2026-06-04-selective-mcgc-regime-gate` completed. The best variant was
  `Selective-MCGC-hard-q0.5-lp0.1-ln0.5`. It is the best overall validation
  method (`0.6787` vs Feature-only `0.6718`) and converts raw MCGC degradation
  on `low_feature_similarity` from `-2.46 pp` to `+1.90 pp` (`p=0.025`).
  However, the target `feature_similar_cross_class` delta is only `+0.09 pp`
  (`p=0.575`, win rate `0.47`), so the candidate is not selected for
  confirmation and does not satisfy the AAAI stop condition.
- `2026-06-04-fscc-confirmation-rerun` is running after an operational rerun.
  The first attempt only rewrote prompt files and produced placeholder metrics;
  the prompt and runner were patched, smoke succeeded, and the primary matrix is
  currently running.

## Required Reporting

Every experiment must report mean, standard deviation, paired delta vs
Feature-only, effect size, p-value, win rate, runtime, and failure modes. Oracle
results are diagnostic only.
