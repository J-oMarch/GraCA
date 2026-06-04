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

The old edge-gradient hybrid claim is not AAAI-ready, but the method rebuild has
produced a viable new main direction. The supported training-dynamics signal is
prediction stability under graph perturbations, residualized against feature
similarity. Edge-gate gradients should be presented as local sensitivity and
abstention/regularization evidence, not as the primary source of the current
accuracy gain.

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

## Second-Batch Outcome

- `2026-06-04-selective-mcgc-regime-gate` completed. The best variant was
  `Selective-MCGC-hard-q0.5-lp0.1-ln0.5`. It is the best overall validation
  method (`0.6787` vs Feature-only `0.6718`) and converts raw MCGC degradation
  on `low_feature_similarity` from `-2.46 pp` to `+1.90 pp` (`p=0.025`).
  However, the target `feature_similar_cross_class` delta is only `+0.09 pp`
  (`p=0.575`, win rate `0.47`), so the candidate is not selected for
  confirmation and does not satisfy the AAAI stop condition.
- `2026-06-04-fscc-confirmation-rerun` completed the direct confirmation matrix
  after the operational rerun. On the primary FSCC target with 20 seeds across
  Cora/CiteSeer/PubMed, Feature-only is the best overall method (`0.6116 ±
  0.0496`). GraGE-Hybrid loses by `-2.50 pp` (`p=0.0012`, win rate `0.10`,
  Cohen's d `-1.40`), and MCGC loses by `-0.72 pp` (`p=0.143`, win rate
  `0.43`).
- The only positive MCGC slice is Cora FSCC (`+1.58 pp`, `p<0.001`), but
  Random-Matched (`+4.35 pp`) and DegreeAwareRandom (`+4.00 pp`) also beat
  Feature-only on Cora. This makes the Cora gain look like a pruning
  budget/degree effect rather than evidence that training dynamics add residual
  edge information.
- On control regimes, Feature-only and GCN-Jaccard are tied (`0.6903` vs
  `0.6905`), while GraGE-Hybrid (`0.6716`) and MCGC (`0.6627`) lose. On the
  heterophily slice, Feature-only again wins (`0.5155`), while GraGE-Hybrid
  (`0.5044`) and MCGC (`0.4985`) lose.

## Method Rebuild Direction

Do not continue adding small sweeps around the current rank-normalized hybrid
or MCGC score. The bottleneck is the signal itself: gradient magnitudes are
near-zero, gradient signs are close to random after feature-risk control, and
rank normalization turns tiny fluctuations into pruning decisions. A viable next
method must change the information channel, for example:

- score edges from validation-free prediction stability under graph
  perturbations, then use edge-gate gradients only as a consistency constraint;
- learn a no-leak support/score split where dynamics are estimated on one
  graph view and applied to another;
- replace rank-normalized gradient addition/subtraction with calibrated
  uncertainty-aware gating that can abstain on near-zero gradients;
- treat current GraGE as a diagnostic framework and make the paper's main claim
  about when training-dynamics edge signals fail relative to static similarity.

## Third-Batch Outcome

- `2026-06-04-stability-channel-rebuild` implemented StabilityResidual-GraGE.
  The method trains multiple stochastic graph views, computes node prediction
  entropy/JSD/variance/confidence, converts instability to edge scores,
  residualizes against feature risk, and optionally uses edge-gate gradient
  confidence as an abstention rule.
- Search over Cora/CiteSeer found all real StabilityResidual variants beating
  Feature-only on FSCC. The selected candidate was
  `StabilityResidual-v5-dp0.15-grad-frozen`, with FSCC search delta `+2.56 pp`
  (`p=0.0010`, win rate `0.90`). The no-gradient variant was close (`+2.45 pp`),
  so the main signal is prediction stability, not temporal edge-gradient
  evolution.
- Validation over Cora/CiteSeer/PubMed, 10 seeds, and FSCC/LFS/DAR controls
  supports the stability channel. On FSCC, StabilityResidual beats Feature-only
  by `+2.00 pp` (`p=0.0001`, win rate `0.87`, Cohen's d `0.41`). It also avoids
  material degradation on low-feature-similarity (`+0.73 pp`, `p=0.1638`) and
  degree-aligned-random (`+0.30 pp`, `p=0.3802`) controls.
- Per-dataset FSCC effects are positive but uneven: Cora `+4.43 pp`
  (`p=0.0001`, 10/10 wins), CiteSeer `+0.72 pp` (`p=0.2555`), PubMed `+0.84 pp`
  (`p=0.1054`). The paper should report this as consistent positive direction
  with strongest evidence on Cora, not as uniformly significant per dataset.
- Diagnostics support residual value beyond feature similarity: projection ratio
  `<0.005`, residual-feature-similarity correlation `<0.01`, and residual AUC
  around `0.65`. Shuffled-stability is not significant (`+1.90 pp`,
  `p=0.1456`, win rate `0.50`), but gradient-frozen/shuffled controls show that
  edge-gradient confidence is not the dominant driver.

## Updated Paper-Facing Claim

The strongest current claim is:

```text
Prediction stability under stochastic graph perturbations provides a
train-dynamics-derived edge signal that is residual to feature similarity and
improves matched-budget graph evolution in feature-ambiguous homophilic regimes.
```

The claim still needs heterophily validation, residualization ablation, dropout
schedule sensitivity, and comparison to graph structure learning baselines
before the stop condition for an AAAI-ready final package is fully satisfied.

## Required Reporting

Every experiment must report mean, standard deviation, paired delta vs
Feature-only, effect size, p-value, win rate, runtime, and failure modes. Oracle
results are diagnostic only.
