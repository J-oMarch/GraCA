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

## Fourth-Batch Outcome To Date

- `2026-06-04-stability-ablation-confirmation` completed the core paper-facing
  ablation and 20-seed confirmation. On the 20-seed confirmation over
  Cora/CiteSeer/PubMed and FSCC/LFS/DAR, StabilityResidual-frozen has overall
  mean `0.6806` vs Feature-only `0.6708`.
- The primary FSCC claim strengthens to `+1.59 pp` over Feature-only
  (`p<0.001`, win rate `0.83`, Cohen's d `0.70`). StabilityResidual is the only
  method that beats Feature-only on FSCC; GCN-Jaccard, MCGC, GraGE-Hybrid,
  Random-Matched, and DegreeAwareRandom all lose or tie.
- Control regimes pass: LFS `+0.55 pp` (`p=0.144`, no degradation) and DAR
  `+0.81 pp` (`p<0.001`, win rate `0.72`).
- Per-dataset FSCC: Cora `+3.19 pp` (`20/20` wins), PubMed `+1.05 pp`
  (`p=0.002`, win rate `0.80`), CiteSeer `+0.52 pp` (`p=0.353`, win rate
  `0.70`). CiteSeer remains the weakest dataset but positive.
- Ablations show raw and residualized stability are nearly tied: raw
  `+1.25 pp`, residualized `+1.11 pp`, raw-vs-residualized difference
  `+0.14 pp` (`p=0.772`). Residualization is mainly a principled paper device
  for the "beyond feature similarity" claim, not a large accuracy driver.
- Gradient confidence is auxiliary but useful: no-gradient `+1.29 pp`, shuffled
  gradient `+1.43 pp`, frozen gradient `+1.79 pp`, real gradient `+1.95 pp` in
  the 5-seed FSCC ablation. The main signal remains prediction stability.
- Dropout schedule and view-count sensitivity are acceptable. All tested
  dropout schedules beat Feature-only; 3 and 5 views are both strong, while 7
  views is weaker.

## Current Stop-Condition Assessment

The homophilic citation portion now satisfies the main statistical stop
condition: mean improvement over Feature-only is above `0.5 pp`, multi-seed
stability is strong, ablations and failure analysis exist, and theory explains
the residual and abstention mechanisms. Heterophily validation is complete and
negative, so the paper must be regime-limited rather than universal. GSL proxy
positioning is complete enough for an honest non-superiority claim.
`2026-06-05-ambiguity-stability-evidence` now completes the P0 ambiguity and P1
alignment gates. The work should move from evidence expansion to paper assembly:

- final paper drafting around the narrower claim;
- figure/table construction for the main confirmation, P0 buckets, P1 controls,
  heterophily boundary, and GSL positioning;
- final reviewer-risk audit for confidence control, CiteSeer weakness, runtime,
  and GSL proxy limitations.

## Heterophily Boundary

- `2026-06-04-stability-heterophily-regime` completed 840 runs over Texas,
  Wisconsin, Actor, four regimes, and 10 seeds. The broader heterophily claim is
  not supported.
- Feature-only is the best heterophily method (`0.4975 ± 0.1557`). StabilityResidual
  loses by `-1.14 pp` overall (`p=0.0133`, win rate `0.31`), and every other
  tested method also loses to Feature-only.
- StabilityResidual loses on all three datasets: Texas `-1.35 pp`, Wisconsin
  `-1.27 pp`, Actor `-0.80 pp`. The heterophily FSCC slice is especially
  negative: `-2.89 pp` (`p=0.0035`, win rate `0.17`).
- Diagnostics explain the boundary: edge homophily is very low (`0.087` Texas,
  `0.192` Wisconsin, `0.218` Actor), feature similarity is low (`0.17-0.35`),
  feature risk is already informative on Texas/Wisconsin, and residual AUC is
  near random (`~0.52`).
- Paper-facing consequence: do not claim universal graph evolution. The
  supported claim is homophilic, feature-ambiguous citation regimes; heterophily
  should be reported as a failure mode and motivation for future regime
  detection/fallback.

## GSL Baseline Audit

- `2026-06-04-stability-gsl-baseline-audit` implemented runnable GSL-inspired
  proxies for IDGL, ProGNN, and LDS, plus a feasibility analysis for full
  reproductions. These are proxies, not exact reproductions of the published
  methods.
- StabilityResidual remains supported vs Feature-only on the audit matrix:
  `+1.91 pp` (`p=0.0003`, win rate `0.77`, Cohen's d `0.77`).
- LDS-Proxy is the best proxy overall: `0.6383` mean accuracy, `+2.76 pp` vs
  Feature-only. It beats StabilityResidual by `+0.85 pp` (`p=0.040`, win rate
  `0.63`).
- This LDS advantage is concentrated on Cora, where Random-Matched also gains
  `+4.55 pp` over Feature-only. On CiteSeer and PubMed, LDS-Proxy and
  StabilityResidual are statistically tied.
- Paper-facing consequence: do not claim StabilityResidual beats GSL baselines.
  Claim it is competitive with GSL-inspired proxies and clearly mark full
  LDS/IDGL/ProGNN reproductions as camera-ready risk or future work.

## P0/P1 Evidence

- `2026-06-05-ambiguity-stability-evidence` completes the two remaining
  reviewer-critical checks. It ran 2160 rows over Cora/CiteSeer/PubMed,
  FSCC/LFS/DAR, 20 seeds, and P0/P1 method variants.
- P0 defines Low/Medium/High ambiguity buckets using feature-derived signals
  only, centered on distance to the Feature-only pruning decision boundary.
  Labels and `bad_edge_mask` are used only after bucket assignment for AUC,
  precision/recall/F1, pruning overlap, and changed-prune diagnostics.
- On FSCC, StabilityResidual beats Feature-only by `+2.06 pp`
  (`p=6.68e-10`, Wilcoxon `p=1.19e-8`, win rate `0.85`, Cohen's d `0.95`).
  High-only residual activation gives `+1.68 pp` (`p<1e-5`, win rate `0.80`)
  and explains `81.4%` of the full gain. Medium-only is weak (`+0.55 pp`), and
  Low-only is negative (`-0.09 pp`). This supports the ambiguity-region story
  rather than a uniform perturbation story.
- Bucket diagnostics show the practical edge-quality mechanism. In the
  High-Ambiguity bucket, Feature-only has F1 `0.3425`, while StabilityResidual
  has F1 `0.4990`; SR-only changed prunes have `68.9%` bad-edge rate.
- P1 compares Feature-only, Feature+Confidence, Feature+Stability,
  Feature+Random Stability, Feature+Shuffled Stability, and Feature+Permuted
  Stability. Aligned Feature+Stability beats random by `+1.73 pp`, shuffled by
  `+1.78 pp`, and node-permuted by `+1.63 pp` with `p<1e-8` and win rates
  `0.83-0.87`. Confidence is closer (`+0.31 pp`, `p=0.198`) and should be
  reported as a related uncertainty control, but shuffled and permuted controls
  are not competitive.

## Current Camera-Ready Risk

The method now has a viable AAAI story, but the comparison claim must remain
disciplined:

```text
StabilityResidual improves over Feature-only and static/pruning baselines on
homophilic feature-ambiguous citation graphs, with gains concentrated near the
feature-derived ambiguity boundary. It is competitive with GSL-inspired proxies,
fails on heterophily, and is not proven superior to full GSL methods.
```

## Required Reporting

Every experiment must report mean, standard deviation, paired delta vs
Feature-only, effect size, p-value, win rate, runtime, and failure modes. Oracle
results are diagnostic only.
