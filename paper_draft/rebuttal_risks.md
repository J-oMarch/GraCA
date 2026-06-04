# Rebuttal Risks

High-risk reviewer questions:

- Is GraGE just feature-similarity pruning with extra noise?
- Does the method beat Feature-only under matched pruning budgets and many
  seeds?
- Are oracle labels, validation labels, test labels, or `bad_edge_mask` leaked
  into edge scoring?
- Are improvements statistically meaningful, or are they cherry-picked from a
  small seed sweep?
- Do edge-gate gradients provide residual information after controlling for
  feature cosine?
- First-batch answer: not for bad-edge detection. Full diagnostics found raw
  gradient AUC near random and real-vs-shuffled hybrid deltas around `0.003`.
- Are gains caused by the inner training schedule rather than graph evolution?
- Does the method work outside homophilic citation graphs?
- Why should raw edge gradients be considered novel given GNN explanation work?

Evidence needed:

- Paired statistics, win rates, effect sizes, and confidence intervals vs
  Feature-only.
- Feature-bin and residual diagnostics.
- Shuffled/frozen dynamic controls.
- Heterophily failure analysis.
- Clear theorem-style local sensitivity claim with proof sketch.
- A new selective-dynamics experiment showing that MCGC-style signal is used
  only in feature-ambiguous regimes and falls back to Feature-only elsewhere.

Current answer after `2026-06-04-selective-mcgc-regime-gate`:

- The selective gate does not yet answer the strongest reviewer concern. FSCC
  improvement is `+0.09 pp`, not significant (`p=0.575`), with win rate `0.47`.
- The gate does answer a narrower failure-mode question: raw MCGC degrades LFS
  by `-2.46 pp`, while selective MCGC improves LFS by `+1.90 pp`. This supports
  "gating prevents dynamic-gradient noise from hurting" more than "training
  dynamics add residual edge information."
- A reviewer can still argue that GraGE is Feature-only plus an expensive
  regularizer unless the FSCC confirmation rerun finds a stronger multi-seed
  delta over Feature-only.
