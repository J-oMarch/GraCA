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

