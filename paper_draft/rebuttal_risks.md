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

Current answer after `2026-06-04-fscc-confirmation-rerun`:

- The strongest reviewer concern is confirmed, not resolved. In the direct
  matched-budget FSCC rerun, Feature-only is the strongest practical method
  (`0.6116 ± 0.0496`). GraGE-Hybrid loses by `-2.50 pp` (`p=0.0012`, win rate
  `0.10`, Cohen's d `-1.40`), and MCGC loses by `-0.72 pp` (`p=0.143`, win rate
  `0.43`).
- The Cora-only MCGC gain is not enough for the paper claim because
  Random-Matched and DegreeAwareRandom gain more on the same slice. A reviewer
  can reasonably attribute this to pruning budget or degree effects.
- Control regimes and heterophily data do not rescue the method. Feature-only
  and GCN-Jaccard tie on controls, while GraGE variants lose; Feature-only also
  wins the heterophily slice.
- The paper cannot claim that current edge-gate training dynamics provide useful
  graph evolution information beyond static feature similarity. It must either
  present a new mechanism with substantially different evidence or reframe the
  contribution as a diagnostic/falsification study.

Current answer after `2026-06-04-stability-channel-rebuild`:

- The positive paper path is reopened, but with a different mechanism.
  StabilityResidual-GraGE beats Feature-only on FSCC by `+2.00 pp` (`p=0.0001`,
  win rate `0.87`) across Cora/CiteSeer/PubMed with no material LFS/DAR
  degradation.
- The strongest rebuttal risk changes from "GraGE loses to Feature-only" to
  "the successful method is prediction-stability graph augmentation, not
  edge-gate gradients." This is a fair concern: no-gradient variants are close,
  and the selected candidate uses a frozen-gradient control. The paper should
  state that prediction stability is the main training-dynamics signal, with
  edge-gate gradients serving as auxiliary confidence/abstention.
- The residual evidence is paper-facing: projection ratio `<0.005`,
  residual-feature-similarity correlation `<0.01`, and residual AUC around
  `0.65`. This directly answers the feature-only collapse risk better than the
  earlier hybrid and MCGC experiments.
- Remaining reviewer demands before final submission: heterophily validation,
  raw-vs-residual stability ablation, dropout schedule/number-of-views
  sensitivity, comparison to LDS/IDGL/ProGNN-style GSL baselines, and a clear
  explanation of why validation early stopping is not used for edge scoring.

Current answer after `2026-06-04-stability-ablation-confirmation`:

- The homophilic citation claim now has stronger evidence: 20-seed FSCC
  confirmation gives `+1.59 pp` over Feature-only (`p<0.001`, win rate `0.83`,
  Cohen's d `0.70`), with LFS `+0.55 pp` and DAR `+0.81 pp`.
- StabilityResidual is the only method in the confirm20 matrix that beats
  Feature-only on FSCC. GCN-Jaccard nearly ties Feature-only overall but does
  not beat it; MCGC and GraGE-Hybrid remain negative.
- Residualization ablation is nuanced. Raw stability and residualized stability
  are nearly tied, so residualization should be motivated as a diagnostic and
  theoretical device for the "beyond feature similarity" claim, not as a large
  performance improvement.
- Gradient confidence has a defensible auxiliary role: real gradient is best,
  frozen is close, shuffled and no-gradient are weaker but still positive. This
  supports the wording "edge-gate confidence regularizes prediction stability",
  not "edge gradients alone solve graph evolution."
- Remaining high-risk gaps are heterophily behavior and direct comparison to
  graph structure learning baselines.

Current answer after `2026-06-04-stability-heterophily-regime`:

- Heterophily behavior is now known and negative. StabilityResidual loses to
  Feature-only by `-1.14 pp` overall (`p=0.0133`, win rate `0.31`) on
  Texas/Wisconsin/Actor, and `-2.89 pp` on heterophily FSCC.
- This does not invalidate the homophilic citation result, but it blocks any
  universal graph-evolution claim. The paper must explicitly state the regime:
  homophilic citation graphs with feature-ambiguous harmful edges.
- A likely reviewer objection is "the method fails exactly where heterophily
  matters." The response is to position heterophily as a boundary condition and
  future work for regime detection/fallback, not to overclaim robustness.
- Feature-only being best on heterophily reinforces the paper's central honesty:
  static feature similarity is a strong baseline and should be used when the
  graph regime is feature-clear or heterophilic.
