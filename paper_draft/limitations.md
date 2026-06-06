# Limitations Draft

- Feature-only pruning is a strong baseline. The paper claim depends on beating
  Feature-only under matched pruning budgets, not only random pruning or weak
  graph-cleaning baselines.
- StabilityResidual-GraGE is regime-limited. The supported evidence is on
  homophilic citation graphs with feature-ambiguous harmful edges. It should not
  be presented as universal graph learning or general graph rewiring.
- Heterophily is a confirmed failure boundary. On Texas/Wisconsin/Actor,
  StabilityResidual loses to Feature-only by `-1.14 pp` overall (`p=0.0133`,
  win rate `0.31`) and by `-2.89 pp` on the heterophily FSCC slice. The method
  should abstain or fall back in low-homophily regimes; current results do not
  support broader heterophily claims.
- GSL comparison is not a superiority result. LDS-Proxy beats StabilityResidual
  by `+0.85 pp` overall (`p=0.040`) in the GSL audit, although the advantage is
  concentrated on Cora and likely entangled with pruning budget/degree effects.
  The paper can claim competitiveness with GSL-inspired proxies, not superiority
  over full LDS/IDGL/ProGNN.
- Prediction stability is the supported training-dynamics signal. Raw
  edge-gate gradients, GraGE-Hybrid, MCGC, and Selective-MCGC are historical
  negative or auxiliary routes. Edge-gate gradients should be framed as local
  sensitivity and confidence/abstention, not as the main empirical driver.
- Residualization is theoretically useful but not empirically decisive. Raw
  stability is slightly better than residualized stability in the 5-seed
  ablation, with a nonsignificant `+0.14 pp` difference. The paper should not
  imply that residualization alone creates the accuracy gain.
- Earlier shuffled-residual ablations were somewhat competitive (`+0.87 pp`),
  but the full P1 alignment-destruction test is more favorable: aligned
  stability beats random, shuffled, and node-permuted stability by `+1.63` to
  `+1.78 pp` with `p<1e-8`. The remaining nuance is that confidence is closer
  in paired accuracy (`+0.31 pp`, `p=0.198`), so the paper should describe
  stability as related to uncertainty but empirically stronger than
  alignment-destroyed controls.
- The confidence risk audit (`2026-06-05-confidence-risk-audit`) confirms that
  stability provides signal beyond confidence. StabilityResidual AUC (0.803)
  exceeds Confidence AUC (0.798) globally. Within confidence strata, residual
  stability adds `+0.029` AUC on average, rising to `+0.032` in
  High-Ambiguity edges. The partial correlation coefficient for residual
  stability is `+0.21` after controlling for feature risk and confidence. This
  evidence reduces the reviewer risk that stability is merely confidence under
  another name, though the `+0.31 pp` paired accuracy delta remains
  nonsignificant.
- Per-dataset evidence is uneven. Cora and PubMed support the FSCC claim more
  strongly than CiteSeer; CiteSeer is positive but not individually significant
  in the 20-seed confirmation.
- Runtime is higher than Feature-only. StabilityResidual trains multiple graph
  views and is about `4x` slower in the confirmation runs. This is acceptable for
  evidence construction but should be reported.
- The practical no-leak setting cannot use validation labels, test labels,
  oracle labels, or `bad_edge_mask` for scoring. These signals are diagnostic
  only, which limits direct access to task-harmful edges.

Current remaining paper-facing risks:

- The ambiguity contribution analysis is supportive but attributional, not a
  causal intervention proof. High-only residual activation explains `81.4%` of
  the full gain, but the paper should avoid claiming that every improvement is
  exclusively caused by High-Ambiguity edges.
- Feature+Confidence is close to Feature+Stability and not significantly worse
  in the P1 paired test (`+0.31 pp`, `p=0.198`). This should be discussed as a
  related uncertainty baseline rather than ignored.
- Whether reviewers accept GSL-inspired proxy baselines without full
  LDS/IDGL/ProGNN reproductions remains a camera-ready risk.
