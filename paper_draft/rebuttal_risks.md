# Rebuttal Risks

## High-Risk Reviewer Questions

- Is StabilityResidual just Feature-only pruning with expensive score
  diversification?
- Are gains concentrated in feature-defined ambiguity regions, or do they come
  from random perturbation of the pruning set?
- Does aligned prediction stability beat confidence, random stability, shuffled
  stability, and node-permuted stability controls?
- Are oracle labels, validation labels, test labels, or `bad_edge_mask` leaked
  into edge scoring or ambiguity-bucket definitions?
- Does the method beat Feature-only under matched pruning budgets and many
  seeds?
- Does the method work outside homophilic citation graphs?
- How should the method be positioned against LDS, IDGL, ProGNN, and broader
  graph structure learning?
- Is the successful signal prediction stability rather than edge-gate gradients?

## Current Evidence

- Matched-budget 20-seed confirmation supports the homophilic citation FSCC
  claim: StabilityResidual-frozen beats Feature-only by `+1.59 pp`
  (`p<0.001`, win rate `0.83`, Cohen's d `0.70`), with no material LFS/DAR
  degradation.
- Feature-only remains a strong baseline and must be foregrounded, not
  downplayed.
- Raw edge-gate gradients, GraGE-Hybrid, MCGC, and Selective-MCGC are not viable
  main methods. They remain historical negative evidence or auxiliary
  local-sensitivity analysis.
- Prediction stability is the supported training-dynamics channel. Gradient
  confidence has an auxiliary role, but no-gradient and frozen-gradient controls
  show it is not the dominant source of improvement.
- Heterophily is a confirmed failure boundary. StabilityResidual loses to
  Feature-only on Texas/Wisconsin/Actor by `-1.14 pp` overall and by `-2.89 pp`
  on heterophily FSCC.
- GSL positioning is competitive, not superior. LDS-Proxy beats
  StabilityResidual by `+0.85 pp` overall in the current proxy audit.
- P0 ambiguity evidence is supportive. On FSCC, High-only residual activation
  recovers `81.4%` of the full StabilityResidual gain, while Medium-only is weak
  and Low-only is negative. SR-only changed prunes in the High-Ambiguity bucket
  have `68.9%` bad-edge rate.
- P1 alignment evidence is supportive. Aligned stability beats random,
  shuffled, and node-permuted stability by `+1.63` to `+1.78 pp` with
  `p<1e-8`; shuffled and permuted controls are not competitive in the full
  matrix.

## Current Open Risks

- P0 is supportive but still attributional. The paper can say the gain is
  concentrated in feature-defined ambiguity regions, but should not overstate
  this as a complete causal decomposition.
- Feature+Confidence is close to aligned stability (`+0.31 pp` lower,
  `p=0.198`). A reviewer may argue that the signal is uncertainty-like. The
  response is that aligned stability decisively beats distribution-preserving
  alignment destruction controls, while confidence remains a strong related
  ablation.
- CiteSeer is positive but weak individually, so per-dataset claims must avoid
  implying uniform significance.
- Full LDS/IDGL/ProGNN are not reproduced. The paper must say
  "GSL-inspired proxies" unless full official baselines are added later.

## Rebuttal Stance

- If asked whether this is general graph structure learning: no. The paper is
  about training-dynamics-guided edge disambiguation in homophilic,
  feature-ambiguous citation regimes.
- If asked whether it beats GSL: no superiority claim. It improves over
  Feature-only and is competitive with GSL-inspired proxies, while LDS-Proxy is
  stronger in the current audit.
- If asked about heterophily: heterophily is a reported failure boundary, not a
  hidden negative result.
- If asked about edge-gate gradients: raw gradients failed as a main signal;
  they motivate local sensitivity and confidence/abstention only.
- For P0/P1: claim aligned prediction stability provides complementary
  edge-quality evidence specifically in feature-defined ambiguity regions, while
  noting that confidence is a close uncertainty control.
- For any future negative extension: record the failure in limitations and
  narrow the claim rather than launching a new method search.
