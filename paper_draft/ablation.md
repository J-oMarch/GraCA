# Ablation Plan

## Completed StabilityResidual Ablations

`2026-06-04-stability-ablation-confirmation` completed the core ablations and a
20-seed confirmation.

- 20-seed FSCC confirmation: StabilityResidual-frozen beats Feature-only by
  `+1.59 pp` (`p<0.001`, win rate `0.83`, Cohen's d `0.70`).
- Control regimes: LFS `+0.55 pp` and DAR `+0.81 pp`; no material degradation.
- Raw vs residualized stability: raw `+1.25 pp`, residualized `+1.11 pp`,
  raw-vs-residualized difference `+0.14 pp` (`p=0.772`). Residualization is
  primarily for the feature-residual paper claim, not a large accuracy gain.
- Shuffled residual: `+0.87 pp`, competitive enough to report as a risk but
  weaker than real residual.
- Gradient confidence: no-gradient `+1.29 pp`, shuffled-gradient `+1.43 pp`,
  frozen-gradient `+1.79 pp`, real-gradient `+1.95 pp`. Gradient confidence is
  useful as an auxiliary mechanism, but prediction stability remains the primary
  signal.
- Dropout schedules: all tested schedules beat Feature-only; wider schedule
  `[0, 0.10, 0.15, 0.20, 0.30]` is strongest.
- Views: 3, 5, and 7 views all work; 3/5 are stronger than 7.

## Remaining Ablations

Required ablations:

- Feature prior only: `Feature-only = 1 - cosine(x_u, x_v)`.
- Dynamic gradient only: first-order positive gradient, negative gradient, and
  absolute gradient variants.
- Hybrid positive/negative split: separate `relu(S_e)` and `relu(-S_e)`.
- Degree preservation: minimum degree and degree-normalized scores.
- Support/score split: train-internal split ratios and stability.
- Unrolled hypergradient: compare first-order, `K=1`, `K=3`, and selected
  practical approximations.
- Shuffled or frozen dynamic signal: test whether real training-dynamics order
  matters beyond feature risk and inner-loop schedule.
- Selective regime gate: hard versus soft gate, tau quantiles, zero-gate
  fallback, dynamic active fraction, and pruned-edge overlap with active
  dynamic edges.
- Feature-clear conservation: verify that low-feature-similarity or other
  feature-clear regimes do not degrade relative to Feature-only when the gate is
  inactive or near inactive.
- Heterophily slice: report where GraGE should abstain or adapt instead of
  overclaiming universal graph cleaning.
