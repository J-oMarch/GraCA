# Ablation Plan

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
- Heterophily slice: report where GraGE should abstain or adapt instead of
  overclaiming universal graph cleaning.

