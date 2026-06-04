# Limitations Draft

- Feature-only pruning is a strong baseline. If GraGE does not consistently beat
  it, the current paper claim is unsupported.
- Edge-gate gradients are local sensitivity signals and can be noisy under
  unstable proxy training.
- The practical no-leak setting cannot use validation labels, test labels,
  oracle labels, or `bad_edge_mask` for scoring, which limits direct access to
  task-harmful edges.
- Citation datasets may favor feature-similarity methods; heterophily datasets
  may require adaptive graph-regime detection rather than uniform pruning.
- Runtime for unrolled hypergradients may be too high for the main method unless
  the benefit is clearly larger than the first-order hybrid.

