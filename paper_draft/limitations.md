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
- First-batch diagnostics show raw edge-gate gradients are near-random for
  bad-edge detection after feature-risk control. Claims about residual dynamic
  edge signal must be restricted or replaced.
- Multi-checkpoint gradient consistency helps in the feature-similar cross-class
  search regime but degrades low-feature-similarity validation cases. A practical
  method needs explicit regime detection and fallback to Feature-only.
- Selective MCGC prevents raw MCGC degradation on low-feature-similarity and
  degree-aligned-random regimes, but its target feature-similar cross-class gain
  is only `+0.09 pp` with `p=0.575` and win rate `0.47`. This supports a
  regularization/failure-mode story, not the stronger claim that training
  dynamics provide useful residual FSCC information beyond Feature-only.
- The current selective gate chooses a median feature-similarity threshold and
  activates on about half of edges. That behavior is not yet a convincing
  feature-ambiguous regime detector.
