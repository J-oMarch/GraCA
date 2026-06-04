# Abstract Draft

Graph neural networks rely on an input graph whose edges are often noisy,
incomplete, or misaligned with the downstream task. Existing graph cleaning and
structure learning methods frequently rely on static feature similarity or
topological priors, which are strong but can fail when harmful edges connect
feature-similar nodes. We study whether training dynamics expose edge-level
signals that are not captured by static similarity. GraGE treats graph structure
as an evolvable object and scores edges using train-internal dynamics. Early
edge-gate gradient variants were negative: raw gradients are weak bad-edge
detectors after feature-risk control, and a 20-seed confirmation rerun finds
that GraGE-Hybrid loses to Feature-only by `-2.50 pp`. We therefore rebuild the
dynamic channel around prediction stability under graph perturbations.
StabilityResidual-GraGE trains multiple stochastic graph views, converts
prediction instability into edge scores, residualizes the signal against feature
cosine, and uses edge-gate gradient confidence only as an abstention/regularizing
constraint. In a 10-seed validation across Cora, CiteSeer, and PubMed,
StabilityResidual-GraGE beats Feature-only by `+2.00 pp` on
feature-similar cross-class noise (`p=0.0001`, win rate `0.87`) with no material
degradation on low-feature-similarity or degree-aligned-random controls. The
current paper path is viable again, but the claim must be precise: prediction
stability is the supported residual training-dynamics signal, while raw
edge-gate gradients remain auxiliary rather than the main source of gains.
