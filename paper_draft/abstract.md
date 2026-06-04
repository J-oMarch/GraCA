# Abstract Draft

Graph neural networks rely on an input graph whose edges are often noisy,
incomplete, or misaligned with the downstream task. Existing graph cleaning and
structure learning methods frequently rely on static feature similarity or
topological priors, which are strong but can fail when harmful edges connect
feature-similar nodes. We study whether training dynamics expose edge-level
signals that are not captured by static similarity. GraGE treats graph structure
as an evolvable object by attaching differentiable gates to edges and scoring
each edge with first-order or unrolled approximations to the effect of the gate
on a train-internal score loss. A hybrid score combines feature risk with
training-dynamics terms that promote harmful-edge pruning while protecting edges
whose gradients indicate useful message passing. Experiments compare GraGE
against Feature-only, similarity pruning, random matched pruning, and graph
robustness baselines under matched budgets across citation and heterophily
datasets. First-batch evidence shows that raw edge-gate gradients are weak as
bad-edge detectors, but multi-checkpoint gradient consistency can help in
feature-ambiguous regimes while hurting feature-clear regimes. The current paper
claim is therefore being revised toward selective, regime-aware graph evolution
rather than universal dynamic pruning.
