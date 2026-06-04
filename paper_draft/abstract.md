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
datasets. The current evidence is primarily negative. Raw edge-gate gradients
are weak bad-edge detectors after feature-risk control, a selective
multi-checkpoint gate does not produce a meaningful feature-similar cross-class
gain, and a 20-seed confirmation rerun finds that GraGE-Hybrid loses to
Feature-only by `-2.50 pp` while MCGC loses by `-0.72 pp`. The current paper is
therefore not AAAI-ready as a positive method paper. A viable version must either
introduce a stronger no-leak training-dynamics channel or reframe GraGE as a
diagnostic study of when differentiable edge-gate signals fail relative to
static feature similarity.
