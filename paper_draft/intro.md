# Introduction Draft

Graph structure is usually treated as fixed input to a GNN, but in practice the
observed graph may include task-harmful edges and miss task-useful ones. A common
response is to remove edges between dissimilar node features or to impose
homophily-style structural priors. Those baselines are strong in citation graphs
and must be treated as serious competitors, not strawmen.

GraGE asks a narrower question: can the training trajectory of a GNN reveal
edge-level information beyond static feature similarity? This matters most in
feature-ambiguous regimes, where harmful cross-class edges are feature-similar
and therefore hard to prune with cosine or Jaccard similarity alone.

The working contribution is a graph evolution framework with differentiable edge
gates. Each edge has a gate `m_e in [0, 1]`; a proxy model is trained on a
train-internal support split; and edge scores are derived from the effect of each
gate on a train-internal score loss. The practical method combines feature risk
with positive and negative edge-gradient components, so it can both prune
harmful edges and protect edges that appear beneficial under training dynamics.

The paper must prove three empirical points before the claim is defensible:

- GraGE-Hybrid beats Feature-only under matched pruning budgets, not only
  Random-Matched or DropEdge.
- Dynamic edge-gate signals retain residual detection power after controlling
  for feature cosine.
- Gains are stable across seeds and have explicit failure-mode analysis.

