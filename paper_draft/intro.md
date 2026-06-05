# Introduction Draft

Graph neural networks rely on an observed graph whose edges are often noisy,
incomplete, or misaligned with the downstream task. In homophilic citation
graphs, a strong and natural response is to prune edges by static feature
similarity: edges connecting dissimilar feature vectors are often suspicious.
This baseline is not a strawman. In the current evidence, Feature-only pruning is
strong, and several earlier GraGE variants fail to beat it under matched pruning
budgets.

This paper asks a narrower question: can training-dynamics-derived prediction
stability provide useful edge-quality evidence beyond static feature similarity?
The question is most relevant in feature-ambiguous regimes, where feature-based
scores are close to the pruning decision boundary and therefore provide weak
discrimination between likely good and likely bad edges.

StabilityResidual-GraGE answers this question by training multiple stochastic
graph views, measuring node prediction instability across those views, converting
that instability into edge scores, and residualizing the resulting score against
Feature-only risk. The practical edge score combines a feature prior with this
stability residual. Edge-gate gradients are retained only as local sensitivity
motivation and optional confidence/abstention; they are not the primary
empirical signal.

The supported empirical claim is deliberately limited. On Cora, CiteSeer, and
PubMed with feature-similar cross-class noise, the selected
`StabilityResidual-v5-dp0.15-grad-frozen` candidate improves over Feature-only by
`+1.59 pp` in a 20-seed confirmation (`p<0.001`, win rate `0.83`, Cohen's
`d≈0.70`) without material degradation on low-feature-similarity or
degree-aligned-random controls. The P0/P1 evidence strengthens the mechanism:
in the FSCC matrix, the full StabilityResidual score improves over Feature-only
by `+2.06 pp` (`p=6.68e-10`, win rate `0.85`), and a High-Ambiguity-only
residual recovers `81.4%` of that gain. Aligned stability also beats random,
shuffled, and node-permuted stability controls by at least `+1.63 pp`, indicating
that the node-to-stability alignment matters rather than merely the residual
score distribution.

On Texas, Wisconsin, and Actor, the method loses to Feature-only, so heterophily
is treated as a failure boundary rather than a success case. Against
GSL-inspired proxies, the method is competitive but not superior; LDS-Proxy is
stronger overall in the current audit.
