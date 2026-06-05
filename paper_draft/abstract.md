# Abstract Draft

Feature-similarity pruning is a strong baseline for noisy homophilic citation
graphs, but it becomes ambiguous near the feature-derived pruning boundary:
static features alone may weakly discriminate likely good and bad edges. We
study whether prediction stability under training-time graph perturbations
provides complementary edge-quality evidence in this regime. StabilityResidual-
GraGE trains multiple stochastic graph views using training labels only,
converts node prediction instability into edge scores, residualizes the signal
against Feature-only risk, and combines the residual with the feature prior
under a matched pruning budget. Edge-gate gradients are retained as local
sensitivity motivation and optional confidence/abstention, not as the main
empirical signal.

The current evidence supports a narrow claim. In a 20-seed confirmation across
Cora, CiteSeer, and PubMed with feature-similar cross-class noise,
`StabilityResidual-v5-dp0.15-grad-frozen` improves over Feature-only by
`+1.59 pp` (`p<0.001`, win rate `0.83`, Cohen's d `0.70`) without material
degradation on low-feature-similarity or degree-aligned-random controls.
A paper-facing ambiguity analysis further improves over Feature-only by
`+2.06 pp` on the same FSCC matrix, with `81.4%` of the full gain reproduced by
activating the residual only in the High-Ambiguity bucket. StabilityResidual-only
changed prunes in that bucket have `68.9%` bad-edge rate. Alignment controls
show that aligned stability beats random, shuffled, and node-permuted stability
by `+1.63` to `+1.78 pp`; confidence is closer but still below aligned
stability. Negative results are explicit: raw edge-gate gradient routes,
GraGE-Hybrid, MCGC, and Selective-MCGC are not viable main methods; heterophily
experiments on Texas, Wisconsin, and Actor fail; and an LDS-inspired proxy beats
StabilityResidual in the current GSL audit. The paper therefore positions
prediction stability as complementary edge evidence for homophilic,
feature-ambiguous citation regimes, not as universal graph structure learning or
GSL superiority.
