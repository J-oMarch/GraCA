# Related Work Notes

## Graph Structure Learning

[Learning Discrete Structures for GNNs / LDS, ICML 2019](https://proceedings.mlr.press/v97/franceschi19a.html)
learns graph structure and GCN parameters by approximately solving a bilevel
program over edge distributions. GraGE is closest in its bilevel motivation, but
the intended distinction is narrower and more diagnostic: prediction stability
is used to score edges on an observed noisy graph, and the central claim is
complementary evidence beyond feature similarity in feature-ambiguous homophilic
regimes.

[IDGL, NeurIPS 2020](https://arxiv.org/abs/2006.13009) iteratively learns graph
structure and embeddings, largely through similarity metric learning and graph
regularization. GraGE should position feature similarity as a strong baseline
and should not claim novelty from similarity-based graph reconstruction.

[Continuous bilevel GSL, IJCAI 2022](https://www.ijcai.org/proceedings/2022/424)
models continuous graph structure with bilevel optimization and Neumann-IFT
approximations. GraGE should emphasize matched-budget pruning, feature-residual
diagnostics, and the fact that edge-gate sensitivities are auxiliary rather than
the main empirical signal.

## Robust Graph Cleaning

[ProGNN, KDD 2020](https://arxiv.org/abs/2005.10203) jointly learns a robust GNN
and graph structure using properties such as sparsity, low rank, and feature
smoothness. The novelty risk is direct: if GraGE gains come only from feature
smoothness or budget effects, the method is graph cleaning rather than
training-dynamics-guided edge disambiguation.

## Graph Sparsification And Pruning

[TEDDY, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html)
is a one-shot edge sparsification framework using edge-degree statistics for
graph lottery tickets. GraGE should report matched pruning budgets and degree
effects to avoid conflating edge-gate signal with structural sparsification.

[PSGNN, SDM 2024](https://epubs.siam.org/doi/10.1137/1.9781611978032.16)
performs pruning and sprouting during sparse GNN training, using predicted label
similarity and graph sprouting. GraGE should distinguish itself by using
prediction-stability residuals under no-leak train-internal scoring rather than
predicted label similarity alone.

## Edge Gradients And Explanation

[Explaining GNN Explanations with Edge Gradients, 2025](https://arxiv.org/abs/2508.01048)
connects edge gradients with explanation methods. This increases novelty risk
for raw edge-gradient scoring. GraGE should state that raw gradients failed as
the main empirical signal and are retained only as local sensitivity and
confidence/abstention motivation.

## Current Novelty Risk

[Bilevel GSL Revisited, 2026](https://arxiv.org/abs/2605.07577) argues that
bilevel graph-structure gains can be partly explained by inner-loop training
dynamics rather than graph rewiring itself. GraGE answers this risk with P1
controls: aligned stability is compared against confidence, random, shuffled,
and node-permuted stability, so improvements are not attributed to alignment
without an alignment-destruction test.

## Adaptive Regime-Aware Graph Learning

[Robust Graph Structure Learning under Heterophily, 2025](https://www.sciencedirect.com/science/article/pii/S0893608025000851)
and its [arXiv version](https://arxiv.org/abs/2403.03659) highlight that robust
graph learning must account for heterophilic regimes instead of assuming one
homophily-style graph prior. GraGE should report heterophily as a failure
boundary and future fallback/regime-detection problem, not as a solved setting.

[Separation Coefficient-Guided Adaptive Graph Structure Adjustment, IJCAI 2025](https://www.ijcai.org/proceedings/2025/663)
adapts graph structure based on representation separation to mitigate
over-smoothing. The novelty risk is that "adaptive graph adjustment" alone is
not enough; GraGE's distinction must be edge-gate training dynamics and
matched-budget evidence against Feature-only.

[Powerful GCNs with Adaptive Propagation for Homophily and Heterophily](https://arxiv.org/abs/2112.13562)
adapts propagation according to homophily or heterophily between node pairs.
This supports a cautious regime-limited framing: methods that do not detect
heterophily should avoid claiming uniform graph rewiring success.

[Revisiting the Role of Heterophily in Graph Representation Learning: An Edge
Classification Perspective](https://arxiv.org/abs/2205.11322) frames
heterophily handling as an edge classification problem, either avoiding messages
through heterophilous edges or using heterophilous neighbors differently. This is
a novelty risk for any GraGE story that sounds like "classify edges as
homophilic or heterophilic." The required distinction is that StabilityResidual
does not train an edge-label classifier; it uses prediction stability as a
no-leak residual score and evaluates matched-budget downstream effects.

[GREET / AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/download/25573/25345)
learns representations by discriminating and leveraging homophilic and
heterophilic edge views in an unsupervised setting. GraGE can cite this as
evidence that edge-regime separation is important, while emphasizing that the
current claim does not cover heterophily and does not introduce a standalone
edge discriminator.

[Curriculum Graph Sparsification, KDD 2024](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2024_Towards%20Lightweight%20Graph%20Neural%20Network%20Search%20with%20Curriculum%20Graph%20Sparsification.pdf)
uses curriculum sparsification for lightweight GNN search. GraGE should avoid
claiming generic sparsification novelty and instead report whether the dynamic
gate changes which edges are pruned under the same budget.

## Stability And Self-Supervised Structure Signals

[Regularizing GNNs via Consistency-Diversity Graph Augmentations, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20307)
argues that graph augmentations should be evaluated by both consistency and
diversity. For GraGE, this suggests a stronger no-leak channel than raw
edge-gradient signs: score edges by whether they destabilize predictions across
graph views, then use edge-gate gradients only as optional
confidence/abstention.

[SLAPS, 2021](https://openreview.net/forum?id=JWRRBHFPKTJ) uses
self-supervision to improve adjacency learning. The novelty risk is that a
prediction-stability GraGE rebuild may look like self-supervised structure
learning unless it explicitly keeps the observed noisy graph, reports
matched-budget pruning against Feature-only, and isolates residual evidence
beyond feature cosine.

[Self-Supervised Graph Structure Refinement, 2022](https://arxiv.org/abs/2211.06545)
refines graph structure using multi-view contrastive pretraining and estimated
edge probabilities. GraGE can absorb the multi-view idea, but must keep the
edge-disambiguation framing: stability is a train-dynamics-derived residual edge
signal, not a replacement by link prediction.

[On the Prediction Instability of Graph Neural Networks](https://www.catalyzex.com/paper/on-the-prediction-instability-of-graph-neural)
studies how node predictions vary with training randomness and graph/model
choices. This supports using prediction instability as a measurable training
dynamics object, but GraGE must convert node-level instability into edge-level
scores and test confidence, random, shuffled, and node-permuted controls.
