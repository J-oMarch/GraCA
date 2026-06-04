# Related Work Notes

## Graph Structure Learning

[Learning Discrete Structures for GNNs / LDS, ICML 2019](https://proceedings.mlr.press/v97/franceschi19a.html)
learns graph structure and GCN parameters by approximately solving a bilevel
program over edge distributions. GraGE is closest in its bilevel motivation, but
the intended distinction is narrower and more diagnostic: edge gates are used to
extract training-dynamics scores on an observed noisy graph, and the central
claim is residual signal beyond feature similarity.

[IDGL, NeurIPS 2020](https://arxiv.org/abs/2006.13009) iteratively learns graph
structure and embeddings, largely through similarity metric learning and graph
regularization. GraGE should position feature similarity as a strong baseline
and should not claim novelty from similarity-based graph reconstruction.

[Continuous bilevel GSL, IJCAI 2022](https://www.ijcai.org/proceedings/2022/424)
models continuous graph structure with bilevel optimization and Neumann-IFT
approximations. GraGE should emphasize practical first-order and unrolled
edge-gate scores, feature-residual diagnostics, and matched-budget pruning.

## Robust Graph Cleaning

[ProGNN, KDD 2020](https://arxiv.org/abs/2005.10203) jointly learns a robust GNN
and graph structure using properties such as sparsity, low rank, and feature
smoothness. The novelty risk is direct: if GraGE gains come only from feature
smoothness, the method is graph cleaning rather than training-dynamics-guided
evolution.

## Graph Sparsification And Pruning

[TEDDY, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/90d1fc07f46e31387978b88e7e057a31-Abstract-Conference.html)
is a one-shot edge sparsification framework using edge-degree statistics for
graph lottery tickets. GraGE should report matched pruning budgets and degree
effects to avoid conflating edge-gate signal with structural sparsification.

[PSGNN, SDM 2024](https://epubs.siam.org/doi/10.1137/1.9781611978032.16)
performs pruning and sprouting during sparse GNN training, using predicted label
similarity and graph sprouting. GraGE should distinguish itself by using
edge-gate gradients and no-leak train-internal score splits rather than predicted
label similarity alone.

## Edge Gradients And Explanation

[Explaining GNN Explanations with Edge Gradients, 2025](https://arxiv.org/abs/2508.01048)
connects edge gradients with explanation methods. This increases novelty risk
for raw edge-gradient scoring. GraGE needs to frame gradients as a graph
evolution signal, validate residual value beyond features, and include shuffled
or frozen-gradient controls.

## Current Novelty Risk

[Bilevel GSL Revisited, 2026](https://arxiv.org/abs/2605.07577) argues that
bilevel graph-structure gains can be partly explained by inner-loop training
dynamics rather than graph rewiring itself. GraGE should add an inner-channel or
frozen/shuffled diagnostic so improvements are not misattributed.

## Adaptive Regime-Aware Graph Learning

[Robust Graph Structure Learning under Heterophily, 2025](https://www.sciencedirect.com/science/article/pii/S0893608025000851)
and its [arXiv version](https://arxiv.org/abs/2403.03659) highlight that robust
graph learning must account for heterophilic regimes instead of assuming one
homophily-style graph prior. GraGE can absorb this as a per-edge or per-regime
gate that suppresses training-dynamics terms when feature evidence is already
reliable.

[Separation Coefficient-Guided Adaptive Graph Structure Adjustment, IJCAI 2025](https://www.ijcai.org/proceedings/2025/663)
adapts graph structure based on representation separation to mitigate
over-smoothing. The novelty risk is that "adaptive graph adjustment" alone is
not enough; GraGE's distinction must be edge-gate training dynamics and
matched-budget evidence against Feature-only.

[Powerful GCNs with Adaptive Propagation for Homophily and Heterophily](https://arxiv.org/abs/2112.13562)
adapts propagation according to homophily or heterophily between node pairs.
This supports the next GraGE direction: a selective dynamics gate should act at
edge level and choose between feature-only pruning and MCGC-style dynamic
calibration.

[Curriculum Graph Sparsification, KDD 2024](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2024_Towards%20Lightweight%20Graph%20Neural%20Network%20Search%20with%20Curriculum%20Graph%20Sparsification.pdf)
uses curriculum sparsification for lightweight GNN search. GraGE should avoid
claiming generic sparsification novelty and instead report whether the dynamic
gate changes which edges are pruned under the same budget.
