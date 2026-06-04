# Method Draft

Let `G = (V, E, X)` be an attributed graph with node set `V`, edge set `E`, and
node features `X`. GraGE attaches a differentiable gate `m_e in [0, 1]` to each
edge and writes `M = {m_e : e in E}`. Message passing uses the gated adjacency,
so reducing `m_e` weakens or removes edge `e`.

The ideal bilevel objective is:

```text
theta*(M) = argmin_theta L_train(theta, M)
S_e       = d L_score(theta*(M), M) / d m_e
```

where `L_train` and `L_score` are computed on disjoint train-internal support
and score splits. Validation labels, test labels, oracle labels, and
`bad_edge_mask` are not used for practical edge scoring.

## First-Order Edge-Gate Score

The first-order approximation trains a proxy model on the noisy graph, freezes
`theta`, and computes:

```text
S_e^FO = d L_score(theta, M) / d m_e | M=1
```

Positive `S_e` means reducing the gate is expected to reduce score loss, so the
edge is treated as harmful. Negative `S_e` means the edge may be protective.

## Unrolled Hypergradient Score

The unrolled approximation differentiates through `K` inner training updates:

```text
theta_{k+1}(M) = theta_k(M) - alpha d L_support(theta_k(M), M) / d theta
S_e^K          = d L_score(theta_K(M), M) / d m_e
```

This estimates the direct and indirect effect of edge gates on the model after a
small number of train-internal optimization steps.

## Hybrid Practical Score

The current practical score combines static feature risk with positive and
negative dynamic components:

```text
R_f(e) = rank(1 - cosine(x_u, x_v))
R_+(e) = rank(relu(S_e))
R_-(e) = rank(relu(-S_e))

score(e) = R_f(e) + lambda_pos R_+(e) - lambda_neg R_-(e)
```

Edges with the highest `score(e)` are pruned under a matched budget with minimum
degree protection. The first-batch confirmation focuses on
`GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`, which historically produced the strongest
feature-similar cross-class result.

