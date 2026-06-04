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

## Adaptive Candidate: Multi-Checkpoint Gradient Consistency

The adaptive search introduced MCGC, which collects edge-gate gradients at
multiple training checkpoints and weights the gradient term by sign consistency.
For checkpoint gradients `{S_e^(t)}`, define:

```text
bar S_e = mean_t S_e^(t)
C_e     = fraction_t[sign(S_e^(t)) = sign(bar S_e)]
score(e)= R_f(e) + C_e lambda_pos R(relu(bar S_e))
                - C_e lambda_neg R(relu(-bar S_e))
```

MCGC improved the feature-similar cross-class search slice but failed validation
because it degraded low-feature-similarity cases.

## Historical Candidate: Selective MCGC Regime Gate

The selective MCGC candidate adds a feature-regime gate `A_e` so dynamic terms
are suppressed when feature risk is already reliable:

```text
score(e)= R_f(e) + A_e C_e lambda_pos R(relu(bar S_e))
                - A_e C_e lambda_neg R(relu(-bar S_e))
```

For the hard gate:

```text
A_e = 1[cos(x_u, x_v) >= tau]
```

For the soft gate:

```text
A_e = sigmoid(k (cos(x_u, x_v) - tau))
```

`tau` is selected without labels, using a fixed candidate-edge feature-similarity
quantile or an unsupervised stability criterion. The intended behavior is not to
replace Feature-only pruning everywhere, but to use edge-gate dynamics only in
the feature-ambiguous region where first-batch search found MCGC gains. The
required controls are shuffled checkpoint gradients, zero-gate fallback, and
threshold sensitivity.

The confirmation result is mixed: selective MCGC avoids some degradation, but it
does not produce a meaningful FSCC gain. It is therefore an ablation/historical
candidate rather than the main method.

## Current Candidate: StabilityResidual-GraGE

The current supported candidate changes the dynamic signal from raw edge-gate
gradient ranks to prediction stability under stochastic graph perturbations.
For edge scoring, train `K` no-leak probe models or graph views with dropout
rates `{r_k}` using training labels only. For node `i`, collect prediction
distributions:

```text
P_i = {p_k(i) = softmax(f_{theta_k}(i; G_k, X)) : k = 1..K}.
```

Define node instability as a rank-normalized mixture:

```text
U_i = 0.3 R(mean_k H(p_k(i)))
    + 0.3 R(JSD(p_1(i), ..., p_K(i)))
    + 0.2 R(mean_c Var_k p_k(i,c))
    + 0.2 R(1 - mean_k max_c p_k(i,c)).
```

For edge `e=(u,v)`, the raw stability score is:

```text
T_e = (|U_u - U_v| + U_u U_v) (1 + sim_norm(x_u, x_v)).
```

To make the paper claim residual to static feature similarity, regress the
ranked stability score against ranked feature risk:

```text
R_T(e) = R(T_e)
R_f(e) = R(1 - cosine(x_u, x_v))
beta   = <R_T, R_f> / <R_f, R_f>
Z_e    = R(R_T(e) - beta R_f(e)).
```

The practical pruning score is:

```text
score(e) = R_f(e) + alpha Z_e,
```

with optional edge-gate confidence abstention:

```text
score(e) = R_f(e)                         if C_e < tau
score(e) = R_f(e) + alpha Z_e             otherwise.
```

`C_e` is derived from edge-gate gradient consistency/magnitude across
checkpoints, but validation shows the prediction-stability residual is the main
source of improvement. Gradient confidence should be described as an auxiliary
abstention or regularization mechanism, not as the primary signal.
