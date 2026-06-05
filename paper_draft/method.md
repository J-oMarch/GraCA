# Method Draft

Let `G = (V, E, X)` be an attributed graph with node set `V`, edge set `E`, and
node features `X`. The goal is matched-budget edge pruning in a no-leak setting:
training labels may be used to train probe models, but validation labels, test
labels, oracle labels, and `bad_edge_mask` are not used for practical edge
scoring.

## Feature Prior

The required baseline is Feature-only pruning. For edge `e=(u,v)`, define

```text
R_f(e) = R(1 - cosine(x_u, x_v)),
```

where `R` is rank normalization. Edges with high `R_f` are considered risky
under static feature similarity and are pruned under the same budget and
minimum-degree protection used by all practical methods.

## Prediction-Stability Signal

StabilityResidual-GraGE trains multiple stochastic graph views using training
labels only. For view `k`, let

```text
p_k(i) = softmax(f_{theta_k}(i; G_k, X))
```

be the predicted class distribution for node `i`. The method computes node
instability from entropy, Jensen-Shannon divergence across views, class
probability variance, and inverse confidence:

```text
U_i = 0.3 R(mean_k H(p_k(i)))
    + 0.3 R(JSD(p_1(i), ..., p_K(i)))
    + 0.2 R(mean_c Var_k p_k(i,c))
    + 0.2 R(1 - mean_k max_c p_k(i,c)).
```

For edge `e=(u,v)`, node instability is converted to an edge-level stability
score:

```text
T_e = (|U_u - U_v| + U_u U_v) (1 + sim_norm(x_u, x_v)).
```

The first factor identifies edges whose endpoints have mismatched or jointly
unstable predictions. The feature-similarity factor emphasizes cases where
static features alone may not clearly separate good and bad edges, but feature
similarity is not used to define success labels.

## Stability Residual

To isolate the part of prediction stability that is not explained by the
Feature-only ranking, the ranked stability score is residualized against ranked
feature risk:

```text
R_T(e) = R(T_e)
beta   = <R_T, R_f> / <R_f, R_f>
Z_e    = R(R_T(e) - beta R_f(e)).
```

The practical score is

```text
score(e) = R_f(e) + alpha Z_e.
```

The selected candidate uses five graph views with dropout schedule
`[0, 0.10, 0.15, 0.20, 0.30]`, residualized stability, and a frozen
edge-gradient confidence control:

```text
StabilityResidual-v5-dp0.15-grad-frozen.
```

## Auxiliary Edge-Gate Confidence

GraGE originally introduced differentiable edge gates `m_e in [0,1]` and local
edge-gate sensitivities:

```text
theta*(M) = argmin_theta L_train(theta, M)
S_e       = d L_score(theta*(M), M) / d m_e.
```

Empirically, raw edge-gate gradients, GraGE-Hybrid, MCGC, and Selective-MCGC are
not strong enough to serve as the main method. Their role in this paper is
historical and auxiliary: edge-gate gradients motivate local sensitivity and can
serve as a confidence or abstention signal, but prediction stability is the
supported training-dynamics channel.

When confidence abstention is active, the method falls back to Feature-only for
low-confidence edges:

```text
score(e) = R_f(e)             if C_e < tau
score(e) = R_f(e) + alpha Z_e otherwise.
```

## Diagnostics and Controls

The paper-facing diagnostics separate three questions:

- Feature ambiguity: are improvements concentrated near feature-derived pruning
  decision boundaries?
- Stability alignment: does aligned stability beat confidence, random,
  shuffled, and node-permuted controls?
- Regime boundary: does the method fail on heterophily, where Feature-only is a
  safer baseline?

Labels and injected bad-edge masks are used only after scoring for diagnostics
such as AUC, precision, recall, F1, and pruning-overlap analysis.
