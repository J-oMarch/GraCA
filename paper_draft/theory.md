# Theory Draft

## Setup

For graph `G = (V, E, X)`, let `M in [0, 1]^|E|` be edge gates and let
`A(M)` be the gated adjacency used by a differentiable GNN `f_theta`. GraGE
defines:

```text
theta*(M) = argmin_theta L_train(theta, M)
S_e(M)    = d L_score(theta*(M), M) / d m_e
```

The practical algorithm uses either a first-order approximation with fixed
`theta` or an unrolled approximation through `K` support-gradient steps.

## Proposition 1: First-Order Gate Pruning Direction

Assume `L_score(theta, M)` is differentiable in `m_e` at `M = 1`. For a small
gate reduction `epsilon > 0` on edge `e`, holding `theta` fixed:

```text
L_score(theta, M - epsilon e_e)
= L_score(theta, M) - epsilon S_e^FO + O(epsilon^2)
```

Therefore, if `S_e^FO > 0`, decreasing the gate reduces the score loss to first
order; if `S_e^FO < 0`, decreasing the gate increases the score loss to first
order.

Proof sketch: Apply the first-order Taylor expansion of `L_score` with respect
to the coordinate `m_e`.

## Proposition 2: Unrolled Gate Score Approximates Bilevel Sensitivity

Let `theta_K(M)` be produced by `K` differentiable support-gradient updates. If
the update map is differentiable in `(theta, M)`, then:

```text
d L_score(theta_K(M), M) / d m_e
= partial L_score / partial m_e
  + (partial L_score / partial theta_K)
    (d theta_K / d m_e)
```

This is the exact hypergradient for the truncated inner problem and a
finite-step approximation to the ideal bilevel sensitivity as `theta_K(M)`
approaches `theta*(M)`.

Proof sketch: Repeatedly apply the chain rule through the differentiable inner
updates. The approximation error depends on optimization error after `K` steps
and on smoothness of the inner objective.

## Paper Claim To Validate

The theory only justifies edge-gate gradients as local sensitivity signals. The
empirical paper claim requires showing that these sensitivities provide residual
information beyond feature similarity and improve downstream accuracy under
matched pruning budgets.

