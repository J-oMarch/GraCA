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

## Proposition 3: Selective Gate Conserves Feature-Only Ranking Off-Regime

Let the selective score be:

```text
score(e)= R_f(e) + A_e D_e
```

where `D_e` is any bounded training-dynamics term, such as the MCGC positive and
negative gradient contribution. If `A_e = 0` for all edges in a subset `B`, then
the selective score restricted to `B` has exactly the same ordering as
Feature-only:

```text
score(e) = R_f(e),  for all e in B.
```

Therefore, a hard gate that turns off dynamics in feature-clear regimes cannot
change the pruning order inside that off-regime. For a soft gate with
`0 <= A_e <= eta` on `B` and `|D_e| <= c`, the score perturbation is bounded by
`eta c`; any pairwise Feature-only margin larger than `2 eta c` is order
preserving.

Proof sketch: The hard-gate case follows by substitution. For the soft-gate
case, each edge score changes by at most `eta c`, so the pairwise score
difference changes by at most `2 eta c`.

## Paper Claim To Validate

The theory only justifies edge-gate gradients as local sensitivity signals. The
empirical paper claim requires showing that these sensitivities provide residual
information beyond feature similarity and improve downstream accuracy under
matched pruning budgets.

After the first and second batches, raw edge-gate gradients are not sufficient as
the main empirical signal. The current viable claim uses prediction stability as
the primary training-dynamics signal and edge-gate gradients only as auxiliary
confidence.

## Proposition 4: Stability Residual Removes Linear Feature-Risk Component

Let `R_T in R^|E|` be a rank-normalized prediction-stability edge score and
`R_f in R^|E|` be rank-normalized feature risk. Define:

```text
beta = <R_T, R_f> / <R_f, R_f>
Z    = R_T - beta R_f.
```

Then `Z` is orthogonal to the linear feature-risk direction:

```text
<Z, R_f> = 0.
```

Therefore, any nonzero downstream contribution of `Z` cannot be explained by
the linear component of feature-risk ranking alone. The practical method
rank-normalizes `Z` after residualization, so exact orthogonality may not be
preserved after the final rank transform, but the residualization step removes
the feature-correlated component before score combination.

Proof sketch: Substitute the definition of `Z`:

```text
<Z, R_f> = <R_T, R_f> - beta <R_f, R_f>
         = <R_T, R_f> - <R_T, R_f> = 0.
```

## Proposition 5: Confidence Abstention Bounds Damage From Weak Dynamics

Let the StabilityResidual score be:

```text
score(e) = R_f(e) + A_e alpha Z_e,
```

where `A_e in {0,1}` is a confidence gate and `|Z_e| <= 1`. If `A_e=0`, the
method exactly falls back to Feature-only on edge `e`. If `0 <= A_e <= eta` in a
soft-gated variant, the score perturbation is bounded by `eta alpha`; any pair
of edges whose Feature-only score margin exceeds `2 eta alpha` keeps the same
relative order.

Proof sketch: The hard-gate case follows by substitution. For the soft-gate
case, each edge score changes by at most `eta alpha`, so any pairwise difference
changes by at most `2 eta alpha`.

## Updated Paper Claim To Validate

The current experiments support the narrower claim that prediction stability
under graph perturbations provides an edge-level residual signal beyond feature
similarity on homophilic citation graphs. Before final paper claims, the method
still needs heterophily validation, residualization ablations, and sensitivity
analysis for dropout schedules and number of views.
