# Theory Draft

The theory supports a narrow ranking claim. It does not prove universal graph
learning, heterophily success, or optimal graph structure.

## Definition: Feature Ambiguity Region

Let `R_f(e)` be a feature-derived edge-risk score, such as ranked
`1 - cosine(x_u, x_v)`. For a matched pruning budget, let `tau_f` be the
Feature-only decision boundary or its feature-only quantile approximation. The
feature ambiguity region is

```text
A_delta = { e in E : |R_f(e) - tau_f| <= delta }.
```

This region is defined only from feature-derived quantities. Labels, stability,
bad-edge masks, oracle labels, validation/test labels, and downstream outcomes
are not used to define it.

## Definition: Stability Residual

Let `R_T(e)` be a rank-normalized prediction-stability edge score computed from
training-only stochastic graph views. The linear projection of `R_T` onto
feature risk is removed by

```text
beta = <R_T, R_f> / <R_f, R_f>
Z(e) = R_T(e) - beta R_f(e).
```

The practical method rank-normalizes this residual before combining it with the
feature prior:

```text
score(e) = R_f(e) + alpha R(Z(e)).
```

## Proposition 1: Residualization Removes the Linear Feature-Risk Component

Before the final rank transform, the stability residual is orthogonal to the
feature-risk direction:

```text
<Z, R_f> = 0.
```

Proof sketch: Substitute the definition of `Z`:

```text
<Z, R_f> = <R_T, R_f> - beta <R_f, R_f>
         = <R_T, R_f> - <R_T, R_f>
         = 0.
```

The final rank transform may not preserve exact orthogonality, but the
residualization step removes the linear feature-risk component before the score
combination.

## Proposition 2: Aligned Residuals Improve Pairwise Ranking in Ambiguity Regions

Consider two edges `e_bad` and `e_good` inside `A_delta`, where the downstream
edge-quality target ranks `e_bad` as more harmful than `e_good`. Suppose the
feature prior is ambiguous on this pair:

```text
|R_f(e_bad) - R_f(e_good)| <= gamma.
```

If the stability residual is aligned with residual edge quality by margin

```text
R(Z(e_bad)) - R(Z(e_good)) > gamma / alpha,
```

then the combined score ranks the harmful edge above the good edge:

```text
score(e_bad) > score(e_good).
```

Proof sketch: Expand the combined score difference:

```text
score(e_bad) - score(e_good)
= R_f(e_bad) - R_f(e_good)
  + alpha (R(Z(e_bad)) - R(Z(e_good))).
```

The feature term can hurt by at most `gamma` in the ambiguous region. The assumed
positive residual margin exceeds that possible feature disadvantage, so the
combined score is positive and the pairwise ranking improves.

## Proposition 3: Confidence Abstention Preserves Feature-Only Decisions Off Signal

Let

```text
score(e) = R_f(e) + A_e alpha R(Z(e)),
```

where `A_e in {0,1}` is an abstention gate. If `A_e=0`, the method exactly
falls back to Feature-only on edge `e`. For a soft gate with `0 <= A_e <= eta`
and `|R(Z(e))| <= 1`, any pairwise Feature-only margin larger than
`2 eta alpha` keeps its ordering.

Proof sketch: The hard-gate case follows by substitution. In the soft-gate
case, each edge receives a perturbation bounded by `eta alpha`, so a pairwise
score difference changes by at most `2 eta alpha`.

## Paper Claim

The propositions justify only a conditional mechanism: if prediction-stability
residuals are aligned with residual edge quality in feature-defined ambiguity
regions, adding them to Feature-only can improve edge ranking there while
abstention can preserve Feature-only behavior off signal. The empirical paper
must verify that this alignment occurs on homophilic citation regimes and fails
honestly where it does not.
