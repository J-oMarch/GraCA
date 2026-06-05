# Failure Analysis: GSL Baseline Audit

## Reproducibility Risks

### GSL Proxy Simplification

The implemented GSL proxies (IDGL-Proxy, ProGNN-Proxy, LDS-Proxy) are simplified
versions of the original methods. Key simplifications:

1. **IDGL-Proxy**: Uses single-round k-NN graph construction from GCN embeddings
   instead of IDGL's iterative similarity metric learning with graph
   regularization. The proxy does not learn a parametric similarity function.

2. **ProGNN-Proxy**: Uses feature smoothness + low-rank SVD refinement instead
   of ProGNN's alternating optimization with sparsity, symmetry, and low-rank
   constraints. The proxy does not jointly train the GNN and graph.

3. **LDS-Proxy**: Uses first-order edge weight updates with a proxy loss instead
   of LDS's bilevel optimization with implicit function theorem gradients. The
   proxy uses a heuristic target (same-prediction + feature similarity) rather
   than true validation-loss gradients through the graph.

**Risk**: If the simplified proxies perform worse than the real methods, the
experiment may falsely conclude that StabilityResidual beats GSL baselines.
Conversely, if a simplified proxy accidentally captures the key mechanism, it
may serve as a valid lower bound on real GSL performance.

**Mitigation**: Report the proxies as "IDGL-inspired" / "ProGNN-inspired" /
"LDS-inspired" baselines, not as reproductions. The paper should state that
full GSL reproduction is deferred and link to original implementations.

### Device and Numerical Risks

All experiments run on CUDA with float32 precision. Stochastic components
(edge dropout, random pruning) are seeded for reproducibility. The GSL proxies
use SVD (ProGNN) and k-NN construction (IDGL) which are deterministic given
the same inputs.

## Leakage Risks

### Training Labels

All methods use only `train_mask` labels for model training. No test or
validation labels leak into edge scoring or graph construction. This is
consistent with the main StabilityResidual method.

### Validation Labels

Validation labels are used in two places:
1. **Early stopping**: Standard model selection, used by all methods equally.
2. **LDS-Proxy outer loop**: The proxy uses validation loss to update edge
   weights. This is analogous to validation-based early stopping — the
   validation set is used for model selection, not for edge oracle scoring.

**Risk**: A reviewer could argue that LDS-Proxy uses validation labels more
directly than other methods. The response is that (a) the proxy loss uses
a heuristic target, not direct validation gradients, and (b) the validation
set's role is model selection, consistent with all other methods.

### Edge Oracle Scoring

No method uses `bad_edge_mask` for training or graph construction. The
`bad_edge_mask` appears only in diagnostic evaluation code (AUC computation),
which is clearly marked as evaluation-only.

## Dependency Risks

### No External GSL Libraries Required

The experiment uses only the existing codebase dependencies:
- PyTorch + CUDA
- PyTorch Geometric (GCNConv)
- scikit-learn (AUC computation)
- numpy, pandas, scipy

No external GSL repositories (LDS, IDGL, ProGNN source code) are cloned or
imported. This eliminates dependency on potentially unmaintained code.

### Runtime Risk

StabilityResidual is the most expensive method (~9s per run on Cora). The full
audit matrix (240 runs) is estimated at ~1-2 hours on a single GPU. The GSL
proxies add moderate overhead from embedding computation and k-NN construction.

## Can the Paper Claim Comparison to GSL Baselines?

### What Can Be Claimed

The paper can honestly state:

1. We implement three GSL-inspired proxy baselines that capture the core
   mechanisms of IDGL, ProGNN, and LDS under matched pruning budgets.
2. These proxies use only training labels and feature information for graph
   construction (no label leakage).
3. StabilityResidual outperforms these proxies on homophilic citation FSCC
   (if the results support this).

### What Cannot Be Claimed

The paper cannot claim:

1. Comparison to the exact LDS, IDGL, or ProGNN methods as published.
2. That StabilityResidual would beat the full GSL methods with their complete
   bilevel optimization, iterative refinement, or joint training.
3. A comprehensive GSL baseline survey.

### Recommended Paper Wording

```text
We compare against GSL-inspired baselines that implement the core mechanisms
of IDGL (embedding-based graph construction), ProGNN (feature-smoothness
graph refinement), and LDS (edge weight learning). These are simplified
proxies under matched pruning budgets; full GSL reproduction with bilevel
optimization is deferred. [If proxies lose:] StabilityResidual outperforms
these GSL-inspired baselines, suggesting that prediction stability provides
a more effective edge signal than embedding similarity or feature smoothness
alone. [If proxies win:] The GSL-inspired baselines are competitive,
indicating that the pruning budget and feature information account for much
of the observed gain.
```

## Summary

| Risk Category | Level | Mitigation |
|---------------|-------|------------|
| GSL proxy simplification | Medium | Report as "inspired" baselines, not reproductions |
| Label leakage | Low | All methods use same label access |
| Validation label use | Low | LDS-Proxy uses val loss for model selection only |
| Device/numerical | Low | CUDA + float32, seeded |
| External dependencies | None | No external GSL repos needed |
| Runtime | Low | ~1-2 hours total on single GPU |
