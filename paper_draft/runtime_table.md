# Runtime Profile Table

## Accuracy-Cost Tradeoff

| Method | Total Time | Test Acc | Overhead |
|---|---|---|---|
| Feature-only | 1.91s | 0.6089 | baseline |
| StabilityResidual-v5-dp0.15-grad-frozen | 8.04s | 0.6258 | +6.14s (4.22x) |

Accuracy delta: **1.69 pp**.

## StabilityResidual Component Breakdown

| Component | Time |
|---|---|
| Probe/model-view training | 0.25s |
| Gradient confidence collection | 3.11s |
| Stability scoring (multi-view + residualization) | 2.88s |
| Downstream retraining | 0.31s |

## Claim

**Accuracy-cost tradeoff, not efficiency superiority.**

The extra overhead is dominated by multi-view graph training and stability
residualization.  This buys approximately 1.69 pp on homophilic citation
FSCC at the cost of roughly 4.22x wall-clock time.
