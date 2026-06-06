# AAAI Readiness Audit

## Current Verdict

Status: evidence-ready, paper assembly still required.

The main StabilityResidual result and the P0/P1 mechanism evidence now support
the narrowed AAAI claim. The remaining work is paper assembly, figure/table
polish, and final reviewer-risk checking, not open-ended method search.

## Completed Evidence

- Main method fixed: `StabilityResidual-v5-dp0.15-grad-frozen`.
- Primary result: `+1.59 pp` vs Feature-only on Cora/CiteSeer/PubMed FSCC,
  `p<0.001`, win rate `0.83`, Cohen's d `0.70`.
- Control regimes: no material degradation on LFS (`+0.55 pp`) or DAR
  (`+0.81 pp`).
- Old routes demoted: raw edge-gate gradients, GraGE-Hybrid, MCGC, and
  Selective-MCGC are historical negative or auxiliary evidence.
- Heterophily boundary complete: Texas/Wisconsin/Actor fail; do not claim
  heterophily robustness.
- GSL positioning complete at proxy level: StabilityResidual is competitive with
  GSL-inspired proxies but does not beat LDS-Proxy.
- Theory draft now uses Definition / Proposition / Proof Sketch and avoids
  universal or optimal-structure claims.
- P0 ambiguity contribution complete:
  `2026-06-05-ambiguity-stability-evidence` shows FSCC
  StabilityResidual `+2.06 pp` vs Feature-only (`p=6.68e-10`, win rate `0.85`,
  d `0.95`). High-only residual activation gives `+1.68 pp` and recovers
  `81.4%` of the full gain. High-bucket SR-only changed prunes have `68.9%`
  bad-edge rate.
- P1 alignment validation complete: aligned stability beats random, shuffled,
  and node-permuted controls by `+1.63` to `+1.78 pp` with `p<1e-8`.
- Confidence risk audit complete: `2026-06-05-confidence-risk-audit` analyzed
  703,990 edges across Cora, CiteSeer, PubMed with 10 seeds. StabilityResidual
  AUC (0.803) exceeds Confidence AUC (0.798) globally. Within confidence
  strata, residual stability adds `+0.029` AUC, rising to `+0.032` in
  High-Ambiguity edges. Partial correlation coefficient for residual stability
  is `+0.21` after controlling for feature risk and confidence. This confirms
  stability provides edge-quality evidence beyond confidence.

## Implementation Audit

- High-Ambiguity corresponds to edges closest to the Feature-only pruning
  decision boundary. The public bucket convention is `0=Low/farthest`,
  `1=Medium`, `2=High/closest`, and the full results use this convention.
- The experiment writes the full output contract: `result.md`, `metrics.json`,
  `failure_analysis.md`, `logs/full/results.csv`, and summary tables.
- The full CSV has 2160 rows: 3 datasets, 3 noise regimes, 20 seeds, P0/P1
  phases, and 6 methods per phase. No test accuracy values are missing.
- Bucket definitions use only feature-derived signals; labels and
  `bad_edge_mask` appear only in post-hoc diagnostics.

## Claim Gate

Allowed:

```text
Prediction stability provides complementary edge-quality evidence beyond feature
similarity in homophilic feature-ambiguous citation regimes.
```

P0/P1 are supportive, so no additional claim shrinkage is required beyond the
existing homophilic feature-ambiguous citation regime boundary.

## Current Reviewer Risks

- Feature+Confidence is close to aligned stability in P1 paired accuracy
  (`+0.31 pp` lower, `p=0.198`), but the confidence risk audit
  (`2026-06-05-confidence-risk-audit`) shows StabilityResidual AUC (0.803)
  exceeds Confidence AUC (0.798) globally, with `+0.029` AUC within confidence
  strata and `+0.032` in High-Ambiguity edges. Residual stability contributes
  after controlling for feature risk and confidence (partial correlation
  coefficient `+0.21`). This evidence substantially reduces the reviewer risk.
- CiteSeer is positive but weak individually.
- Full LDS/IDGL/ProGNN are not reproduced.
- Runtime is about `4x` Feature-only.
- Heterophily failure must be explicit and not softened.

## Stop / Continue Decision

Do not start P4 datasets yet. P0-P3 are complete enough for the current claim;
the next step is paper drafting, figure/table construction, and final
submission-risk review. Coauthor-CS, Coauthor-Physics, and ogbn-arxiv should
only be considered if they strengthen the same narrow claim without expanding
the scope.


## Runtime Profile

- Runtime ratio: **4.28x** Feature-only.
- Extra overhead: **5.00s**.
- Accuracy delta: **4.5 pp**.
- Claim: accuracy-cost tradeoff, not efficiency superiority.
- Component breakdown: see `paper_draft/runtime_table.md`.
