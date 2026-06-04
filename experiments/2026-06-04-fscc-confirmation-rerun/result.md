# FSCC Matched-Budget Confirmation Rerun — Decision Report

## Executive Summary

**The primary claim is not supported.** Under matched pruning budgets (prune_ratio=0.2) across 20 seeds × 3 datasets × 6 methods on `feature_similar_cross_class` noise, Feature-only pruning consistently outperforms all GraGE variants. GraGE-Hybrid loses by −2.50 pp (p<0.001, Cohen's d=−1.40, win rate=0.10). MCGC loses by −0.72 pp (not consistently significant). The negative result holds across control regimes and heterophily datasets. Feature-only is the strongest practical method in this experimental setting.

## Primary FSCC Results (feature_similar_cross_class, 20 seeds)

| Method | Cora Mean±Std | CiteSeer Mean±Std | PubMed Mean±Std | Overall Mean±Std |
|--------|--------------|-------------------|-----------------|-----------------|
| Feature-only | 0.603±0.016 | 0.560±0.018 | 0.672±0.016 | 0.612±0.050 |
| Random-Matched | 0.646±0.015 | 0.544±0.012 | 0.640±0.018 | 0.610±0.050 |
| GCN-Jaccard | 0.598±0.018 | 0.554±0.016 | 0.673±0.015 | 0.608±0.052 |
| DegreeAwareRandom | 0.643±0.015 | 0.547±0.019 | 0.634±0.015 | 0.608±0.047 |
| MCGC-cw3.0-lp0.1-ln0.5 | 0.618±0.020 | 0.557±0.013 | 0.638±0.015 | 0.604±0.038 |
| GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 | 0.595±0.014 | 0.529±0.016 | 0.636±0.018 | 0.587±0.047 |

## Paired Statistics vs Feature-only (FSCC)

| Method | Mean Delta (pp) | t-test p-value | Win Rate | Cohen's d |
|--------|----------------|---------------|----------|-----------|
| GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 | −2.50 | 0.0012 | 0.10 | −1.40 |
| MCGC-cw3.0-lp0.1-ln0.5 | −0.72 | 0.1430 | 0.43 | −0.35 |
| DegreeAwareRandom | −0.38 | — | 0.47 | — |
| GCN-Jaccard | −0.35 | — | 0.35 | — |
| Random-Matched | −0.16 | — | 0.42 | — |

Per-dataset detail on FSCC:
- **Cora**: MCGC +1.58 pp (p<0.001, significant), DegreeAwareRandom +4.00 pp (p<0.001), Random-Matched +4.35 pp (p<0.001). But GraGE-Hybrid −0.80 pp.
- **CiteSeer**: All methods lose to Feature-only. GraGE-Hybrid −3.08 pp (p<0.001).
- **PubMed**: All methods lose to Feature-only. GraGE-Hybrid −3.63 pp (p<0.001), MCGC −3.47 pp (p<0.001).

## Control Regime Results (10 seeds)

| Method | cross_class_oracle | low_feature_similarity | degree_aligned_random |
|--------|-------------------|----------------------|---------------------|
| Feature-only | 0.728 | 0.711 | 0.699 |
| GCN-Jaccard | 0.729 | 0.712 | 0.699 |
| GraGE-Hybrid | 0.712 | 0.699 | 0.685 |
| MCGC | 0.698 | 0.692 | 0.682 |
| Random-Matched | 0.692 | 0.677 | 0.672 |
| DegreeAwareRandom | 0.689 | 0.665 | 0.672 |

On control regimes, Feature-only and GCN-Jaccard are tied. GraGE methods lose on all control regimes.

## Heterophily Slice (5 seeds)

| Method | Texas | Wisconsin | Actor | Overall |
|--------|-------|-----------|-------|---------|
| Feature-only | 0.619 | 0.587 | 0.280 | 0.515 |
| GraGE-Hybrid | 0.586 | 0.558 | 0.276 | 0.504 |
| MCGC | 0.589 | 0.554 | 0.275 | 0.499 |
| GCN-Jaccard | 0.551 | 0.565 | 0.281 | 0.490 |

On heterophily data, Feature-only wins. GraGE-Hybrid loses by −4.15 pp overall.

## Decision

Based on the decision rules in `docs/PROJECT_STATE.md`:

**Verdict: "revise method or reframe paper"**

Rationale:
1. GraGE-Hybrid consistently loses to Feature-only on FSCC (−2.50 pp, p<0.001, Cohen's d=−1.40).
2. MCGC does not consistently beat Feature-only (−0.72 pp overall, only significant on Cora).
3. No GraGE variant beats Feature-only on control regimes.
4. The previous adaptive-grage-search result (MCGC +1.5 pp on FSCC with 5 seeds) did not survive the expanded 20-seed confirmation.
5. Random-Matched and DegreeAwareRandom beat Feature-only on Cora, suggesting the Cora result is driven by budget/degree effects rather than signal quality.

## Failure Modes

1. **CiteSeer**: All methods lose to Feature-only. The gradient signal adds noise.
2. **PubMed**: All methods lose to Feature-only. Feature-only is already near-optimal.
3. **GraGE-Hybrid**: Systematic loss across all datasets (−0.80 to −3.63 pp).
4. **Heterophily**: All GraGE methods lose on Texas, Wisconsin, and Actor.

## Generated Tables

- `logs/tables/primary_fscc.csv`
- `logs/tables/control_regimes.csv`
- `logs/tables/heterophily.csv`
- `logs/tables/primary_paired_stats.csv`
- `logs/tables/controls_paired_stats.csv`
- `logs/primary/results.csv` (360 rows)
- `logs/controls/results.csv` (540 rows)
- `logs/heterophily/results.csv` (180 rows)
