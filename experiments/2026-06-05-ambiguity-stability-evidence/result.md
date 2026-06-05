# Ambiguity and Stability Evidence Result

## Executive Summary

- Mode: `full`.
- Rows: `2160`.
- Primary FSCC StabilityResidual vs Feature-only: +2.06 pp, paired-t p=6.682e-10, Wilcoxon p=1.191e-08, win=0.85, d=0.95, n=60.
- P0 high-ambiguity gain share: `0.8137651821862352`.
- P1 Feature+Stability vs Feature+Confidence: +0.31 pp, paired-t p=0.1984, Wilcoxon p=0.2508, win=0.53, d=0.17, n=60.
- P1 Feature+Stability vs Feature+Random Stability: +1.73 pp, paired-t p=1.204e-09, Wilcoxon p=3.208e-08, win=0.87, d=0.93, n=60.
- P1 Feature+Stability vs Feature+Shuffled Stability: +1.78 pp, paired-t p=8.233e-11, Wilcoxon p=2.638e-09, win=0.85, d=1.02, n=60.
- P1 Feature+Stability vs Feature+Permuted Stability: +1.63 pp, paired-t p=2.023e-09, Wilcoxon p=3.153e-08, win=0.83, d=0.91, n=60.

## P0 Leakage Audit

Low/Medium/High ambiguity buckets are defined only from Feature-only risk and
distance to the Feature-only pruning decision boundary. Labels and
`bad_edge_mask` are used only after bucket assignment for diagnostics.

Bucket convention:

- `0`: Low ambiguity, farthest from the feature-only pruning boundary.
- `1`: Medium ambiguity.
- `2`: High ambiguity, closest to the feature-only pruning boundary.

## P0 Contribution

The High-only gain share is `0.8137651821862352`. Interpret this as the fraction
of the full StabilityResidual FSCC gain reproduced when the residual is active
only in the High-Ambiguity bucket. This is an attribution heuristic, not a causal
proof.

### P0 FSCC Method Summary

| phase | method | mean | std | count |
| --- | --- | --- | --- | --- |
| P0 | Feature+Stability | 0.6325 | 0.0497 | 60 |
| P0 | StabilityResidual-v5-dp0.15-grad-frozen | 0.6315 | 0.0514 | 60 |
| P0 | Feature+Residual-HighOnly | 0.6277 | 0.0479 | 60 |
| P0 | Feature+Residual-MediumOnly | 0.6164 | 0.0487 | 60 |
| P0 | Feature-only | 0.6109 | 0.0500 | 60 |
| P0 | Feature+Residual-LowOnly | 0.6100 | 0.0505 | 60 |

### P0 Paired Statistics

| comparison | delta_pp | paired_t_p | wilcoxon_p | win_rate | cohens_d | n |
| --- | --- | --- | --- | --- | --- | --- |
| StabilityResidual vs Feature-only | 2.0583 | 0.0000 | 0.0000 | 0.8500 | 0.9500 | 60 |
| HighOnly vs Feature-only | 1.6750 | 0.0000 | 0.0000 | 0.8000 | 0.7942 | 60 |
| MediumOnly vs Feature-only | 0.5467 | 0.1115 | 0.2015 | 0.4667 | 0.2086 | 60 |
| LowOnly vs Feature-only | -0.0900 | 0.3765 | 0.3624 | 0.4500 | -0.1150 | 60 |

### P0 Bucket Diagnostics

| bucket | count | bad_count | fo_precision | fo_recall | fo_f1 | sr_precision | sr_recall | sr_f1 | feature_risk_auc | raw_stability_auc | residual_auc | sr_only_bad_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Low | 15647.4333 | 442.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7712 | 0.6062 | 0.6068 | 0.0000 |
| Medium | 15664.6000 | 3364.7000 | 0.0000 | 0.0000 | 0.0000 | 0.4005 | 0.0834 | 0.1347 | 0.6373 | 0.6279 | 0.6281 | 0.6107 |
| High | 15620.6333 | 7022.9667 | 0.3057 | 0.3997 | 0.3425 | 0.4428 | 0.5945 | 0.4990 | 0.2990 | 0.6331 | 0.6323 | 0.6891 |

## P1 Alignment

Aligned stability is considered supported only if Feature+Stability beats
confidence, random, shuffled, and node-permuted stability controls on the paired
FSCC matrix.

### P1 FSCC Method Summary

| phase | method | mean | std | count |
| --- | --- | --- | --- | --- |
| P1 | Feature+Stability | 0.6331 | 0.0492 | 60 |
| P1 | Feature+Confidence | 0.6300 | 0.0513 | 60 |
| P1 | Feature+Permuted Stability | 0.6168 | 0.0461 | 60 |
| P1 | Feature+Random Stability | 0.6158 | 0.0463 | 60 |
| P1 | Feature+Shuffled Stability | 0.6153 | 0.0484 | 60 |
| P1 | Feature-only | 0.6114 | 0.0499 | 60 |

### P1 Paired Statistics

| comparison | delta_pp | paired_t_p | wilcoxon_p | win_rate | cohens_d | n |
| --- | --- | --- | --- | --- | --- | --- |
| Stability vs Feature-only | 2.1667 | 0.0000 | 0.0000 | 0.8667 | 0.9948 | 60 |
| Stability vs Confidence | 0.3117 | 0.1984 | 0.2508 | 0.5333 | 0.1679 | 60 |
| Stability vs Random | 1.7333 | 0.0000 | 0.0000 | 0.8667 | 0.9305 | 60 |
| Stability vs Shuffled | 1.7783 | 0.0000 | 0.0000 | 0.8500 | 1.0193 | 60 |
| Stability vs Permuted | 1.6300 | 0.0000 | 0.0000 | 0.8333 | 0.9133 | 60 |

## Decision Answers

1. Supports current claim: `True`.
2. Strengthens AAAI story: `True`.
3. Reduces reviewer risk: `True`.
4. New failure evidence: `False`.
5. Claim needs shrinkage: `False`.

## Output Tables

- `experiments/2026-06-05-ambiguity-stability-evidence/logs/full_method_summary_all.csv`
- `experiments/2026-06-05-ambiguity-stability-evidence/logs/full_method_summary_fscc.csv`
- raw results under `experiments/2026-06-05-ambiguity-stability-evidence/logs/full/results.csv`
