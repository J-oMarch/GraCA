# Confidence Risk Audit Result

## Executive Summary

- Mode: `full`.
- Edges analyzed: `703990`.
- Datasets: `['Cora', 'CiteSeer', 'PubMed']`.
- Seeds: `10`.

## 1. Is Stability Distinguishable from Confidence?

**Global AUC comparison** (higher = better bad-edge detection):

| Score | AUC |
| --- | --- |
| Feature Risk | 0.7535 |
| Confidence | 0.7979 |
| Raw Stability | 0.6028 |
| Residualized Stability | 0.5991 |
| StabilityResidual Final | 0.8027 |

StabilityResidual AUC delta over Confidence: **+0.0047**.

**Same-confidence-bucket AUC delta**: +0.0290
(Positive = residual stability improves bad-edge detection within confidence strata.)

**Partial correlation** (residual stability coefficient after controlling for
feature risk and confidence): +0.2121
(AUC improvement from adding residual: +0.0024)

**Conclusion**: Stability provides signal beyond confidence.

## 2. Does the Distinction Hold in High-Ambiguity FSCC Edges?

High-ambiguity same-confidence-bucket AUC delta: +0.0317.

Residual stability still improves detection in High-Ambiguity edges after confidence control.

## 3. Confidence-Bucket AUC Analysis

| conf_bucket | count | bad_count | feature_risk_auc | confidence_edge_score_auc | stability_residual_final_auc | resid_auc_minus_conf_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 140799.0000 | 1749.0000 | 0.6687 | 0.6842 | 0.6847 | 0.0005 |
| 1.0000 | 140797.0000 | 8170.0000 | 0.6645 | 0.6262 | 0.6591 | 0.0329 |
| 2.0000 | 140799.0000 | 27733.0000 | 0.6240 | 0.6118 | 0.6517 | 0.0399 |
| 3.0000 | 140802.0000 | 55768.0000 | 0.4800 | 0.5466 | 0.5825 | 0.0359 |
| 4.0000 | 140793.0000 | 69030.0000 | 0.2552 | 0.4615 | 0.4971 | 0.0357 |

## 4. Confidence-Matched Bad-Edge Rate

| conf_stratum | count | bad_count | sr_bad_edge_rate | conf_bad_edge_rate | sr_minus_conf_bad_rate |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 70399.0000 | 397.0000 |  |  |  |
| 1.0000 | 70400.0000 | 1352.0000 |  |  |  |
| 2.0000 | 70399.0000 | 2594.0000 |  |  |  |
| 3.0000 | 70398.0000 | 5576.0000 | 0.0000 |  |  |
| 4.0000 | 70400.0000 | 10172.0000 | 0.2759 |  |  |
| 5.0000 | 70399.0000 | 17561.0000 | 0.4533 |  |  |
| 6.0000 | 70398.0000 | 25828.0000 | 0.5112 |  |  |
| 7.0000 | 70404.0000 | 29940.0000 | 0.4824 | 0.2822 | 0.2002 |
| 8.0000 | 70396.0000 | 35773.0000 | 0.5199 | 0.5083 | 0.0116 |
| 9.0000 | 70397.0000 | 33257.0000 | 0.4784 | 0.4724 | 0.0060 |

## 5. What Should the Paper Claim?

**Claim recommendation**: `support_stability_beyond_confidence`.

StabilityResidual provides edge-quality evidence beyond what confidence alone captures, particularly in feature-ambiguous regions. The paper can maintain the current claim with an explicit discussion of the confidence relationship.

## 6. What Should Be Admitted in Limitations?

- Feature+Confidence is close to Feature+Stability in the P1 paired test (+0.31 pp, p=0.20).
- Residual stability adds modest but detectable signal beyond confidence within confidence strata.
- The partial correlation analysis confirms residual stability contributes after controlling for feature risk and confidence.

## 7. Output Files

- Edge diagnostics: `experiments/2026-06-05-confidence-risk-audit/logs/full/edge_diagnostics.csv`
- Confidence bucket summary: `experiments/2026-06-05-confidence-risk-audit/logs/full/confidence_bucket_summary.csv`
- AUC summary: `experiments/2026-06-05-confidence-risk-audit/logs/full/auc_summary.csv`
- Matched analysis: `experiments/2026-06-05-confidence-risk-audit/logs/full/matched_analysis.csv`
