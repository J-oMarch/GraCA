# Failure Analysis: FSCC Confirmation Rerun

## Per-Dataset/Noise-Type Failure Diagnosis

### FSCC (feature_similar_cross_class) — Primary Target

| (Dataset) | GraGE-Hybrid Δ (pp) | MCGC Δ (pp) | p-value (Hybrid) | p-value (MCGC) |
|-----------|---------------------|-------------|------------------|----------------|
| Cora | −0.80 | +1.58 | 0.004 | <0.001 |
| CiteSeer | −3.08 | −0.26 | <0.001 | 0.429 |
| PubMed | −3.63 | −3.47 | <0.001 | <0.001 |

**Diagnosis per dataset:**

1. **Cora (GraGE-Hybrid loses, MCGC wins)**: The gradient signal is non-random on Cora (grad_mean ~0.000002–0.000007, pos_ratio ~0.38–0.46). MCGC's consistency weighting helps here. But GraGE-Hybrid's rank-normalized gradient subtraction hurts. The feature_risk signal on Cora is weaker (AUC ~0.61), so there is room for gradient contribution. However, the improvement is small (+1.58 pp) and may be a budget/degree effect since Random-Matched also beats Feature-only on Cora (+4.35 pp).

2. **CiteSeer (Both lose)**: The gradient signal is very weak (grad_mean ~0.000000, pos_ratio ~0.48–0.50). Features are informative (AUC ~0.76), so gradient adds noise. The hybrid score's rank-normalized gradient terms flip edge rankings in the feature-similar regime.

3. **PubMed (Both lose)**: Similar to CiteSeer. Feature-only is near-optimal (0.672). Gradient signal is near-zero. Any gradient contribution is pure noise.

### Control Regimes

**cross_class_oracle**: Feature-only has strong signal (AUC ~0.81). Gradient adds noise. All GraGE methods lose by 2–7 pp.

**low_feature_similarity**: Features are already informative. Gradient adds noise. All GraGE methods lose by 1–5 pp.

**degree_aligned_random**: Random noise. Feature-only works by chance. GraGE methods lose by 1–4 pp.

### Heterophily Datasets

**Texas**: Feature-only wins. GraGE-Hybrid −6.49 pp on FSCC. The graph structure is very different from homophilic datasets.

**Wisconsin**: Feature-only wins. GraGE-Hybrid −5.49 pp on FSCC.

**Actor**: Feature-only wins. All methods perform poorly (~0.28 accuracy). The task is inherently hard.

## Mechanism Diagnosis

### Why does the gradient signal not help?

1. **Gradient magnitude is too small**: grad_mean ~0.000000–0.000007, while feature_risk ~0.84. The rank-normalized gradient contribution is dwarfed by feature_risk.

2. **Gradient is near-random**: pos_ratio ~0.38–0.54, close to 0.5 (random). The signal-to-noise ratio is too low.

3. **Rank normalization amplifies noise**: When gradient magnitudes are near-zero, rank normalization assigns arbitrary ranks, turning noise into signal.

4. **The hybrid score formula is flawed**: `R_feature + λ_pos * R(relu(grad)) − λ_neg * R(relu(-grad))` subtracts the rank of protective edges. When gradients are near-random, this subtracts noise from the feature signal.

### Does the gradient signal add value beyond feature similarity?

**No.** On FSCC, the gradient signal is near-random (AUC ~0.49–0.52 from prior diagnostics). Combining it with feature_risk degrades performance relative to feature_risk alone.

### Is the improvement consistent across seeds?

**No for GraGE-Hybrid** (win rate 0.10). **Partially for MCGC** on Cora (win rate 0.85), but not on other datasets.

### Is the improvement consistent across datasets?

**No.** MCGC wins on Cora but loses on CiteSeer and PubMed. GraGE-Hybrid loses everywhere.

## Recommendations for the Paper

1. **Do not claim**: "Training-dynamics signals improve graph pruning beyond feature similarity."

2. **Reframe**: The paper could focus on the engineering contribution (hybrid score framework, multi-checkpoint consistency) with honest reporting of where it helps and where it doesn't.

3. **Alternative direction**: If the paper must claim gradient signal utility, focus only on Cora and acknowledge the result is dataset-specific. But this is weak for AAAI.

4. **Strongest honest claim**: "Feature-only pruning is a strong baseline that current GraGE methods cannot consistently beat. The gradient signal is too weak to contribute meaningfully in most regimes."

5. **Possible rescue**: Investigate why Random-Matched and DegreeAwareRandom beat Feature-only on Cora. This suggests budget/degree effects dominate signal quality. A paper on "why pruning budget matters more than signal quality" could be interesting.

## Key Numbers Summary

- Primary FSCC: GraGE-Hybrid −2.50 pp (p=0.0012, d=−1.40), MCGC −0.72 pp (p=0.14)
- Best GraGE method: MCGC-cw3.0-lp0.1-ln0.5 on Cora (+1.58 pp, p<0.001)
- Total experiment rows: 1080 (360 primary + 540 controls + 180 heterophily)
- Runtime: ~3.5 hours total (160 min primary + 33 min controls + 21 min heterophily)
