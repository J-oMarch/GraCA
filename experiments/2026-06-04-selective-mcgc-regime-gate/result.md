# Result: Selective MCGC Regime Gate

## Summary

**Status**: Partial — the selective gate prevents MCGC degradation but provides
marginal gains over Feature-only on the target noise type.

**Best selective variant**: `Selective-MCGC-hard-q0.5-lp0.1-ln0.5`
**Gate type**: Hard indicator, tau = feature-similarity median (quantile 0.5)
**Candidate NOT selected for confirmation**: FSCC gains too small, win rate below
0.6 threshold.

## Key Validation Results (5 seeds × 3 datasets × 3 noise types)

### Paired Deltas vs Feature-only

| Noise Type | Delta | p-value | Win Rate | Significant |
|-----------|-------|---------|----------|-------------|
| feature_similar_cross_class | +0.0009 ± 0.0063 | 0.575 | 0.47 | No |
| low_feature_similarity | +0.0190 ± 0.0293 | **0.025** | 0.53 | **Yes** |
| degree_aligned_random | +0.0009 ± 0.0077 | 0.670 | 0.27 | No |

### Per-Dataset Deltas (feature_similar_cross_class)

| Dataset | Delta |
|---------|-------|
| Cora | +0.0026 |
| CiteSeer | -0.0008 |
| PubMed | +0.0010 |

### Per-Dataset Deltas (low_feature_similarity)

| Dataset | Delta |
|---------|-------|
| Cora | **+0.0554** |
| CiteSeer | +0.0018 |
| PubMed | -0.0002 |

### MCGC vs Feature-only (Validation)

| Noise Type | Delta | p-value | Significant |
|-----------|-------|---------|-------------|
| feature_similar_cross_class | -0.0051 | 0.478 | No |
| low_feature_similarity | **-0.0246** | **0.012** | **Yes** |
| degree_aligned_random | **-0.0191** | **0.001** | **Yes** |

### Selective vs MCGC (Direct Comparison)

| Noise Type | Delta |
|-----------|-------|
| feature_similar_cross_class | +0.0060 |
| low_feature_similarity | **+0.0436** |
| degree_aligned_random | +0.0199 |

## Overall Validation Performance

| Method | Mean Test Acc | Std |
|--------|--------------|-----|
| **Selective-MCGC-hard-q0.5-lp0.1-ln0.5** | **0.6787** | 0.0727 |
| Feature-only | 0.6718 | 0.0673 |
| MCGC-cw3.0-lp0.1-ln0.5 | 0.6555 | 0.0622 |
| GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 | 0.6548 | 0.0729 |
| Random-Matched | 0.6452 | 0.0678 |

## Gate Behavior

- **Gate active fraction**: 0.501 (top 50% of edges by feature similarity)
- **Tau quantile**: 0.5 (median feature similarity)
- **Mean dynamic contribution**: near zero (gradient signal is very weak)

## Controls

### Shuffled-Gradient Control (from search)
- Real selective FSCC delta: +0.0030
- Shuffled control FSCC delta: -0.0017
- Real-vs-shuffled: +0.0047 (positive, but small)

### Zero-Gate Control (from search)
- Zero-gate FSCC delta: +0.0018
- Zero-gate LFS delta: +0.0318
- **Note**: Zero-gate control is contaminated by different downstream random
  state (MCGC pipeline consumes random numbers before downstream training)

## Runtime

| Method | Mean Runtime | Ratio vs FO |
|--------|-------------|-------------|
| Feature-only | 2.6s | 1.0x |
| MCGC | 6.6s | 2.5x |
| Selective MCGC | 7.9s | 3.0x |

## Interpretation

### What works
1. **The selective gate successfully prevents MCGC degradation.** Raw MCGC
   degrades LFS by -2.5pp (significant, p=0.012) and DAR by -1.9pp
   (significant, p=0.001). The selective gate converts these to +1.9pp and
   +0.1pp respectively.
2. **The selective variant is the best overall method** across all conditions
   (0.6787 vs 0.6718 for Feature-only).
3. **LFS improvement is statistically significant** (p=0.025), driven primarily
   by Cora (+5.5pp).

### What does not work
1. **FSCC gains are negligible.** The selective variant gets only +0.1pp on the
   target noise type (not significant, p=0.575).
2. **Win rate on FSCC is 0.47**, below the 0.6 threshold.
3. **The gradient signal is too weak to meaningfully improve FSCC.** Even with
   the gate activated for 50% of edges, the dynamic contribution is near zero.
4. **The gate is not selective in a meaningful way.** With tau at the median
   feature similarity, the gate activates for exactly 50% of edges — essentially
   random selection rather than regime-aware gating.

### Root cause
The edge-gate gradient signal has near-zero magnitude (mean ~0.000003) and
near-random direction. Even with multi-checkpoint consistency weighting, the
signal cannot meaningfully improve upon feature-risk ranking. The selective gate
prevents degradation by suppressing the noise from MCGC, but cannot create
improvement because the underlying signal is too weak.

## Candidate Selected for Confirmation

**No.** The selective gate is a successful engineering solution (it prevents MCGC
degradation), but it does not support the paper claim that "training-dynamics
signals provide useful information beyond feature similarity in the
feature-ambiguous regime." The FSCC gains are within noise, and the gate
behavior (50% activation at median) does not demonstrate regime-aware gating.

## Commands Run

```bash
# Compilation check
python -m py_compile src/grage/adaptive_score.py
python -m py_compile scripts/run_adaptive_grage_search.py

# Tests
pytest -q tests/test_adaptive_score.py tests/test_edge_gate.py tests/test_scoring.py tests/test_pruning.py

# Smoke
python scripts/run_adaptive_grage_search.py \
  --mode selective_smoke \
  --output_dir experiments/2026-06-04-selective-mcgc-regime-gate/logs/smoke

# Search
python scripts/run_adaptive_grage_search.py \
  --mode selective_search \
  --output_dir experiments/2026-06-04-selective-mcgc-regime-gate/logs/search

# Validation
python scripts/run_adaptive_grage_search.py \
  --mode selective_validate \
  --best_candidate '{"name":"Selective-MCGC-hard-q0.5-lp0.1-ln0.5","type":"selective_mcgc","gate_type":"hard","tau_quantile":0.5,"lambda_pos":0.1,"lambda_neg":0.5,"consistency_weight":3.0,"score_ratio":0.3,"checkpoint_fractions":[0.3,0.5,0.7,0.9],"total_epochs":200,"checkpoint_control":"real"}' \
  --output_dir experiments/2026-06-04-selective-mcgc-regime-gate/logs/validate

# Analysis
python scripts/analyze_adaptive_grage_search.py \
  --selection_mode selective \
  --search_csv experiments/2026-06-04-selective-mcgc-regime-gate/logs/search/results.csv \
  --validation_csv experiments/2026-06-04-selective-mcgc-regime-gate/logs/validate/results.csv \
  --output_dir experiments/2026-06-04-selective-mcgc-regime-gate
```
