# Experiment: Selective MCGC Regime Gate

## Objective

Design and evaluate a no-leak selective dynamics gate that keeps the useful
feature-ambiguous gains of MCGC while preventing degradation when feature
similarity already solves the edge-pruning problem.

Primary hypothesis:

```text
A feature-regime gate can use MCGC dynamics in feature-ambiguous regions and
fall back to Feature-only in feature-clear regions, improving
feature_similar_cross_class without degrading low_feature_similarity.
```

This experiment should produce a method suitable for the next AAAI confirmation
run, or clearly show why this direction fails.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `experiments/2026-06-04-adaptive-grage-search/result.md`
- `experiments/2026-06-04-adaptive-grage-search/failure_analysis.md`
- `experiments/2026-06-04-dynamics-mechanism-diagnostics/result.md`
- `src/grage/adaptive_score.py`
- `scripts/run_adaptive_grage_search.py`
- `tests/test_adaptive_score.py`
- `paper_draft/method.md`
- `paper_draft/related_work.md`

## Method Requirements

Implement at least two no-leak selective variants under `src/grage/`:

1. Hard selective MCGC:

   ```text
   A_e = 1[feature_similarity_e >= tau]
   score_e = R_f(e)
           + A_e C_e lambda_pos R(relu(bar S_e))
           - A_e C_e lambda_neg R(relu(-bar S_e))
   ```

2. Soft selective MCGC:

   ```text
   A_e = sigmoid(k * (feature_similarity_e - tau))
   score_e = R_f(e)
           + A_e C_e lambda_pos R(relu(bar S_e))
           - A_e C_e lambda_neg R(relu(-bar S_e))
   ```

Choose `tau` without label leakage. Allowed choices:

- fixed feature-similarity quantile computed from candidate edges only
- train-internal unsupervised quantile sweep selected by stability, not labels
- a conservative preset such as top feature-similarity quartile

Forbidden:

- validation labels, test labels, oracle labels, or `bad_edge_mask` for score
  construction or threshold selection.

## Tasks

1. Add selective score code and tests.
2. Extend `scripts/run_adaptive_grage_search.py` or add a focused runner for
   selective MCGC.
3. Run smoke:
   - Cora
   - `feature_similar_cross_class`
   - seed `0`
   - methods: Feature-only, MCGC, hard selective, soft selective
4. Search matrix:
   - datasets: `Cora`, `CiteSeer`
   - noise: `feature_similar_cross_class`, `low_feature_similarity`
   - seeds: `0..2`
   - methods: Feature-only, MCGC, at least 4 selective variants
5. Select best variant with constraints:
   - positive mean delta vs Feature-only on `feature_similar_cross_class`
   - low-feature-similarity degradation no worse than `-0.005`
   - win rate vs Feature-only at least `0.6` on feature-similar cases
6. Validation matrix:
   - datasets: `Cora`, `CiteSeer`, `PubMed`
   - noise: `feature_similar_cross_class`, `low_feature_similarity`,
     `degree_aligned_random`
   - seeds: `0..4`
   - compare Feature-only, MCGC, best selective variant, Random-Matched
7. Report whether the selective gate actually gates dynamics:
   - fraction of edges with `A_e > 0.5`
   - mean dynamic contribution by noise type
   - overlap between selected dynamic edges and pruned edges
   - runtime overhead versus MCGC and Feature-only
8. Write result and failure analysis suitable for paper decisions.

## Commands

Run at least:

```bash
python -m py_compile src/grage/adaptive_score.py scripts/run_adaptive_grage_search.py
pytest -q tests/test_adaptive_score.py tests/test_edge_gate.py tests/test_scoring.py tests/test_pruning.py
```

Then run smoke/search/validate commands for the selective runner. Document exact
commands in `result.md`.

## Output Contract

Write:

- `experiments/2026-06-04-selective-mcgc-regime-gate/result.md`
- `experiments/2026-06-04-selective-mcgc-regime-gate/metrics.json`
- `experiments/2026-06-04-selective-mcgc-regime-gate/failure_analysis.md`
- `experiments/2026-06-04-selective-mcgc-regime-gate/logs/`

The `metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-selective-mcgc-regime-gate",
  "status": "completed|partial|failed",
  "best_selective_method": "...",
  "candidate_selected_for_confirmation": false,
  "delta_vs_feature_only_feature_similar_cross_class_pp": 0.0,
  "low_feature_similarity_degradation_vs_feature_only_pp": 0.0,
  "win_rate_vs_feature_only": 0.0,
  "effect_size_vs_feature_only": 0.0,
  "dynamic_gate_active_fraction": 0.0,
  "runtime_vs_feature_only_ratio": 0.0,
  "num_result_rows": 0,
  "new_files_or_modules": [],
  "failure_modes": [],
  "claim_recommendation": "..."
}
```

If the selective gate fails, state whether the bottleneck is threshold selection,
gradient noise, feature-risk dominance, or budget constraints.

