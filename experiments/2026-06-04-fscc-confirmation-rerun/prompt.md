# Experiment: FSCC Matched-Budget Confirmation Rerun

## Objective

Recover the failed matched-budget confirmation experiment with a direct,
auditable runner. The experiment must decide whether any current practical GraGE
variant beats Feature-only under matched pruning budgets, with multi-seed
statistics and failure analysis.

Primary claim to test:

```text
Current practical GraGE variants improve downstream accuracy over Feature-only
on feature_similar_cross_class noise without label leakage.
```

Treat this as a confirmation/rerun, not a method-search task. Do not introduce a
new method unless needed to make existing runners callable.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `experiments/2026-06-04-fscc-hybrid-confirmation/result.md`
- `experiments/2026-06-04-adaptive-grage-search/result.md`
- `experiments/2026-06-04-dynamics-mechanism-diagnostics/result.md`
- `scripts/run_grage_hybrid_sweep.py`
- `scripts/run_adaptive_grage_search.py`
- `src/grage/hybrid_score.py`
- `src/grage/adaptive_score.py`

## Tasks

1. Verify the runner interface and add a small targeted runner only if current
   scripts cannot run the exact matrix. Prefer reusing existing code.
2. Run a one-case smoke test:
   - dataset: `Cora`
   - noise: `feature_similar_cross_class`
   - seed: `0`
   - methods: `Feature-only`, `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`,
     `MCGC-cw3.0-lp0.1-ln0.5`, `Random-Matched`
3. Run the primary confirmation:
   - datasets: `Cora`, `CiteSeer`, `PubMed`
   - noise: `feature_similar_cross_class`
   - seeds: `0..19`
   - downstream model: `GCN`
   - noise ratio: `0.3`
   - prune ratio: `0.2`
   - methods: `Feature-only`, `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`,
     `MCGC-cw3.0-lp0.1-ln0.5`, `Random-Matched`, `DegreeAwareRandom`,
     `GCN-Jaccard`
4. Run control regimes:
   - datasets: `Cora`, `CiteSeer`, `PubMed`
   - noise: `cross_class_oracle`, `low_feature_similarity`,
     `degree_aligned_random`
   - seeds: `0..9`
   - same methods and budget
5. Run a small heterophily slice:
   - datasets: `Texas`, `Wisconsin`, `Actor` if available
   - noise: `feature_similar_cross_class`, `degree_aligned_random`
   - seeds: `0..4`
6. Compute paired stats versus Feature-only:
   - mean/std accuracy
   - paired delta in percentage points
   - paired t-test and Wilcoxon where possible
   - win rate
   - effect size
   - runtime
7. Write a decision report that states whether the current GraGE direction is
   supported, unsupported, or only regime-specific.
8. Write `failure_analysis.md` covering datasets/noise regimes where GraGE loses
   to Feature-only and whether the failure is signal quality, over-pruning,
   degree constraints, or runtime.

## Commands

Run at least:

```bash
python -m py_compile scripts/run_grage_hybrid_sweep.py scripts/run_adaptive_grage_search.py src/grage/hybrid_score.py src/grage/adaptive_score.py
pytest -q tests/test_edge_gate.py tests/test_scoring.py tests/test_adaptive_score.py tests/test_pruning.py
```

If a new targeted runner is added, also run `python -m py_compile` on it.

## Output Contract

Write:

- `experiments/2026-06-04-fscc-confirmation-rerun/result.md`
- `experiments/2026-06-04-fscc-confirmation-rerun/metrics.json`
- `experiments/2026-06-04-fscc-confirmation-rerun/failure_analysis.md`
- `experiments/2026-06-04-fscc-confirmation-rerun/logs/`

The `metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-fscc-confirmation-rerun",
  "status": "completed|partial|failed",
  "primary_claim_supported": false,
  "best_method": "...",
  "best_method_family": "...",
  "feature_similar_cross_class_delta_vs_feature_only_pp": 0.0,
  "feature_similar_cross_class_p_value": 1.0,
  "feature_similar_cross_class_win_rate": 0.0,
  "feature_similar_cross_class_effect_size": 0.0,
  "low_feature_similarity_delta_vs_feature_only_pp": 0.0,
  "num_result_rows": 0,
  "failure_modes": [],
  "tables": [],
  "notes": "..."
}
```

No oracle, validation labels, test labels, or `bad_edge_mask` may be used to
compute practical edge scores. `bad_edge_mask` is evaluation-only.

