# Experiment: Feature-Similar Cross-Class GraGE-Hybrid Confirmation

## Objective

Validate the paper-facing claim that GraGE-Hybrid provides useful graph evolution
information beyond static feature similarity when harmful edges are
feature-similar and therefore hard for Feature-only pruning.

This is an AAAI-readiness experiment, not a routine engineering sweep. Treat
Feature-only / similarity pruning as the main practical baseline, keep
EdgeInfluence-Pseudo historical only if referenced, and never use validation
labels, test labels, oracle labels, or `bad_edge_mask` to compute practical edge
scores. Oracle variants are diagnostic only and must not enter the main claim.

Primary hypothesis:

```text
On feature_similar_cross_class noise, a no-leak GraGE-Hybrid method based on
training dynamics beats Feature-only under matched pruning budgets with paired
statistical support.
```

Secondary hypotheses:

- Gains should be smaller on `low_feature_similarity`, where static feature
  pruning is already near the natural solution.
- The best method should improve bad-edge F1 and/or homophily without relying on
  oracle labels.
- The result should remain interpretable as a practical graph evolution method,
  not only a noisy-edge detector.

## Repository Context

Read these files first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_tables_grage_hybrid/GRAGE_HYBRID_DECISION_REPORT.md`
- `paper_tables_grage_hybrid/hybrid_vs_feature_only.csv`
- `paper_tables_grage_hybrid/hybrid_vs_feature_only_ttest.csv`
- `src/grage/hybrid_score.py`
- `src/grage/edge_gate_influence.py`
- `scripts/run_grage_hybrid_sweep.py`
- `scripts/build_grage_hybrid_tables.py`
- `src/eval/noise_injection.py`

## Tasks

1. Inspect the current GraGE-Hybrid implementation and runner.
2. Add a focused validation runner or extend `scripts/run_grage_hybrid_sweep.py`
   so this experiment can run a targeted matrix without editing constants by
   hand. Prefer a reusable script such as:

   ```bash
   python scripts/run_grage_targeted_validation.py \
     --output_dir experiments/2026-06-04-fscc-hybrid-confirmation/logs/results \
     --datasets Cora CiteSeer PubMed \
     --noise_types feature_similar_cross_class cross_class_oracle low_feature_similarity degree_aligned_random \
     --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
     --methods Feature-only GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 GraGE-Hybrid-FO-posneg-lp0.25-ln0.25 GraGE-Hybrid-FO-pos-lam0.5 Original+Noise Random-Matched DegreeAwareRandom GCN-Jaccard \
     --downstream_models GCN \
     --noise_ratio 0.3 \
     --prune_ratio 0.2
   ```

   It is acceptable to implement equivalent arguments on an existing runner if
   that is cleaner.

3. Run a small smoke command first on Cora, one seed, and two methods.
4. Run the full primary matrix:

   - datasets: `Cora`, `CiteSeer`, `PubMed`
   - noise types: `feature_similar_cross_class`, `cross_class_oracle`,
     `low_feature_similarity`, `degree_aligned_random`
   - seeds: `0..19`
   - downstream model: `GCN`
   - prune ratio: `0.2`
   - noise ratio: `0.3`
   - methods:
     - `Feature-only`
     - `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`
     - `GraGE-Hybrid-FO-posneg-lp0.25-ln0.25`
     - `GraGE-Hybrid-FO-pos-lam0.5`
     - `Original+Noise`
     - `Random-Matched`
     - `DegreeAwareRandom`
     - `GCN-Jaccard`

5. Run a small heterophily regime slice to identify failure modes rather than
   to make the main claim:

   - datasets: `Texas`, `Wisconsin`, and `Actor` if available
   - noise types: `feature_similar_cross_class`, `degree_aligned_random`
   - seeds: `0..4`
   - methods: `Feature-only`, best GraGE-Hybrid candidate,
     `Random-Matched`, `DegreeAwareRandom`, `GCN-Jaccard`
   - report separately from the Cora/CiteSeer/PubMed confirmation table.

6. If the full matrix is clearly too slow, keep all 20 seeds for
   `feature_similar_cross_class` and reduce control noise types to seeds `0..9`.
   Document the reduction explicitly.
7. Build paper-facing tables under:

   ```text
   experiments/2026-06-04-fscc-hybrid-confirmation/logs/tables/
   ```

8. Compute paired statistics against Feature-only:

   - mean delta
   - std delta
   - 95% CI
   - paired t-test
   - Wilcoxon signed-rank test
   - win rate
   - per-dataset breakdown
   - per-noise breakdown

9. Write a concise decision report explaining whether the focused AAAI claim is
   supported, unsupported, or only dataset-specific.
10. Write `failure_analysis.md` with dataset/noise regimes where GraGE-Hybrid
    loses to Feature-only, suspected causes, and the next method change that
    would be justified by the evidence.

## Commands

Run at least:

```bash
python -m py_compile scripts/run_grage_hybrid_sweep.py scripts/build_grage_hybrid_tables.py
pytest -q tests/test_edge_gate.py tests/test_unrolled_hypergradient.py tests/test_pruning.py
```

After implementing the targeted runner, run an equivalent smoke test:

```bash
python scripts/run_grage_targeted_validation.py \
  --output_dir experiments/2026-06-04-fscc-hybrid-confirmation/logs/smoke \
  --datasets Cora \
  --noise_types feature_similar_cross_class \
  --seeds 0 \
  --methods Feature-only GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 \
  --downstream_models GCN \
  --noise_ratio 0.3 \
  --prune_ratio 0.2
```

Then run the full matrix described in Tasks.

## Output Contract

Write:

- `experiments/2026-06-04-fscc-hybrid-confirmation/result.md`
- `experiments/2026-06-04-fscc-hybrid-confirmation/metrics.json`
- `experiments/2026-06-04-fscc-hybrid-confirmation/logs/`
- `experiments/2026-06-04-fscc-hybrid-confirmation/failure_analysis.md`

The `metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-fscc-hybrid-confirmation",
  "status": "completed|partial|failed",
  "primary_claim_supported": true,
  "best_method": "...",
  "feature_similar_cross_class_delta_vs_feature_only": 0.0,
  "feature_similar_cross_class_paired_t_p": 1.0,
  "feature_similar_cross_class_wilcoxon_p": 1.0,
  "feature_similar_cross_class_win_rate": 0.0,
  "feature_similar_cross_class_effect_size": 0.0,
  "heterophily_delta_vs_feature_only": 0.0,
  "failure_modes": [],
  "num_result_rows": 0,
  "notes": "..."
}
```

If the result is negative, do not hide it. State the strongest defensible claim
and the failure mode.
