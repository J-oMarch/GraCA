# Experiment: Training-Dynamics Mechanism Diagnostics

## Objective

Produce paper-facing mechanism evidence for the GraGE/GraCA AAAI direction:

```text
Training-dynamics-derived edge signals provide information beyond static feature
similarity, especially in the feature-ambiguous region where harmful edges have
high endpoint feature similarity.
```

This experiment should explain why GraGE-Hybrid can improve over Feature-only
instead of only reporting that it improves.

This is an AAAI-readiness mechanism diagnostic. It must test whether training
dynamics contain residual information beyond feature cosine. `bad_edge_mask` is
allowed only for evaluation metrics, never for edge score construction, model
selection, or practical pruning decisions.

## Repository Context

Read these files first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_tables_grage_hybrid/GRAGE_HYBRID_DECISION_REPORT.md`
- `paper_tables_grage_hybrid/edge_detection_f1.csv`
- `paper_tables_grage_hybrid/noise_type_breakdown.csv`
- `src/grage/hybrid_score.py`
- `src/grage/edge_gate_influence.py`
- `src/grage/unrolled_hypergradient.py`
- `src/eval/noise_injection.py`
- `scripts/analyze_edge_scores.py`
- `scripts/run_grage_hybrid_sweep.py`

## Tasks

1. Inspect current edge score diagnostics.
2. Add or update a diagnostics script, preferably:

   ```text
   scripts/analyze_grage_mechanisms.py
   ```

   The script should compute edge-level score tables for:

   - `feature_risk = 1 - cosine(x_u, x_v)`
   - raw dynamic edge-gate gradient
   - positive gradient component
   - negative gradient component
   - current best hybrid score:
     `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`
   - shuffled-gradient hybrid score, where the dynamic gradient is randomly
     permuted before combining with feature risk

3. Run diagnostics on:

   - datasets: `Cora`, `CiteSeer`, `PubMed`
   - noise types: `feature_similar_cross_class`, `cross_class_oracle`,
     `low_feature_similarity`
   - seeds: `0..9`
   - noise ratio: `0.3`

4. For each dataset/noise/seed, compute:

   - global ROC-AUC for each score against `bad_edge_mask`
   - precision/recall/F1 at prune ratio `0.2`
   - Spearman correlation between feature risk and dynamic scores
   - AUC within feature-similarity bins, especially the most feature-similar
     quartile or decile
   - residual diagnostic: regress or rank-residual dynamic/hybrid signal after
     feature risk, then evaluate whether residuals still detect harmful edges
   - score shuffling ablation: compare hybrid with real dynamic gradients vs
     hybrid with shuffled dynamic gradients
   - frozen/inner-channel diagnostic where feasible: preserve the same training
     schedule and scoring split while removing or shuffling the graph-channel
     edge-gradient contribution, so gains are not misattributed to a generic
     inner-loop training effect

5. Produce paper-friendly tables:

   ```text
   experiments/2026-06-04-dynamics-mechanism-diagnostics/logs/tables/
   ```

   Required tables:

   - `global_signal_auc.csv`
   - `feature_bin_auc.csv`
   - `shuffle_ablation.csv`
   - `residual_signal.csv`
   - `correlation_summary.csv`

6. Produce at least one plot if dependencies are available:

   - feature-similarity bin AUC plot, or
   - score distribution plot for feature-similar cross-class noise.

   Save plots under:

   ```text
   experiments/2026-06-04-dynamics-mechanism-diagnostics/logs/figures/
   ```

7. Write a decision report. It must answer:

   - Does dynamic signal contain residual information beyond feature risk?
   - Is the effect concentrated in `feature_similar_cross_class`?
   - Does shuffled-gradient hybrid lose the advantage?
   - Does the frozen/inner-channel diagnostic change the interpretation?
   - What claim can safely go into an AAAI paper?
8. Write `failure_analysis.md` describing negative or ambiguous mechanism
   evidence, especially cases where dynamic scores are correlated with feature
   risk but do not improve residual bad-edge detection.

## Commands

Run at least:

```bash
python -m py_compile src/grage/hybrid_score.py src/grage/edge_gate_influence.py
pytest -q tests/test_edge_gate.py tests/test_scoring.py
```

After implementing the diagnostics script, run a one-case smoke test:

```bash
python scripts/analyze_grage_mechanisms.py \
  --output_dir experiments/2026-06-04-dynamics-mechanism-diagnostics/logs/smoke \
  --datasets Cora \
  --noise_types feature_similar_cross_class \
  --seeds 0 \
  --noise_ratio 0.3 \
  --prune_ratio 0.2
```

Then run the full diagnostics matrix described in Tasks.

## Output Contract

Write:

- `experiments/2026-06-04-dynamics-mechanism-diagnostics/result.md`
- `experiments/2026-06-04-dynamics-mechanism-diagnostics/metrics.json`
- `experiments/2026-06-04-dynamics-mechanism-diagnostics/logs/`
- `experiments/2026-06-04-dynamics-mechanism-diagnostics/failure_analysis.md`

The `metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-dynamics-mechanism-diagnostics",
  "status": "completed|partial|failed",
  "residual_dynamic_signal_supported": true,
  "feature_similar_bin_real_vs_shuffled_delta": 0.0,
  "best_signal_in_feature_similar_bin": "...",
  "mean_residual_auc": 0.0,
  "mean_dynamic_feature_spearman": 0.0,
  "inner_channel_diagnostic": "...",
  "failure_modes": [],
  "num_cases": 0,
  "tables": [],
  "figures": [],
  "notes": "..."
}
```

If diagnostics do not support the mechanism claim, state that clearly and
recommend how the paper framing should change.
