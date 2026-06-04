# Experiment: StabilityResidual Ablation and Stronger Confirmation

## Objective

StabilityResidual-GraGE currently supports a positive GraGE path, but the paper
is not AAAI-final yet. The main remaining risks are:

- the selected candidate uses a frozen-gradient control;
- no-gradient variants are close;
- residualization has not been directly ablated against raw stability;
- dropout schedule and number of graph views may be overfit to the search;
- the validation used 10 seeds and should be strengthened before paper claims.

This experiment performs the paper-facing ablations and stronger confirmation
needed to decide whether StabilityResidual can become the main method.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/method.md`
- `paper_draft/theory.md`
- `paper_draft/experiments.md`
- `experiments/2026-06-04-stability-channel-rebuild/result.md`
- `experiments/2026-06-04-stability-channel-rebuild/metrics.json`
- `experiments/2026-06-04-stability-channel-rebuild/failure_analysis.md`

Relevant code:

- `src/grage/adaptive_score.py`
- `scripts/run_adaptive_grage_search.py`
- `tests/test_adaptive_score.py`

## Tasks

1. Add ablation support for StabilityResidual:
   - raw stability without residualization;
   - residualized stability;
   - feature-only + shuffled residual;
   - no-gradient confidence;
   - real-gradient confidence;
   - shuffled-gradient confidence;
   - frozen-gradient confidence;
   - dropout schedules: `[0, .05, .10]`, `[0, .10, .15, .20, .30]`,
     `[0, .20, .35]`;
   - number of views: 3, 5, 7 where runtime permits.
2. Strengthen confirmation on homophilic citation datasets:
   - Cora, CiteSeer, PubMed
   - `feature_similar_cross_class`
   - `low_feature_similarity`
   - `degree_aligned_random`
   - at least 20 seeds for the selected StabilityResidual candidate and core
     baselines
3. Baselines:
   - Feature-only
   - GCN-Jaccard
   - Random-Matched
   - DegreeAwareRandom
   - GraGE-Hybrid
   - MCGC
4. Add paired statistics:
   - mean/std
   - delta vs Feature-only
   - paired t-test and Wilcoxon where available
   - Cohen's d
   - win rate
   - runtime ratio
5. Add residual diagnostics:
   - residual-feature correlation;
   - projection ratio;
   - residual AUC;
   - raw stability AUC;
   - feature risk AUC;
   - shuffled residual control.
6. Do not use validation/test/oracle labels or `bad_edge_mask` for scoring.
   `bad_edge_mask` is diagnostic only.

## Decision Rules

The main claim is paper-facing only if:

- StabilityResidual improves FSCC by `>= +0.5 pp` over Feature-only on the
  20-seed confirmation;
- win rate is clearly above `0.5`;
- no LFS/DAR degradation worse than `-0.5 pp`;
- residualized stability beats raw stability or clearly explains why raw is
  sufficient;
- shuffled residual is not competitive;
- gradient confidence is either shown useful or explicitly demoted to auxiliary.

If these conditions fail, write a method-rebuild recommendation rather than
adding more tables.

## Suggested Commands

Run code checks:

```bash
python -m py_compile src/grage/adaptive_score.py scripts/run_adaptive_grage_search.py
pytest -q tests/test_adaptive_score.py
```

Then run staged commands such as:

```bash
mkdir -p experiments/2026-06-04-stability-ablation-confirmation/logs
python scripts/run_adaptive_grage_search.py \
  --mode stability_ablation \
  --output_dir experiments/2026-06-04-stability-ablation-confirmation/logs/ablation
python scripts/run_adaptive_grage_search.py \
  --mode stability_confirm20 \
  --output_dir experiments/2026-06-04-stability-ablation-confirmation/logs/confirm20
```

If runtime is too high, prioritize:

1. 20-seed FSCC/LFS/DAR confirmation for selected candidate vs baselines.
2. Raw vs residual stability ablation.
3. Gradient confidence controls.
4. Dropout/views sensitivity.

Document any skipped stage and why.

## Required Output Contract

Write:

- `experiments/2026-06-04-stability-ablation-confirmation/result.md`
- `experiments/2026-06-04-stability-ablation-confirmation/metrics.json`
- `experiments/2026-06-04-stability-ablation-confirmation/failure_analysis.md`
- `experiments/2026-06-04-stability-ablation-confirmation/logs/`

`metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-stability-ablation-confirmation",
  "status": "completed_or_failed",
  "primary_claim_supported": false,
  "best_method": "",
  "fscc_delta_vs_feature_only_pp": null,
  "fscc_p_value": null,
  "fscc_win_rate": null,
  "fscc_effect_size": null,
  "lfs_delta_vs_feature_only_pp": null,
  "dar_delta_vs_feature_only_pp": null,
  "raw_vs_residual_delta_pp": null,
  "shuffled_residual_delta_pp": null,
  "gradient_confidence_conclusion": "",
  "dropout_schedule_conclusion": "",
  "num_result_rows": 0,
  "runtime_summary": "",
  "claim_supported": "supported|unsupported|partial|failed"
}
```

`result.md` must include a paper-facing ablation table, stronger confirmation
table, statistical tests, runtime, and a clear recommendation. `failure_analysis.md`
must cover dataset-specific failures, residualization failures, gradient-control
risks, and whether the method is AAAI-ready or still needs a rebuild.
