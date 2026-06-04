# Experiment: StabilityResidual Heterophily and Regime Test

## Objective

`2026-06-04-stability-channel-rebuild` found the first positive GraGE candidate:
StabilityResidual-GraGE beats Feature-only by `+2.00 pp` on FSCC across
Cora/CiteSeer/PubMed. This experiment tests whether that claim survives outside
homophilic citation graphs and identifies the graph regimes where the method
fails.

Core question:

```text
Does prediction-stability residual scoring help graph evolution on heterophily
datasets, or should the paper restrict its claim to homophilic feature-ambiguous
regimes?
```

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/experiments.md`
- `paper_draft/limitations.md`
- `experiments/2026-06-04-stability-channel-rebuild/result.md`
- `experiments/2026-06-04-stability-channel-rebuild/failure_analysis.md`

Relevant code:

- `src/grage/adaptive_score.py`
- `scripts/run_adaptive_grage_search.py`
- `tests/test_adaptive_score.py`

## Tasks

1. Inspect the StabilityResidual implementation and current runner modes.
2. Add or reuse a heterophily/regime validation mode for:
   - Texas
   - Wisconsin
   - Actor
   - any other heterophily datasets already supported locally, if cheap
3. Use the selected candidate from the prior experiment:
   `StabilityResidual-v5-dp0.15-grad-frozen`.
4. Compare against:
   - Feature-only
   - GCN-Jaccard
   - Random-Matched
   - DegreeAwareRandom if supported on these datasets
   - MCGC
   - GraGE-Hybrid
5. Noise/regime matrix:
   - `feature_similar_cross_class`
   - `low_feature_similarity`
   - `degree_aligned_random`
   - clean/no-added-noise if the runner can support it without major surgery
6. Seeds:
   - at least 10 seeds for Texas/Wisconsin
   - at least 5 seeds for Actor if runtime is high, otherwise 10
7. Report graph-regime diagnostics:
   - edge homophily before/after
   - feature-risk AUC where `bad_edge_mask` exists (diagnostic only)
   - residual-feature correlation
   - residual AUC
   - pruning budget and runtime
8. Do not use validation/test/oracle labels or `bad_edge_mask` in scoring.

## Decision Rules

- If StabilityResidual beats Feature-only by `>= +0.5 pp` on heterophily with
  stable win rate and no major degradation, the paper can claim broader
  applicability.
- If it loses or is unstable, restrict the paper claim to homophilic or
  feature-ambiguous citation regimes and use this experiment as failure-mode
  evidence.
- Do not treat heterophily failure as fatal if citation FSCC remains strong, but
  it must be reported honestly.

## Suggested Commands

Run code checks:

```bash
python -m py_compile src/grage/adaptive_score.py scripts/run_adaptive_grage_search.py
pytest -q tests/test_adaptive_score.py
```

Then run your heterophily mode, for example:

```bash
mkdir -p experiments/2026-06-04-stability-heterophily-regime/logs
python scripts/run_adaptive_grage_search.py \
  --mode stability_heterophily \
  --output_dir experiments/2026-06-04-stability-heterophily-regime/logs/heterophily
```

If you add a focused runner instead, document exact commands in `result.md`.

## Required Output Contract

Write:

- `experiments/2026-06-04-stability-heterophily-regime/result.md`
- `experiments/2026-06-04-stability-heterophily-regime/metrics.json`
- `experiments/2026-06-04-stability-heterophily-regime/failure_analysis.md`
- `experiments/2026-06-04-stability-heterophily-regime/logs/`

`metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-stability-heterophily-regime",
  "status": "completed_or_failed",
  "primary_claim_supported": false,
  "heterophily_claim_supported": false,
  "best_method": "",
  "overall_delta_vs_feature_only_pp": null,
  "overall_p_value": null,
  "overall_win_rate": null,
  "fscc_delta_vs_feature_only_pp": null,
  "worst_dataset_delta_pp": null,
  "failure_modes": [],
  "regime_recommendation": "",
  "num_result_rows": 0,
  "runtime_summary": ""
}
```

`result.md` must include tables by dataset/noise/method, paired stats vs
Feature-only, regime diagnostics, and a paper-facing decision. `failure_analysis.md`
must explain whether failures come from heterophily, feature informativeness,
small splits, instability noise, or budget/degree effects.
