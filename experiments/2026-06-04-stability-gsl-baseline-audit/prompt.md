# Experiment: StabilityResidual GSL Baseline Audit

## Objective

StabilityResidual-GraGE now has strong evidence against Feature-only and local
baselines on homophilic citation FSCC, plus a documented heterophily boundary.
The remaining paper-facing gap is comparison or defensible positioning against
graph structure learning and robust graph cleaning baselines such as LDS, IDGL,
and ProGNN.

Core question:

```text
Can we add runnable, no-leak, matched-budget GSL/robust-graph baselines or a
clear feasibility audit that prevents overclaiming against the GSL literature?
```

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/related_work.md`
- `paper_draft/experiments.md`
- `paper_draft/rebuttal_risks.md`
- `experiments/2026-06-04-stability-ablation-confirmation/result.md`
- `experiments/2026-06-04-stability-heterophily-regime/result.md`

Relevant code:

- `scripts/run_adaptive_grage_search.py`
- `src/baselines/`
- `src/grage/adaptive_score.py`
- `tests/test_adaptive_score.py`

## Tasks

0. Execution directive: this experiment is already authorized for tmux
   execution. Do not ask the user to submit it. Run the audit and any feasible
   baseline experiments now.
1. Inspect existing baseline support in `src/baselines/` and runner scripts.
2. Determine which GSL/robust baselines are feasible without unsafe or
   non-reproducible external setup:
   - ProGNN-style robust graph cleaning;
   - IDGL-style iterative similarity graph learning;
   - LDS/bilevel GSL if an implementation already exists or can be added
     compactly;
   - existing GCN-Jaccard and degree-aware baselines as fallback.
3. Do not use labels beyond the same training labels allowed for the main
   method. Validation labels may be used only for normal model selection/early
   stopping, not edge oracle scoring.
4. If a baseline can be implemented compactly and safely, run it under matched
   pruning/reconstruction budget against:
   - Feature-only
   - StabilityResidual-v5-dp0.15-grad-frozen
   - GCN-Jaccard
   - MCGC
   - GraGE-Hybrid
5. Matrix:
   - Cora, CiteSeer, PubMed
   - `feature_similar_cross_class`
   - at least 10 seeds for any newly added expensive baseline
   - reuse prior 20-seed StabilityResidual numbers where appropriate, but mark
     reused rows clearly
6. If full LDS/IDGL/ProGNN reproduction is not feasible in this codebase, write
   a rigorous feasibility report:
   - what dependency or implementation barrier blocks it;
   - which existing baseline is the closest runnable proxy;
   - what claim wording is still defensible;
   - what must be added before camera-ready.
7. Update no paper files directly unless the experiment writes a proposed
   `paper_update.md`; Codex will integrate after review.

## Decision Rules

- If a runnable GSL/robust baseline beats StabilityResidual, the method claim
  must be narrowed or the method rebuilt.
- If StabilityResidual beats feasible GSL/robust baselines, add those rows to
  the paper-facing leaderboard.
- If GSL baselines are infeasible, the paper must say so honestly and avoid
  implying a complete comparison. The experiment should still provide a concrete
  implementation plan for the missing baseline.

## Suggested Commands

Run checks:

```bash
python -m py_compile src/grage/adaptive_score.py scripts/run_adaptive_grage_search.py
pytest -q tests/test_adaptive_score.py
```

Suggested experiment command if you add a runner mode:

```bash
mkdir -p experiments/2026-06-04-stability-gsl-baseline-audit/logs
python scripts/run_adaptive_grage_search.py \
  --mode stability_gsl_baselines \
  --output_dir experiments/2026-06-04-stability-gsl-baseline-audit/logs/gsl
```

If you use a focused script, document the exact command.

## Required Output Contract

Write:

- `experiments/2026-06-04-stability-gsl-baseline-audit/result.md`
- `experiments/2026-06-04-stability-gsl-baseline-audit/metrics.json`
- `experiments/2026-06-04-stability-gsl-baseline-audit/failure_analysis.md`
- `experiments/2026-06-04-stability-gsl-baseline-audit/logs/`

`metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-stability-gsl-baseline-audit",
  "status": "completed_or_failed",
  "runnable_gsl_baselines": [],
  "blocked_gsl_baselines": [],
  "primary_claim_supported": false,
  "best_method": "",
  "stability_vs_best_gsl_delta_pp": null,
  "stability_vs_best_gsl_p_value": null,
  "num_result_rows": 0,
  "claim_wording_recommendation": "",
  "next_implementation_steps": []
}
```

`result.md` must include:

- baseline inventory;
- implementation decisions;
- any runnable comparison table;
- blocked-baseline feasibility analysis;
- claim wording recommendation for paper;
- exact next steps for missing LDS/IDGL/ProGNN-style baselines.

`failure_analysis.md` must cover reproducibility risks, leakage risks,
dependency risks, and whether the current paper can claim comparison to GSL
baselines.
