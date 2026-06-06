# Experiment: Runtime Profile for StabilityResidual-GraGE

## Objective

Build a paper-ready runtime breakdown for the fixed main method:

```text
StabilityResidual-v5-dp0.15-grad-frozen
```

This is not method search. It profiles Feature-only and StabilityResidual cost
components for the accuracy-cost tradeoff table.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/runtime_table.md`
- `paper_draft/reviewer_risk_audit.md`
- `scripts/run_runtime_profile.py`

Relevant implementation:

- `src/training/train_downstream.py`
- `src/grage/adaptive_score.py`
- `scripts/run_confidence_risk_audit.py`
- `scripts/run_ambiguity_stability_evidence.py`

## Tasks

1. Run the runtime profiler for the fixed methods only:
   - Feature-only
   - StabilityResidual-v5-dp0.15-grad-frozen

2. Report the following components:
   - feature scoring time
   - probe/model-view training time
   - gradient confidence collection time
   - stability scoring time
   - pruning time
   - downstream retraining time
   - final inference/evaluation time
   - total profiled time
   - extra overhead vs Feature-only
   - accuracy-cost tradeoff

3. Do not tune hyperparameters, change method definitions, add methods, or
   claim efficiency superiority.

4. If results are noisy, report variance and keep the claim as an approximate
   accuracy-cost tradeoff.

5. Update, if the run completes cleanly:
   - `paper_draft/runtime_table.md`
   - `paper_draft/reviewer_risk_audit.md`
   - `paper_draft/readiness_audit.md`
   - `paper_draft/aaai_readiness_score.md`

## Commands

Smoke:

```bash
python scripts/run_runtime_profile.py --mode smoke \
  --output_dir experiments/2026-06-05-runtime-profile/logs/smoke
```

Full profile:

```bash
python scripts/run_runtime_profile.py --mode full \
  --output_dir experiments/2026-06-05-runtime-profile/logs/full
```

Static checks:

```bash
python -m py_compile scripts/run_runtime_profile.py
python scripts/check_paper_claims.py
```

## Output Contract

Write:

- `experiments/2026-06-05-runtime-profile/result.md`
- `experiments/2026-06-05-runtime-profile/metrics.json`
- `experiments/2026-06-05-runtime-profile/logs/`

Expected tables:

- `experiments/2026-06-05-runtime-profile/logs/full/runtime_profile.csv`
- `experiments/2026-06-05-runtime-profile/logs/full/runtime_summary.csv`

`metrics.json` must include:

- `status`
- `runtime_ratio`
- `extra_overhead`
- `accuracy_delta_pp`
- `stability_probe_train_time`
- `stability_gradient_time`
- `stability_score_time`
- `stability_downstream_train_time`
- `stability_inference_eval_time`
- `claim_recommendation`

Final interpretation must be:

```text
accuracy-cost tradeoff, not efficiency superiority
```
