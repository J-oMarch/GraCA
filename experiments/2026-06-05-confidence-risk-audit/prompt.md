# Experiment: Confidence Risk Audit for StabilityResidual-GraGE

## Objective

Reduce the reviewer risk that prediction stability is merely confidence or
uncertainty under another name.

The target question is:

```text
Does prediction-stability-derived edge evidence remain useful after controlling
for confidence?
```

This is a diagnostic and paper-risk-reduction experiment for the existing main
method `StabilityResidual-v5-dp0.15-grad-frozen`. Do not search for a new main
method. Do not restart GraGE-Hybrid, MCGC, or open-ended module discovery.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/readiness_audit.md`
- `paper_draft/rebuttal_risks.md`
- `paper_draft/limitations.md`

Relevant implementation and results:

- `src/grage/adaptive_score.py`
- `scripts/run_confidence_risk_audit.py`
- `scripts/run_ambiguity_stability_evidence.py`
- `experiments/2026-06-05-ambiguity-stability-evidence/result.md`
- `experiments/2026-06-05-ambiguity-stability-evidence/metrics.json`
- `experiments/2026-06-05-ambiguity-stability-evidence/logs/full/results.csv`

## Tasks

1. Audit the existing P1 evidence.
   - Confirm that `Feature+Stability` beats random, shuffled, and node-permuted
     stability controls.
   - Confirm that `Feature+Confidence` remains close and is the unresolved
     reviewer risk.

2. Use or extend the diagnostic runner without changing the selected main method.
   The runner must export edge-level diagnostics for the FSCC matrix over Cora,
   CiteSeer, and PubMed. For each dataset/seed, write per-edge rows containing:
   - dataset
   - seed
   - edge endpoints or stable edge id
   - feature risk
   - feature similarity
   - node/edge confidence score used by `Feature+Confidence`
   - raw stability score
   - residualized stability score
   - final StabilityResidual score
   - Feature-only prune indicator
   - StabilityResidual prune indicator
   - Confidence-control prune indicator
   - bad-edge indicator for diagnostics only
   - ambiguity bucket if available

3. Implement confidence-controlled analyses:
   - Confidence-matched analysis: compare stability-selected and
     confidence-selected edges after matching or stratifying by confidence.
   - Same-confidence bucket analysis: split edges into confidence quantiles and
     report whether residual stability still improves bad-edge ranking/pruning
     within each bucket, especially within High-Ambiguity edges.
   - Edge-quality AUC analysis: report AUC for feature risk, confidence,
     raw stability, residualized stability, and combined StabilityResidual, both
     globally and within confidence buckets.
   - Partial-correlation or regression-style diagnostic if feasible: estimate
     whether residual stability predicts bad edges after controlling for feature
     risk and confidence.

4. Report paired accuracy only as supporting context.
   The primary goal is edge-quality evidence beyond confidence, not another
   accuracy search.

5. Leakage audit:
   - Practical scores must use training labels only.
   - Validation labels, test labels, oracle labels, and `bad_edge_mask` may be
     used only after scoring for diagnostics.
   - Bucket construction must not use labels or `bad_edge_mask`.

6. Update paper-facing risk files if the evidence is clear:
   - `paper_draft/rebuttal_risks.md`
   - `paper_draft/limitations.md`
   - optionally `paper_draft/readiness_audit.md`

   If confidence explains most of the stability signal, write that plainly and
   shrink the claim.

## Commands

Start with a smoke run:

```bash
python scripts/run_confidence_risk_audit.py --mode smoke \
  --output_dir experiments/2026-06-05-confidence-risk-audit/logs/smoke
```

If the smoke run passes, run the full diagnostic:

```bash
python scripts/run_confidence_risk_audit.py --mode full \
  --output_dir experiments/2026-06-05-confidence-risk-audit/logs/full
```

Run focused tests or syntax checks for modified code:

```bash
python -m py_compile scripts/run_confidence_risk_audit.py
python -m pytest tests/test_adaptive_score.py -q
```

## Output Contract

Write:

- `experiments/2026-06-05-confidence-risk-audit/result.md`
- `experiments/2026-06-05-confidence-risk-audit/metrics.json`
- `experiments/2026-06-05-confidence-risk-audit/logs/`

Also write, when generated:

- `experiments/2026-06-05-confidence-risk-audit/logs/full/edge_diagnostics.csv`
- `experiments/2026-06-05-confidence-risk-audit/logs/full/confidence_bucket_summary.csv`
- `experiments/2026-06-05-confidence-risk-audit/logs/full/auc_summary.csv`

`metrics.json` must include:

- `status`
- `confidence_risk_reduced`
- `stability_not_confidence_only`
- `stability_vs_confidence_delta_pp`
- `matched_bad_edge_rate_delta`
- `same_confidence_bucket_auc_delta`
- `residual_stability_auc_after_confidence_control`
- `claim_recommendation`
- `failure_modes`

The final `result.md` must answer:

1. Is stability distinguishable from confidence?
2. Does the distinction hold in High-Ambiguity FSCC edges?
3. What should the paper claim?
4. What should be admitted in limitations?
