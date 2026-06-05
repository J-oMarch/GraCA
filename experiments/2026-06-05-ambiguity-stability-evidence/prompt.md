# Experiment: StabilityResidual Ambiguity and Alignment Evidence

## Objective

StabilityResidual-GraGE has a confirmed positive result against Feature-only on
homophilic citation `feature_similar_cross_class` noise, plus documented
heterophily and GSL-positioning boundaries. This experiment adds the missing
paper-facing P0/P1 evidence:

1. P0: show whether the gain comes from feature-defined ambiguity regions rather
   than uniform/random score diversification.
2. P1: show whether prediction stability is an aligned edge-quality signal
   beyond confidence, random residuals, shuffled residuals, and node-permuted
   stability.

This experiment must not introduce a new main method. The only selected main
method is:

```text
StabilityResidual-v5-dp0.15-grad-frozen
```

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/method.md`
- `paper_draft/theory.md`
- `paper_draft/experiments.md`
- `paper_draft/limitations.md`
- `paper_draft/rebuttal_risks.md`
- `experiments/2026-06-04-stability-channel-rebuild/result.md`
- `experiments/2026-06-04-stability-channel-rebuild/metrics.json`
- `experiments/2026-06-04-stability-ablation-confirmation/result.md`
- `experiments/2026-06-04-stability-ablation-confirmation/metrics.json`
- `experiments/2026-06-04-stability-heterophily-regime/result.md`
- `experiments/2026-06-04-stability-gsl-baseline-audit/result.md`

Relevant code:

- `src/grage/adaptive_score.py`
- `src/grage/stats.py`
- `scripts/run_adaptive_grage_search.py`
- `scripts/run_gsl_baseline_audit.py`
- `tests/test_adaptive_score.py`

## Execution Directive

This experiment is already authorized for tmux execution. Do not ask the user to
submit it again. Implement the needed tracked runner or runner mode, run the
experiment, and write the required outputs.

Do not depend on untracked local scripts. If helper code is needed, add it as a
tracked script under `scripts/` or as tracked functions under `src/`.

## Scope Guardrails

Allowed:

- Add a focused diagnosis runner for P0/P1 evidence.
- Reuse StabilityResidual scoring code from `src/grage/adaptive_score.py`.
- Add compact tests or smoke checks for new bucket/permutation utilities.
- Update only experiment outputs and any tracked code required to run the
  experiment.

Not allowed:

- Restart GraGE-Hybrid, MCGC, or Selective-MCGC as candidate methods.
- Create a new main method.
- Run open-ended novelty search.
- Add large heterophily or GSL experiments.
- Use labels, validation labels, test labels, bad-edge masks, oracle labels, or
  downstream outcomes to define ambiguity buckets or practical edge scores.

Labels and `bad_edge_mask` are diagnostic only after scores, buckets, and prune
masks have already been computed.

## Main Matrix

Datasets:

- `Cora`
- `CiteSeer`
- `PubMed`

Noise types:

- primary: `feature_similar_cross_class`
- controls: `low_feature_similarity`, `degree_aligned_random`

Seeds:

- 20 seeds for the primary evidence matrix.

Pruning:

- matched pruning budget, same as prior StabilityResidual experiments;
- keep existing minimum-degree protection;
- no-leak scoring unchanged.

## P0: Ambiguity Contribution Analysis

Implement feature-defined Low / Medium / High ambiguity buckets.

Bucket definition requirements:

- Buckets must use feature-derived signals only.
- Forbidden for bucket definition: labels, stability, bad-edge mask, oracle
  labels, validation/test labels, or any downstream outcome.
- High-Ambiguity means feature-derived scores are close to the pruning decision
  boundary, where feature similarity provides weak discrimination between
  likely good and bad edges.
- A suitable default is to compute the Feature-only risk score
  `R_f(e) = 1 - cosine(x_u, x_v)`, find the matched-budget pruning threshold
  under the same degree constraints when feasible, and bucket edges by absolute
  distance to that feature-only decision boundary:
  - High: closest third to the boundary;
  - Medium: middle third;
  - Low: farthest third.
- If exact degree-constrained boundary extraction is awkward, use a documented
  no-label feature-risk quantile approximation that is consistent across
  datasets and seeds.

For each dataset/noise/seed/method, report:

- bucket-level bad-edge precision, recall, F1;
- bucket-level feature-risk AUC, raw stability AUC, residual AUC;
- bucket-level prune counts and bad-prune counts;
- overlap between Feature-only and StabilityResidual pruned edges by bucket;
- changed-prune attribution:
  - edges pruned by StabilityResidual but not Feature-only;
  - edges pruned by Feature-only but not StabilityResidual;
  - bad-edge enrichment among changed prunes by bucket.

Run bucket-gated variants:

- `Feature-only`
- `Feature+Residual-LowOnly`
- `Feature+Residual-MediumOnly`
- `Feature+Residual-HighOnly`
- `Feature+Stability`
- `StabilityResidual-v5-dp0.15-grad-frozen`

The required P0 conclusion must estimate how much of the
Feature-only to StabilityResidual gain is explained by High-Ambiguity bucket
changes, rather than random perturbation or uniform score diversification.

## P1: Stability Validation and Alignment Destruction

Run the following validation matrix:

- `Feature-only`
- `Feature+Confidence`
- `Feature+Stability`
- `Feature+Random Stability`
- `Feature+Shuffled Stability`
- `Feature+Permuted Stability`

Variant definitions:

- `Feature+Confidence`: replace node instability with feature-score-compatible
  node confidence/uncertainty derived from the same multi-view predictions,
  then convert to edge scores through the same residualization and combination
  pipeline.
- `Feature+Stability`: use real aligned prediction-stability residuals, without
  presenting this as a new method name. It is the P1 diagnostic form of the
  StabilityResidual signal.
- `Feature+Random Stability`: generate a random stability residual control with
  the same shape and rank-normalization pipeline.
- `Feature+Shuffled Stability`: shuffle the real edge-level stability residual
  across edges, preserving the edge-score marginal distribution while destroying
  edge alignment.
- `Feature+Permuted Stability`: preserve the real node-stability value
  distribution but randomly permute node-to-stability assignment before
  converting node stability into edge scores. This is the alignment destruction
  test: it checks whether gains depend on the correct node-stability alignment,
  not merely the residual value distribution.

For all variants, report:

- mean/std test accuracy;
- delta vs Feature-only;
- paired t-test and Wilcoxon where available;
- Cohen's d;
- win rate;
- bad-edge precision/recall/F1;
- runtime;
- residual-feature correlation and projection ratio where applicable.

Required P1 conclusion:

- If `Feature+Stability` beats confidence/random/shuffled/permuted controls,
  write that prediction stability acts as aligned edge-quality evidence.
- If shuffled or permuted controls are competitive, write this as a reviewer
  risk and narrow the claim accordingly.

## Theory and Paper Notes

Do not edit the final paper draft unless the experiment naturally produces a
small `paper_update.md` under this experiment directory. The final paper
integration will be handled after reviewing the result.

The paper-facing theory should use:

- Definition: Feature Ambiguity Region.
- Definition: Stability Residual.
- Proposition: residualized stability can improve edge ranking inside
  feature-defined ambiguity regions when aligned with residual edge quality.
- Proof Sketch.

Avoid theorem-strength wording. Do not claim universal graph learning,
heterophily success, or optimal graph structure.

P2/P3 handling:

- Use existing heterophily failure evidence from
  `2026-06-04-stability-heterophily-regime`.
- Use existing GSL proxy positioning from
  `2026-06-04-stability-gsl-baseline-audit`.
- Do not run new large heterophily or GSL experiments.

## Suggested Commands

Run local/remote checks after implementation:

```bash
python -m py_compile src/grage/adaptive_score.py src/grage/stats.py
python -m py_compile scripts/run_adaptive_grage_search.py
python -m py_compile scripts/run_ambiguity_stability_evidence.py
pytest -q tests/test_adaptive_score.py
```

Suggested experiment commands if a new runner is added:

```bash
mkdir -p experiments/2026-06-05-ambiguity-stability-evidence/logs
python scripts/run_ambiguity_stability_evidence.py \
  --mode smoke \
  --output_dir experiments/2026-06-05-ambiguity-stability-evidence/logs/smoke
python scripts/run_ambiguity_stability_evidence.py \
  --mode full \
  --output_dir experiments/2026-06-05-ambiguity-stability-evidence/logs/full
```

If you instead add a mode to an existing runner, document the exact command in
`result.md`.

## Required Output Contract

Write:

- `experiments/2026-06-05-ambiguity-stability-evidence/result.md`
- `experiments/2026-06-05-ambiguity-stability-evidence/metrics.json`
- `experiments/2026-06-05-ambiguity-stability-evidence/failure_analysis.md`
- `experiments/2026-06-05-ambiguity-stability-evidence/logs/`

`metrics.json` must include:

```json
{
  "exp_id": "2026-06-05-ambiguity-stability-evidence",
  "status": "completed_or_failed",
  "primary_claim_supported": false,
  "p0_ambiguity_claim_supported": false,
  "p1_alignment_claim_supported": false,
  "best_method": "",
  "fscc_delta_vs_feature_only_pp": null,
  "fscc_p_value": null,
  "fscc_win_rate": null,
  "high_ambiguity_gain_share": null,
  "high_ambiguity_changed_prune_enrichment": null,
  "feature_stability_vs_confidence_delta_pp": null,
  "feature_stability_vs_random_delta_pp": null,
  "feature_stability_vs_shuffled_delta_pp": null,
  "feature_stability_vs_permuted_delta_pp": null,
  "permuted_control_competitive": null,
  "shuffled_control_competitive": null,
  "num_result_rows": 0,
  "claim_recommendation": "",
  "failure_modes": []
}
```

`result.md` must include:

- executive summary;
- P0 ambiguity bucket definition and leakage audit;
- P0 contribution and bucket-gated variant tables;
- P1 validation and alignment-destruction tables;
- paired statistics and runtime;
- clear answers:
  1. Does the result support the current claim?
  2. Does it strengthen the AAAI story?
  3. Does it reduce reviewer risk?
  4. Does it add failure evidence?
  5. Does the claim need to shrink?

`failure_analysis.md` must cover:

- whether high-ambiguity evidence failed or passed;
- whether confidence/random/shuffled/permuted controls remain competitive;
- leakage risks;
- dataset-specific weaknesses;
- implications for `limitations.md` and `rebuttal_risks.md`.

