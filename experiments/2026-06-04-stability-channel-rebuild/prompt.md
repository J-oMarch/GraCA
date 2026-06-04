# Experiment: Stability-Channel GraGE Rebuild

## Objective

The current GraGE-Hybrid/MCGC evidence is negative: raw edge-gate gradients are
near-zero/noisy after feature-risk control, and the 20-seed FSCC confirmation
rerun shows Feature-only beats current GraGE variants. This experiment must not
continue small hyperparameter sweeps around the existing rank-normalized hybrid
or MCGC score.

Build and test a materially different no-leak training-dynamics channel:

```text
prediction-stability residual edge score + edge-gate gradient consistency/abstention
```

Core question:

```text
Can prediction stability under train-internal graph/model perturbations provide
edge-level graph evolution information beyond static feature similarity, while
edge-gate gradients act only as a confidence/abstention constraint?
```

Success requires a matched-budget improvement over Feature-only of at least
`+0.5 pp` on `feature_similar_cross_class`, with multi-seed stability, no
material degradation on control regimes, and shuffled/frozen controls showing
that the gain is not just feature similarity or budget/degree effects.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_draft/experiments.md`
- `paper_draft/related_work.md`
- `experiments/2026-06-04-fscc-confirmation-rerun/result.md`
- `experiments/2026-06-04-fscc-confirmation-rerun/failure_analysis.md`
- `experiments/2026-06-04-selective-mcgc-regime-gate/result.md`
- `experiments/2026-06-04-selective-mcgc-regime-gate/failure_analysis.md`

Relevant code:

- `src/grage/adaptive_score.py`
- `src/grage/edge_gate_influence.py`
- `src/grage/hybrid_score.py`
- `src/grage/pruning.py`
- `scripts/run_adaptive_grage_search.py`
- `scripts/analyze_adaptive_grage_search.py`
- `tests/test_adaptive_score.py`

## Non-Negotiable Constraints

- No label leakage. Do not use validation labels, test labels, oracle labels, or
  `bad_edge_mask` in any scoring or threshold-selection path. `bad_edge_mask`
  may be used only for diagnostics after scores are fixed.
- Feature-only, GCN-Jaccard, Random-Matched, and DegreeAwareRandom are strong
  baselines, not strawmen.
- EdgeInfluence-Pseudo is historical only. Do not promote it as the main method.
- Do not claim success from beating Random-Matched/DropEdge alone.
- Report mean, std, paired delta, p-value, effect size, win rate, runtime,
  shuffled/frozen controls, and failure modes.

## Method Requirements

Implement a new candidate method, tentatively:

```text
StabilityResidual-GraGE
```

Recommended design:

1. Train several no-leak probe models or checkpoints using only training labels.
   Use stochastic graph views and/or model stochasticity: edge dropout, feature
   dropout, checkpoint ensemble, or random seeds. Validation labels may be used
   for standard early stopping only if the existing training pipeline already
   does so, but never for edge scoring.
2. For each node, collect prediction distributions across views/checkpoints.
   Compute train-dynamics quantities such as prediction entropy, variance,
   Jensen-Shannon divergence across views, confidence stability, and endpoint
   prediction disagreement.
3. Convert node-level stability to an edge-level score for edge `e=(u,v)`.
   Candidate components:
   - endpoint prediction disagreement;
   - endpoint instability interaction;
   - graph-view sensitivity of endpoint predictions;
   - disagreement residual after controlling for feature cosine and degree.
4. Residualize or calibrate the stability score against static feature risk and
   simple structural priors. The paper-facing claim needs evidence beyond
   feature cosine, so include at least one residualized score test.
5. Use edge-gate gradient information only as confidence/abstention, not as a
   rank-normalized additive term. For example, abstain to Feature-only when
   gradient magnitude is near zero or signs are inconsistent across checkpoints.
6. Include controls:
   - shuffled stability score;
   - frozen/random prediction stability;
   - no-gradient-confidence version;
   - feature-only fallback;
   - degree-preserving/random matched pruning.

Keep the implementation scoped and auditable. Prefer extending existing
`src/grage/adaptive_score.py` and `scripts/run_adaptive_grage_search.py` or add
a focused runner if that is cleaner.

## Experiment Matrix

Run a staged matrix:

1. Smoke:
   - Cora
   - `feature_similar_cross_class`
   - 1 seed
   - Feature-only, StabilityResidual-GraGE, shuffled stability control

2. Search:
   - Cora, CiteSeer
   - `feature_similar_cross_class`, `low_feature_similarity`
   - at least 5 seeds
   - Feature-only, GCN-Jaccard, Random-Matched, DegreeAwareRandom,
     current GraGE-Hybrid, current MCGC, StabilityResidual variants, shuffled
     and frozen controls

3. Validation if search has a candidate with target FSCC delta `>= +0.5 pp`
   and no LFS degradation worse than `-0.5 pp`:
   - Cora, CiteSeer, PubMed
   - `feature_similar_cross_class`, `low_feature_similarity`,
     `degree_aligned_random`
   - at least 10 seeds
   - include paired stats and failure modes

If search fails, do not run a large validation matrix. Instead, write a detailed
negative result and mechanism diagnosis.

## Commands

Run relevant checks where dependencies exist:

```bash
python -m py_compile src/grage/adaptive_score.py scripts/run_adaptive_grage_search.py
pytest -q tests/test_adaptive_score.py
```

Then run the experiment stages you implement. Suggested paths:

```bash
mkdir -p experiments/2026-06-04-stability-channel-rebuild/logs
python scripts/run_adaptive_grage_search.py \
  --mode stability_smoke \
  --output_dir experiments/2026-06-04-stability-channel-rebuild/logs/smoke
python scripts/run_adaptive_grage_search.py \
  --mode stability_search \
  --output_dir experiments/2026-06-04-stability-channel-rebuild/logs/search
```

If the search gate passes, run:

```bash
python scripts/run_adaptive_grage_search.py \
  --mode stability_validate \
  --output_dir experiments/2026-06-04-stability-channel-rebuild/logs/validate
```

If you choose a new focused runner, document the exact commands in
`result.md`.

## Required Output Contract

Write all of the following:

- `experiments/2026-06-04-stability-channel-rebuild/result.md`
- `experiments/2026-06-04-stability-channel-rebuild/metrics.json`
- `experiments/2026-06-04-stability-channel-rebuild/failure_analysis.md`
- `experiments/2026-06-04-stability-channel-rebuild/logs/`

`metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-stability-channel-rebuild",
  "status": "completed_or_failed",
  "primary_claim_supported": false,
  "best_method": "",
  "best_method_family": "",
  "fscc_delta_vs_feature_only_pp": null,
  "fscc_p_value": null,
  "fscc_win_rate": null,
  "fscc_effect_size": null,
  "lfs_delta_vs_feature_only_pp": null,
  "control_degradation_worst_pp": null,
  "shuffled_control_delta_pp": null,
  "frozen_control_delta_pp": null,
  "feature_residual_signal_summary": "",
  "num_result_rows": 0,
  "runtime_summary": "",
  "failure_modes": [],
  "claim_supported": "supported|unsupported|partial|failed"
}
```

`result.md` must include:

- exact method formula and no-leak scoring path;
- implementation summary and modified files;
- smoke/search/validation table;
- paired comparison vs Feature-only;
- residual signal diagnostics beyond feature cosine;
- shuffled/frozen control interpretation;
- runtime comparison;
- decision: continue, revise, or abandon this stability channel.

`failure_analysis.md` must include:

- dataset-specific failures;
- whether instability signal is residual to feature similarity;
- whether edge-gate gradient confidence helps or merely filters noise;
- whether gains, if any, are explainable by budget/degree effects;
- a recommendation for the next paper-facing step.
