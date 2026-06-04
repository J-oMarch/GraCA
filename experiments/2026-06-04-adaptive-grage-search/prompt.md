# Experiment: Adaptive GraGE Method Search

## Objective

Search for a stronger AAAI-level GraGE method while keeping the original paper
direction:

```text
Graph structure is an evolvable object, and model training behavior can guide
automatic graph optimization for graph learning tasks.
```

You may substantially modify code and method details, but the final candidate
must remain no-leak and practical: no validation labels, test labels, or
`bad_edge_mask` may be used to compute edge scores.

Feature-only / similarity pruning is the main practical baseline. Do not
optimize only against Random-Matched or DropEdge. EdgeInfluence-Pseudo may be
mentioned only as a historical ablation and must not become the main method.

The goal is not just to improve a number. The goal is to identify a method with
a clean innovation story that can be packaged as a paper contribution.

## Repository Context

Read these files first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `paper_tables_grage_edge_gate/GRAGE_EDGE_GATE_DECISION_REPORT.md`
- `paper_tables_grage_hybrid/GRAGE_HYBRID_DECISION_REPORT.md`
- `src/grage/hybrid_score.py`
- `src/grage/edge_gate_influence.py`
- `src/grage/unrolled_hypergradient.py`
- `src/grage/pruning.py`
- `src/eval/noise_injection.py`
- `scripts/run_grage_hybrid_sweep.py`
- `tests/test_edge_gate.py`
- `tests/test_unrolled_hypergradient.py`
- `tests/test_scoring.py`

## Candidate Method Families

Implement and evaluate at least two of the following no-leak method families.
Prefer simple, explainable methods over opaque complexity.

### 1. Feature-Ambiguity-Adaptive Hybrid

Use stronger dynamic weighting only where feature similarity is high and static
feature risk is therefore less informative.

Example:

```text
score_e = R(feature_risk_e)
        + alpha(feature_similarity_e) * lambda_pos * R(relu(grad_e))
        - beta(feature_similarity_e) * lambda_neg * R(relu(-grad_e))
```

where `alpha` increases for feature-similar edges.

### 2. Prediction-Stability Edge Signal

Train a proxy model on the noisy graph, collect predictions from multiple
checkpoints or dropout passes, and score an edge by endpoint prediction
instability, disagreement, entropy, or margin change. This uses training behavior
without oracle labels.

### 3. Multi-Checkpoint Gradient Consistency

Compute edge-gate gradients at several training checkpoints and prefer edges
whose harmful gradient sign is stable across checkpoints. Penalize unstable or
protective edges.

### 4. Protective Edge-Aware Pruning

Make the negative gradient term explicit as edge protection rather than only a
subtractive score. For example, forbid pruning the top protective quantile unless
needed to satisfy the pruning budget.

## Tasks

1. Inspect current implementation and decide which candidate families to try.
2. Add method code under `src/grage/` and tests under `tests/` when appropriate.
3. Add a search runner, preferably:

   ```text
   scripts/run_adaptive_grage_search.py
   ```

   It should support a small search matrix and a validation matrix.

4. Search matrix:

   - datasets: `Cora`, `CiteSeer`
   - noise types: `feature_similar_cross_class`, `cross_class_oracle`
   - seeds: `0`, `1`, `2`
   - downstream model: `GCN`
   - baselines:
     - `Feature-only`
     - `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`
     - `Random-Matched`
   - candidates: at least two new method variants

5. Select one best candidate using:

   - primary score: paired delta over Feature-only on
     `feature_similar_cross_class`
   - constraint: no more than `0.005` mean degradation vs Feature-only on
     `low_feature_similarity` in validation
   - constraint: method is no-leak and reproducible

6. Validation matrix for selected candidate:

   - datasets: `Cora`, `CiteSeer`, `PubMed`
   - noise types: `feature_similar_cross_class`, `low_feature_similarity`,
     `degree_aligned_random`
   - seeds: `0..4`
   - downstream model: `GCN`
   - compare against:
     - `Feature-only`
     - current best `GraGE-Hybrid-FO-posneg-lp0.1-ln0.5`
     - `Random-Matched`

7. Build outputs under:

   ```text
   experiments/2026-06-04-adaptive-grage-search/logs/
   ```

   Required tables:

   - `candidate_search_results.csv`
   - `candidate_validation_results.csv`
   - `candidate_vs_feature_only_stats.csv`
   - `method_ablation_summary.csv`

8. Write a paper-claim decision report:

   - What new method was tried?
   - What is the clean algorithmic contribution?
   - Did it beat Feature-only and current GraGE-Hybrid?
   - Is it stable enough for a larger confirmation experiment?
   - What should be the next experiment if it succeeds?
9. Write `failure_analysis.md` explaining each rejected candidate family, where
   it lost to Feature-only, whether the issue is signal quality, budget
   matching, degree preservation, support/score split instability, or runtime,
   and what method redesign is justified.

## Commands

Run at least:

```bash
python -m py_compile src/grage/hybrid_score.py src/grage/edge_gate_influence.py src/grage/unrolled_hypergradient.py
pytest -q tests/test_edge_gate.py tests/test_unrolled_hypergradient.py tests/test_scoring.py tests/test_pruning.py
```

After implementing new code, run relevant new tests, then run a smoke search:

```bash
python scripts/run_adaptive_grage_search.py \
  --mode smoke \
  --output_dir experiments/2026-06-04-adaptive-grage-search/logs/smoke
```

Then run:

```bash
python scripts/run_adaptive_grage_search.py \
  --mode search \
  --output_dir experiments/2026-06-04-adaptive-grage-search/logs/search

python scripts/run_adaptive_grage_search.py \
  --mode validate \
  --output_dir experiments/2026-06-04-adaptive-grage-search/logs/validate
```

If you choose different commands, document why and keep equivalent coverage.

## Output Contract

Write:

- `experiments/2026-06-04-adaptive-grage-search/result.md`
- `experiments/2026-06-04-adaptive-grage-search/metrics.json`
- `experiments/2026-06-04-adaptive-grage-search/logs/`
- `experiments/2026-06-04-adaptive-grage-search/failure_analysis.md`

The `metrics.json` must include:

```json
{
  "exp_id": "2026-06-04-adaptive-grage-search",
  "status": "completed|partial|failed",
  "best_candidate": "...",
  "candidate_family": "...",
  "candidate_selected_for_confirmation": true,
  "delta_vs_feature_only_feature_similar_cross_class": 0.0,
  "delta_vs_current_hybrid_feature_similar_cross_class": 0.0,
  "low_feature_similarity_degradation_vs_feature_only": 0.0,
  "win_rate_vs_feature_only": 0.0,
  "effect_size_vs_feature_only": 0.0,
  "failure_modes": [],
  "num_candidate_methods": 0,
  "num_result_rows": 0,
  "new_files_or_modules": [],
  "claim_recommendation": "..."
}
```

If no candidate works, keep the negative result. Recommend the strongest
alternative paper framing and the next search direction.
