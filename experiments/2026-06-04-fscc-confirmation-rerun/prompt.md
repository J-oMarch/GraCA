# Experiment: FSCC Matched-Budget Confirmation Rerun

## Execution Directive

This prompt is already being run by `scripts/run_exp.sh` inside the remote tmux
experiment `2026-06-04-fscc-confirmation-rerun`. The submit authorization has
already been granted by the user and the local Codex agent. Do **not** stop after
rewriting this prompt, do **not** ask the user to submit, and do **not** merely
prepare files.

Execute the experiment now. The existing `result.md` and `metrics.json` in this
directory are placeholder artifacts from a previous operational failure. You
must overwrite them with real experiment outputs, or with a concrete failed
status that explains the actual blocker after attempting the required commands.

## Objective

Recover the failed matched-budget confirmation experiment with a direct,
auditable runner. The experiment must decide whether any current practical GraGE
variant beats Feature-only under matched pruning budgets, with multi-seed
statistics and failure analysis.

Primary claim to test:

```text
Current practical GraGE variants improve downstream accuracy over Feature-only
on feature_similar_cross_class noise without label leakage.
```

Treat this as a confirmation/rerun, not a method-search task. Do not introduce a
new method unless needed to make existing runners callable.

## Repository Context

Read first:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `experiments/2026-06-04-fscc-hybrid-confirmation/result.md`
- `experiments/2026-06-04-adaptive-grage-search/result.md`
- `experiments/2026-06-04-dynamics-mechanism-diagnostics/result.md`
- `scripts/run_grage_hybrid_sweep.py`
- `scripts/run_adaptive_grage_search.py`
- `src/grage/hybrid_score.py`
- `src/grage/adaptive_score.py`

## Step-by-step Workflow

Follow these steps IN ORDER. Do NOT skip steps. After each step, verify success
before proceeding.

### Step 0: Verify code compiles and tests pass

```bash
python -m py_compile scripts/run_grage_hybrid_sweep.py
python -m py_compile scripts/run_adaptive_grage_search.py
python -m py_compile src/grage/hybrid_score.py
python -m py_compile src/grage/adaptive_score.py
pytest -q tests/test_edge_gate.py tests/test_scoring.py tests/test_adaptive_score.py tests/test_pruning.py
```

All must pass. If any fail, fix the issue before continuing.

### Step 1: Add missing method types to the runner

The existing `scripts/run_adaptive_grage_search.py` already supports these
method types: `feature_only`, `hybrid_baseline`, `faa_hybrid`, `mcgc`,
`selective_mcgc`.

It also has a built-in `Random-Matched` baseline via
`run_random_matched_baseline()`.

Two methods are MISSING and must be added as new method types in
`run_single_experiment()`:

1. **`DegreeAwareRandom`** — degree-aware random pruning with matched budget.
   Implementation exists in `src/baselines/random_pruning.py::run_degree_aware_random()`.
   Add a new method type `"degree_aware_random"` that:
   - Calls `run_degree_aware_random()` from `src/baselines/random_pruning.py`
   - Uses `edge_index_override=noisy_edge_index` to operate on the noisy graph
   - Extracts `test_acc`, `test_f1`, `val_acc` from the returned results dict
   - Computes `prune_mask` from the returned value
   - Evaluates bad edge detection and homophily like other methods

2. **`GCN-Jaccard`** — Jaccard similarity pruning with matched budget.
   Implementation exists in `src/baselines/similarity_pruning.py::run_jaccard_pruning()`.
   Add a new method type `"jaccard"` that:
   - Calls `run_jaccard_pruning()` from `src/baselines/similarity_pruning.py`
   - Uses `edge_index_override=noisy_edge_index` to operate on the noisy graph
   - Extracts `test_acc`, `test_f1`, `val_acc` from the returned results dict
   - Computes `prune_mask` from the returned value
   - Evaluates bad edge detection and homophily like other methods

Both methods need `prune_ratio` passed via `match_graca_ratio=prune_ratio` to
ensure matched budget.

After modifying, verify:
```bash
python -m py_compile scripts/run_adaptive_grage_search.py
```

### Step 2: Run smoke test

```bash
python scripts/run_adaptive_grage_search.py --mode smoke \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/smoke
```

This should run: Feature-only, FAA-Hybrid-as1.0-lp0.1-ln0.5, and
Random-Matched on Cora/feature_similar_cross_class/seed=0.

Verify the output CSV has at least 3 rows with valid test_acc values.

Then also add a manual smoke test for the new methods by creating a small
test script or running inline:

```python
# Quick inline test for new method types
import sys, os
sys.path.insert(0, '.')
# Verify imports work
from src.baselines.random_pruning import run_degree_aware_random
from src.baselines.similarity_pruning import run_jaccard_pruning
print("Import OK")
```

### Step 3: Run primary confirmation

This is the main experiment. Use `run_adaptive_grage_search.py` with a custom
method list.

Create a helper script `scripts/run_fscc_confirmation.py` that:

1. Imports from `run_adaptive_grage_search`:
   - `run_experiment_matrix`
   - `run_random_matched_baseline`
   - `DEFAULT_CONFIG`
   - `compute_feature_risk`, `compute_feature_similarity`
   - `train_model_for_grage`

2. Imports from baselines:
   - `run_degree_aware_random`
   - `run_jaccard_pruning`

3. Defines the method configs (note: Random-Matched is handled separately by
   `run_experiment_matrix` via `include_random_matched=True`):
   ```python
   METHODS = [
       {"name": "Feature-only", "type": "feature_only"},
       {"name": "GraGE-Hybrid-FO-posneg-lp0.1-ln0.5", "type": "hybrid_baseline",
        "lambda_pos": 0.1, "lambda_neg": 0.5, "score_ratio": 0.3},
       {"name": "MCGC-cw3.0-lp0.1-ln0.5", "type": "mcgc",
        "lambda_pos": 0.1, "lambda_neg": 0.5,
        "consistency_weight": 3.0, "score_ratio": 0.3,
        "checkpoint_fractions": [0.3, 0.5, 0.7, 0.9],
        "total_epochs": 200},
       {"name": "DegreeAwareRandom", "type": "degree_aware_random"},
       {"name": "GCN-Jaccard", "type": "jaccard"},
   ]
   ```

4. Runs the matrix:
   ```python
   # Primary: 3 datasets × 1 noise × 20 seeds × 6 methods = 360 runs
   # (5 methods in METHODS + 1 Random-Matched via include_random_matched)
   df_primary = run_experiment_matrix(
       datasets=["Cora", "CiteSeer", "PubMed"],
       noise_types=["feature_similar_cross_class"],
       noise_ratio=0.3,
       seeds=list(range(20)),
       downstream_model="GCN",
       prune_ratio=0.2,
       method_configs=METHODS,
       device=device,
       output_dir="experiments/2026-06-04-fscc-confirmation-rerun/logs/primary",
       include_random_matched=True,
   )
   ```

5. Saves results to:
   `experiments/2026-06-04-fscc-confirmation-rerun/logs/primary/results.csv`

Run:
```bash
python scripts/run_fscc_confirmation.py --stage primary \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/primary
```

This will take a while (360 runs). Use `time` to track runtime.

### Step 4: Run control regimes

Same script, different parameters:

```python
# Controls: 3 datasets × 3 noise × 10 seeds × 6 methods = 540 runs
df_controls = run_experiment_matrix(
    datasets=["Cora", "CiteSeer", "PubMed"],
    noise_types=["cross_class_oracle", "low_feature_similarity", "degree_aligned_random"],
    noise_ratio=0.3,
    seeds=list(range(10)),
    downstream_model="GCN",
    prune_ratio=0.2,
    method_configs=METHODS,
    device=device,
    output_dir="experiments/2026-06-04-fscc-confirmation-rerun/logs/controls",
    include_random_matched=True,
)
```

Run:
```bash
python scripts/run_fscc_confirmation.py --stage controls \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/controls
```

### Step 5: Run heterophily slice

```python
# Heterophily: 3 datasets × 2 noise × 5 seeds × 6 methods = 180 runs
df_hetero = run_experiment_matrix(
    datasets=["Texas", "Wisconsin", "Actor"],
    noise_types=["feature_similar_cross_class", "degree_aligned_random"],
    noise_ratio=0.3,
    seeds=list(range(5)),
    downstream_model="GCN",
    prune_ratio=0.2,
    method_configs=METHODS,
    device=device,
    output_dir="experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily",
    include_random_matched=True,
)
```

Run:
```bash
python scripts/run_fscc_confirmation.py --stage heterophily \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily
```

Note: If Texas, Wisconsin, or Actor datasets are not available, skip them
gracefully and note which were skipped. The `load_dataset` function may raise
an error for missing datasets — catch and continue.

### Step 6: Compute paired statistics

After all experiments complete, create `scripts/compute_fscc_stats.py` that:

1. Loads all result CSVs from:
   - `logs/primary/results.csv`
   - `logs/controls/results.csv`
   - `logs/heterophily/results.csv`

2. For each (dataset, noise_type, method) group vs Feature-only:
   - **Mean/std accuracy**: `groupby(["dataset","noise_type","method"])["test_acc"].agg(["mean","std","count"])`
   - **Paired delta (pp)**: For each seed, compute `method_acc - feature_only_acc`, then mean and std of deltas. Report in percentage points (multiply by 100).
   - **Paired t-test**: `scipy.stats.ttest_rel(method_accs, feature_only_accs)` per (dataset, noise_type)
   - **Wilcoxon signed-rank**: `scipy.stats.wilcoxon(method_accs, feature_only_accs)` where n >= 6
   - **Win rate**: fraction of seeds where method > feature_only
   - **Effect size (Cohen's d)**: `mean_delta / std_delta` for paired differences
   - **Runtime**: mean runtime per method

3. For GraGE-family methods, also compare vs shuffled/frozen controls if
   available in the results.

4. Generate summary tables:
   - `logs/tables/primary_fscc.csv` — main FSCC results
   - `logs/tables/control_regimes.csv` — control regime results
   - `logs/tables/heterophily.csv` — heterophily slice results
   - `logs/tables/paired_stats.csv` — all paired statistics

5. Print a clear summary to stdout showing:
   - Which methods beat Feature-only on FSCC (with p-values)
   - Which methods lose to Feature-only on control regimes
   - Overall verdict

Run:
```bash
python scripts/compute_fscc_stats.py \
    --primary_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/primary \
    --controls_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/controls \
    --heterophily_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/tables
```

### Step 7: Write result.md

Write `experiments/2026-06-04-fscc-confirmation-rerun/result.md` with:

1. **Executive summary**: One paragraph stating whether the primary claim is
   supported, unsupported, or regime-specific.

2. **Primary FSCC results table**:
   | Method | Cora Mean±Std | CiteSeer Mean±Std | PubMed Mean±Std | Overall Mean±Std |
   with paired delta vs Feature-only and p-value.

3. **Control regime results table**: Same format for each control noise type.

4. **Heterophily slice table**: Same format.

5. **Paired statistics summary**: For each method vs Feature-only on FSCC:
   - Mean delta (pp)
   - Paired t-test p-value
   - Wilcoxon p-value
   - Win rate
   - Cohen's d
   - Runtime

6. **Decision**: Based on decision rules in `docs/PROJECT_STATE.md`:
   - If GraGE/hybrid consistently beats Feature-only with statistical support
     (p < 0.05, win rate > 0.5, positive effect size) → "AAA direction viable"
   - If only beats Random-Matched → "needs stronger contribution"
   - If dataset-specific → "regime-specific"
   - If no support → "revise method or reframe paper"

7. **Failure modes**: List datasets/noise types where GraGE loses.

8. **Generated tables**: List all CSV files produced.

### Step 8: Write failure_analysis.md

Write `experiments/2026-06-04-fscc-confirmation-rerun/failure_analysis.md`:

1. For each (dataset, noise_type) where any GraGE method loses to Feature-only:
   - The exact delta and p-value
   - Diagnosis: is it signal quality (bad_edge_f1 low), over-pruning
     (actual_prune_ratio mismatch), degree constraints (high-degree nodes
     over-pruned), or runtime (method too slow)?

2. For the FSCC noise regime specifically:
   - Does the gradient signal add value beyond feature similarity?
   - Is the improvement consistent across seeds?
   - Is the improvement consistent across datasets?

3. Recommendations for the paper.

### Step 9: Write metrics.json

Write `experiments/2026-06-04-fscc-confirmation-rerun/metrics.json`:

```json
{
  "exp_id": "2026-06-04-fscc-confirmation-rerun",
  "status": "completed",
  "primary_claim_supported": <true|false>,
  "best_method": "<method name with highest mean FSCC accuracy>",
  "best_method_family": "<feature_only|hybrid|mcgc|random|jaccard>",
  "feature_similar_cross_class_delta_vs_feature_only_pp": <delta in pp>,
  "feature_similar_cross_class_p_value": <p-value>,
  "feature_similar_cross_class_win_rate": <win rate>,
  "feature_similar_cross_class_effect_size": <Cohen's d>,
  "low_feature_similarity_delta_vs_feature_only_pp": <delta in pp>,
  "num_result_rows": <total rows in all result CSVs>,
  "failure_modes": ["<list of (dataset, noise) where GraGE loses>"],
  "tables": ["logs/tables/primary_fscc.csv", ...],
  "notes": "<any caveats or observations>"
}
```

## Important Constraints

1. **No label leakage**: No oracle, validation labels, test labels, or
   `bad_edge_mask` may be used to compute practical edge scores. The
   `bad_edge_mask` is evaluation-only (for computing detection metrics AFTER
   pruning).

2. **Matched budget**: All methods must use the same `prune_ratio=0.2` and the
   same `min_degree=1`. The pruning function `prune_graph` handles budget
   enforcement.

3. **Reproducibility**: Use `set_seed(seed)` before each experiment. The
   existing runner already does this.

4. **No new methods**: Do not introduce new scoring methods. Use only:
   - Feature-only (existing)
   - GraGE-Hybrid-FO-posneg-lp0.1-ln0.5 (existing)
   - MCGC-cw3.0-lp0.1-ln0.5 (existing)
   - Random-Matched (existing)
   - DegreeAwareRandom (existing in baselines, needs wiring)
   - GCN-Jaccard (existing in baselines, needs wiring)

5. **Graceful failures**: If a dataset or method fails, log the error and
   continue. Do not abort the entire experiment for one failure.

## Commands to Run

```bash
# Compile check
python -m py_compile scripts/run_fscc_confirmation.py
python -m py_compile scripts/compute_fscc_stats.py

# Smoke test
python scripts/run_adaptive_grage_search.py --mode smoke \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/smoke

# Primary confirmation
python scripts/run_fscc_confirmation.py --stage primary \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/primary

# Control regimes
python scripts/run_fscc_confirmation.py --stage controls \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/controls

# Heterophily slice
python scripts/run_fscc_confirmation.py --stage heterophily \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily

# Compute statistics
python scripts/compute_fscc_stats.py \
    --primary_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/primary \
    --controls_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/controls \
    --heterophily_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/heterophily \
    --output_dir experiments/2026-06-04-fscc-confirmation-rerun/logs/tables
```

## Output Contract

Write:

- `experiments/2026-06-04-fscc-confirmation-rerun/result.md`
- `experiments/2026-06-04-fscc-confirmation-rerun/metrics.json`
- `experiments/2026-06-04-fscc-confirmation-rerun/failure_analysis.md`
- `experiments/2026-06-04-fscc-confirmation-rerun/logs/` (all CSVs and tables)
