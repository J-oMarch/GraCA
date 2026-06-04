# Notes

The first confirmation experiment failed operationally because Claude exited
without producing real metrics. This rerun should avoid broad implementation
language and use a direct, script-driven workflow. The goal is not to invent a
new method, but to recover a reliable matched-budget comparison table for the
current best practical methods against Feature-only.

## Key Design Decisions

1. **Reuse existing runners**: `run_adaptive_grage_search.py` already handles
   most methods. Only `DegreeAwareRandom` and `GCN-Jaccard` need wiring as
   new method types.

2. **No new methods**: Only use existing implementations from the codebase.

3. **Script-driven**: Create two new scripts:
   - `scripts/run_fscc_confirmation.py` — runs the experiment matrix
   - `scripts/compute_fscc_stats.py` — computes paired statistics

4. **Step-by-step verification**: Each step must complete and be verified
   before proceeding to the next.

## Runtime Estimates

- Primary: ~360 runs × ~30s each ≈ 3 hours
- Controls: ~540 runs × ~30s each ≈ 4.5 hours
- Heterophily: ~180 runs × ~30s each ≈ 1.5 hours
- Total: ~9 hours

These are rough estimates. Actual runtime depends on dataset size and GPU
availability.

## Do Not Submit

Do not submit until explicitly requested.
