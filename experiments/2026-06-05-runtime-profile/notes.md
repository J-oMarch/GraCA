# Notes: Runtime Profile

This experiment addresses the runtime reviewer risk only.

Current paper table reports raw CSV end-to-end runtime:

- confirm20 FSCC: StabilityResidual `7.93s` vs Feature-only `2.38s`, about
  `3.33x`, for `+1.59 pp`.
- GSL audit: StabilityResidual is about `4.15x` Feature-only.

Missing split:

- feature scoring
- probe/view training
- gradient confidence collection
- stability scoring
- pruning
- downstream retraining
- inference/evaluation

The profiler should not be used to claim efficiency superiority. It should only
support an accuracy-cost tradeoff table.

Local Codex verification on 2026-06-05:

- `python scripts/run_runtime_profile.py --help` works without ML dependencies.
- `python -m py_compile scripts/run_runtime_profile.py` passes.
- Full execution still requires a Python environment with `torch`,
  `torch_geometric`, and the repository ML dependencies installed.
