# Runtime Profile Result

## Executive Summary

- Mode: `full`.
- Rows: `30`.
- Runtime ratio (SR / FO): `4.22`.
- Extra overhead: `6.14s`.
- Accuracy delta: `1.69 pp`.
- Claim: **accuracy-cost tradeoff, not efficiency superiority**.

## Per-Component Timing

| Component | Feature-only | StabilityResidual |
| --- | --- | --- |
| Feature scoring | 0.01 ± 0.02s | 0.00 ± 0.00s |
| Probe/model-view training | 0.00s | 0.25 ± 0.15s |
| Gradient confidence | 0.00s | 3.11 ± 2.31s |
| Stability scoring | 0.00s | 2.88 ± 1.72s |
| Pruning | 1.51 ± 1.39s | 1.49 ± 1.42s |
| Downstream retraining | 0.39 ± 0.16s | 0.31 ± 0.13s |
| Inference/evaluation | 0.00s | 0.00s |
| Total | 1.91 ± 1.41s | 8.04 ± 5.56s |

## Per-Dataset Summary

| Dataset | FO_time | SR_time | FO_acc | SR_acc |
| --- | --- | --- | --- | --- |
| Cora | 0.90 ± 0.34s | 4.10 ± 1.82s | 0.6040 ± 0.0138 | 0.6384 ± 0.0169 |
| CiteSeer | 1.02 ± 0.37s | 4.67 ± 1.77s | 0.5576 ± 0.0167 | 0.5626 ± 0.0107 |
| PubMed | 3.80 ± 0.21s | 15.36 ± 0.98s | 0.6652 ± 0.0142 | 0.6764 ± 0.0082 |

## Interpretation

StabilityResidual-v5-dp0.15-grad-frozen adds multi-view graph training and
stability residualization on top of the Feature-only pipeline.  The extra
overhead is dominated by probe training and stability scoring (multi-view
predictions + node stability + residualization).  This is an accuracy-cost
tradeoff: the method buys `1.69 pp` at the cost of roughly
`4.22x` wall-clock time.

**Do not claim efficiency superiority.**  The profiler reports where time is
spent; it does not optimize runtime.

## Output Tables

- Raw profile: `experiments/2026-06-05-runtime-profile/logs/full/runtime_profile.csv`
- Summary: `experiments/2026-06-05-runtime-profile/logs/full/runtime_summary.csv`
