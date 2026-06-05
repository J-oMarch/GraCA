# Ablation and Evidence Status

## Completed StabilityResidual Ablations

`2026-06-04-stability-ablation-confirmation` completed the core confirmation
and method ablations for the selected candidate.

- 20-seed FSCC confirmation: StabilityResidual-frozen beats Feature-only by
  `+1.59 pp` (`p<0.001`, win rate `0.83`, Cohen's d `0.70`).
- Control regimes: LFS `+0.55 pp` and DAR `+0.81 pp`; no material degradation.
- Raw vs residualized stability: raw `+1.25 pp`, residualized `+1.11 pp`,
  raw-vs-residualized difference `+0.14 pp` (`p=0.772`). Residualization is
  primarily for the feature-residual paper claim, not a large accuracy gain.
- Shuffled residual: `+0.87 pp`, competitive enough to report as a reviewer risk
  but weaker than real residual in the 5-seed ablation.
- Gradient confidence: no-gradient `+1.29 pp`, shuffled-gradient `+1.43 pp`,
  frozen-gradient `+1.79 pp`, real-gradient `+1.95 pp`. Gradient confidence is
  useful as an auxiliary mechanism, while prediction stability remains the
  primary signal.
- Dropout schedules: all tested schedules beat Feature-only; wider schedule
  `[0, 0.10, 0.15, 0.20, 0.30]` is strongest.
- Views: 3, 5, and 7 views all work; 3/5 are stronger than 7.

## Completed Boundary Evidence

- `2026-06-04-stability-heterophily-regime` shows heterophily is a failure
  boundary. StabilityResidual loses to Feature-only by `-1.14 pp` overall on
  Texas/Wisconsin/Actor and by `-2.89 pp` on heterophily FSCC.
- `2026-06-04-stability-gsl-baseline-audit` shows StabilityResidual is positive
  vs Feature-only but not superior to all GSL-inspired proxies. LDS-Proxy beats
  StabilityResidual by `+0.85 pp`; the paper must use "competitive with
  GSL-inspired proxies" and avoid any GSL-superiority claim.

## Completed P0/P1 Evidence

`2026-06-05-ambiguity-stability-evidence` completes the missing paper-facing
mechanism checks over Cora/CiteSeer/PubMed, FSCC/LFS/DAR, and 20 seeds.

- P0 Feature Ambiguity Evidence: Low/Medium/High buckets are defined from
  feature-derived signals only, by distance to the Feature-only pruning decision
  boundary. On FSCC, full StabilityResidual improves over Feature-only by
  `+2.06 pp` (`p=6.68e-10`, win rate `0.85`, Cohen's d `0.95`). Activating the
  residual only in the High-Ambiguity bucket gives `+1.68 pp` and explains
  `81.4%` of the full gain. Medium-only is weak (`+0.55 pp`, `p=0.112`), and
  Low-only is negative (`-0.09 pp`).
- Bucket diagnostics support the intended mechanism. In the High-Ambiguity
  bucket, StabilityResidual raises pruning F1 from `0.3425` to `0.4990`; SR-only
  changed prunes have `68.9%` bad-edge rate. Low/Medium buckets do not explain
  the main gain.
- P1 Stability Validation: aligned Feature+Stability beats random stability by
  `+1.73 pp`, shuffled stability by `+1.78 pp`, and node-permuted stability by
  `+1.63 pp`, all with `p<1e-8` and win rates `0.83-0.87`. Confidence is closer
  (`+0.31 pp`, `p=0.198`) and should be discussed as a related uncertainty
  control, but it does not erase the aligned-stability result.

These results support the claim:

```text
Prediction stability provides complementary edge-quality evidence beyond
feature similarity in homophilic feature-ambiguous citation regimes.
```
