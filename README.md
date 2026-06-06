# Training-Dynamics-Guided Edge Disambiguation in Feature-Ambiguous Homophilic Graphs

This repository contains the executable code, experiment logs, and paper draft
materials for the StabilityResidual-GraGE project. The current project phase is
AAAI final assembly: the main method and evidence chain are fixed, no new main
experiments are planned, and the remaining work is paper polish, figure/table
styling, citation completion, and final reviewer-risk checking.

## Project Summary

The project studies a narrow graph denoising question:

> Can training-dynamics-derived prediction stability provide useful
> edge-quality evidence beyond static feature similarity?

The target setting is homophilic citation graphs with feature-similar
cross-class noise. In this regime, Feature-only pruning is a strong baseline,
but harmful edges can lie near the feature-derived pruning boundary and remain
ambiguous under static feature similarity alone.

## Current Paper Title

**Training-Dynamics-Guided Edge Disambiguation in Feature-Ambiguous Homophilic
Graphs**

## Current Main Method

The fixed main method is:

```text
StabilityResidual-v5-dp0.15-grad-frozen
```

StabilityResidual-GraGE trains stochastic graph views using training labels
only, measures node prediction instability, maps instability to edge scores,
residualizes the stability score against Feature-only risk, and combines the
residual with the feature prior under a matched pruning budget.

Historical GraGE-Hybrid, MCGC, Selective-MCGC, and raw edge-gate-gradient routes
are not main methods. They are retained only as historical negative evidence or
auxiliary local-sensitivity / confidence-abstention motivation.

## Current Claim

Supported claim:

```text
Prediction stability provides complementary edge-quality evidence beyond feature
similarity in homophilic feature-ambiguous citation regimes.
```

Unsupported claims:

- Universal graph learning.
- Works on all graph regimes.
- Heterophily robustness or heterophily success.
- Learns optimal graph structure.
- Outperforms GSL or full LDS/IDGL/ProGNN.
- Runtime or efficiency advantage.

## Core Experimental Results

### Main Confirmation

On Cora, CiteSeer, and PubMed under feature-similar cross-class noise (FSCC),
the fixed StabilityResidual method improves over Feature-only by:

- `+1.59 pp`
- `p<0.001`
- win rate `0.83`
- Cohen's d `0.70`

Control regimes do not show material degradation:

- Low-feature-similarity noise (LFS): `+0.55 pp`
- Degree-aligned random noise (DAR): `+0.81 pp`

Per-dataset FSCC effects are positive but uneven:

- Cora: `+3.19 pp`, 20/20 wins
- PubMed: `+1.05 pp`, `p=0.002`, win rate `0.80`
- CiteSeer: `+0.52 pp`, not individually significant, win rate `0.70`

## P0 / P1 Evidence Chain

### P0: Feature Ambiguity Contribution

The P0 analysis defines Low/Medium/High ambiguity buckets using only
feature-derived quantities: distance to the Feature-only pruning boundary.
Labels and injected bad-edge masks are diagnostic only.

FSCC results:

- Full StabilityResidual vs Feature-only: `+2.06 pp`
- `p=6.68e-10`
- win rate `0.85`
- Cohen's d `0.95`
- High-Ambiguity-only residual activation: `+1.68 pp`
- High-Ambiguity gain share: `81.4%`

High-Ambiguity pruning diagnostics:

- Feature-only pruning F1: `0.3425`
- StabilityResidual pruning F1: `0.4990`
- SR-only changed prunes bad-edge rate: `68.9%`

Interpretation: the gain is concentrated where Feature-only is least decisive,
not uniformly across all edges.

### P1: Stability Alignment

P1 tests whether aligned prediction stability matters rather than only the score
distribution or residualization pipeline.

On the FSCC matrix:

- Feature+Stability vs Feature+Random Stability: `+1.73 pp`, `p<1e-8`
- Feature+Stability vs Feature+Shuffled Stability: `+1.78 pp`, `p<1e-8`
- Feature+Stability vs Feature+Permuted Stability: `+1.63 pp`, `p<1e-8`

Feature+Confidence remains close:

- Feature+Stability is only `+0.31 pp` higher
- `p=0.198`

This is treated as a reviewer risk, not ignored.

### Confidence Risk Audit

The confidence audit reduces the risk that stability is merely confidence under
another name:

- StabilityResidual bad-edge AUC: `0.8027`
- Confidence bad-edge AUC: `0.7979`
- Same-confidence-bucket residual AUC gain: `+0.0290`
- High-Ambiguity same-confidence residual AUC gain: `+0.0317`
- Residual-stability coefficient after controlling for feature risk and
  confidence: `+0.212`

Interpretation: stability is related to confidence/uncertainty, but the
edge-quality evidence is not fully explained by confidence.

### Runtime Profile

Runtime is a cost tradeoff, not an efficiency claim.

- Feature-only: `1.91s`, accuracy `0.6089`
- StabilityResidual: `8.04s`, accuracy `0.6258`
- Ratio: `4.22x`
- Accuracy delta: `+1.69 pp`

Dominant added costs:

- Gradient confidence collection: `3.11s`
- Stability scoring: `2.88s`

## Known Limitations

- Feature-only pruning is a strong baseline and must be foregrounded.
- Confidence is a close uncertainty baseline; the strongest response is the
  same-confidence edge-quality analysis, not the small paired-accuracy gap.
- CiteSeer evidence is positive but weaker than Cora and PubMed.
- Full LDS/IDGL/ProGNN are not reproduced; current positioning is only against
  GSL-inspired proxies.
- LDS-Proxy is stronger than StabilityResidual in the current proxy audit, so
  the paper must not claim GSL superiority.
- Runtime is about `4.22x` Feature-only and should be reported as an
  accuracy-cost tradeoff.
- Texas, Wisconsin, and Actor heterophily experiments fail. Heterophily is a
  failure boundary, not a success case.
- P0 gain share is attributional evidence, not a complete causal proof.

## Repository Structure

```text
GraCA/
├── src/
│   ├── graca/                 # Historical GraCA components
│   ├── grage/                 # StabilityResidual / adaptive score logic
│   ├── models/                # GNN architectures
│   ├── training/              # Training and evaluation loops
│   ├── eval/                  # Noise injection and diagnostics
│   └── utils/                 # Seeds, masks, config utilities
├── scripts/                   # Experiment runners and paper utilities
├── configs/                   # YAML experiment configs
├── tests/                     # Focused tests
├── experiments/               # Submitted experiment prompts and results
├── paper_draft/               # AAAI draft, figures, tables, readiness audits
└── docs/                      # Workflow docs, project state, archived notes
```

Important paper files:

- `paper_draft/aaai_english_draft.md`
- `paper_draft/experiments.md`
- `paper_draft/limitations.md`
- `paper_draft/rebuttal_risks.md`
- `paper_draft/readiness_audit.md`
- `paper_draft/aaai_readiness_score.md`
- `paper_draft/figures_plan.md`
- `paper_draft/runtime_table.md`
- `paper_draft/leaderboard.csv`

## Current Paper Status

AAAI readiness: **8.4 / 10**.

Status:

- Main method fixed.
- P0/P1 mechanism evidence complete.
- Confidence risk substantially reduced.
- Runtime table complete.
- Heterophily failure boundary explicit.
- GSL positioning fixed as competitive with GSL-inspired proxies, not superior.
- Coherent English draft assembled.
- Figure plan complete; camera-ready styling remains.

No new main experiment is currently needed unless a final reviewer-risk pass
finds a direct contradiction. The project is in pre-submission polishing:
camera-ready figures/tables, citation completion, wording compression, and
final claim-boundary audit.

## Experiment Workflow

Experiment prompts and results are managed under `experiments/<exp_id>/`.
Long-running experiments use the repository workflow scripts:

```bash
bash scripts/submit_exp_tmux.sh <exp_id>
bash scripts/check_exp_status.sh <exp_id>
```

At the current stage, these scripts should be used only for explicitly approved
maintenance or verification tasks. The main experimental matrix is closed.
