# GraCA / GraGE Project State

This file is the durable context memory for Codex, ChatGPT, and Claude Code.
Read it before proposing new experiments, interpreting results, or changing the
research direction.

## Publication Target

The current goal is to develop the project into an AAAI-level paper. Experiments
should be designed to support or falsify clear paper claims, not only to improve
engineering metrics.

Target paper direction:

```text
GraGE: Training-Dynamics-Guided Graph Evolution via Differentiable Edge Gates
```

The core paper question is:

```text
Can training-dynamics-derived edge signals provide useful graph evolution
information beyond static feature similarity?
```

## Research Direction

The project started as GraCA:

```text
Gradient-guided Graph Connection Assessment
```

It has shifted toward GraGE:

```text
Gradient-guided Graph Evolution
```

The current framing is not merely "graph cleaning." The stronger framing is:

```text
Graph structure should be treated as a trainable/evolvable object whose edge
weights or edge gates are guided by training dynamics, validation behavior,
prediction stability, and generalization signals.
```

## Current Technical Hypothesis

The main hypothesis to test is:

```text
Edge-level training dynamics, especially differentiable edge-gate gradients or
hypergradients, contain information that static feature-similarity pruning does
not capture.
```

The central mathematical object is:

```text
S_e = d L_score(theta*(M), M) / d m_e
```

where:

```text
m_e in [0, 1]
```

is a differentiable edge gate.

## Important Prior Conclusions

1. Feature-only pruning is a strong baseline, not the main method.

   `Feature-only = -feature_cosine` performed very strongly in controlled
   experiments. It is too close to existing feature-similarity pruning methods
   to serve as the main novelty, but it must remain as a strong baseline.

2. The old EdgeInfluence-Pseudo method is not strong enough as the main method.

   It behaved more like pseudo delta-softmax plus feature cosine than a true
   differentiable edge-gate hypergradient. It should be retained only as an
   ablation or historical comparison.

3. The project must avoid label leakage.

   Any method that uses validation/test labels or full oracle labels must be
   clearly marked as oracle/diagnostic, not practical.

4. A publishable claim requires beating strong static baselines.

   The key practical comparison is not only against Random-Matched or DropEdge.
   It must include Feature-only / Similarity-Pruning / Homophily-style baselines.

5. Oracle success alone is insufficient.

   Oracle edge signals show whether useful edge-level task signal exists, but
   they do not prove the practical method works.

6. The current GraGE-Hybrid/MCGC practical claim is not supported by the
   second-batch confirmation.

   `2026-06-04-fscc-confirmation-rerun` completed a matched-budget 20-seed
   FSCC confirmation across Cora, CiteSeer, and PubMed. Feature-only was the
   strongest overall method (`0.6116 ± 0.0496`). GraGE-Hybrid lost by
   `-2.50 pp` (`p=0.0012`, win rate `0.10`, Cohen's d `-1.40`), and MCGC lost
   by `-0.72 pp` (`p=0.143`, win rate `0.43`). MCGC improved Cora FSCC, but
   Random-Matched and DegreeAwareRandom improved more on the same slice, so the
   positive Cora result is likely a pruning budget/degree effect rather than
   evidence of a residual training-dynamics edge signal.

7. The current paper path must be method-rebuild or diagnostic reframing.

   Do not keep adding small sweeps around the existing rank-normalized hybrid or
   MCGC score. First-batch diagnostics and second-batch confirmation indicate
   that edge-gate gradient magnitudes are near-zero, signs are near-random after
   feature-risk control, and rank normalization can amplify noise. A new AAAI
   attempt must either introduce a substantially different no-leak dynamics
   channel or honestly frame GraGE as a diagnostic/falsification study relative
   to Feature-only pruning.

8. StabilityResidual-GraGE is the current positive main-method candidate.

   `2026-06-04-stability-channel-rebuild` implemented a new no-leak
   prediction-stability residual channel. The method trains stochastic graph
   views, converts node prediction instability to edge scores, residualizes the
   signal against feature risk, and uses edge-gate gradient confidence only as an
   optional abstention/regularization mechanism. Validation over
   Cora/CiteSeer/PubMed with 10 seeds and FSCC/LFS/DAR controls found FSCC
   `+2.00 pp` vs Feature-only (`p=0.0001`, win rate `0.87`, Cohen's d `0.41`)
   with no material degradation on LFS (`+0.73 pp`) or DAR (`+0.30 pp`).

   `2026-06-04-stability-ablation-confirmation` strengthened this with 20-seed
   confirmation and ablations. FSCC `+1.59 pp` vs Feature-only (`p<0.001`,
   win rate `0.83`, Cohen's d `0.70`). LFS `+0.55 pp` (not significant, no
   degradation). DAR `+0.81 pp` (`p<0.001`). Per-dataset: Cora `+3.19 pp`
   (100% wins), PubMed `+1.05 pp` (`p=0.002`), CiteSeer `+0.52 pp` (not
   significant). Ablations show: raw stability is nearly as good as residualized
   (`+0.14 pp` difference, not significant); gradient confidence adds `~0.5 pp`;
   all dropout schedules and view counts work. The method is AAAI-ready with
   honest reporting.

9. The supported claim has changed.

   Do not claim that raw edge-gate gradients are the main successful signal.
   The current supported claim is that prediction stability under graph
   perturbations provides a training-dynamics-derived edge signal residual to
   feature similarity. Edge-gate gradients remain useful as local sensitivity
   theory and possible confidence/abstention, but the selected candidate and
   controls show they are not the dominant empirical driver.

10. Heterophily is a confirmed boundary condition.

   `2026-06-04-stability-heterophily-regime` tested Texas, Wisconsin, and Actor
   with 10 seeds and four regimes. StabilityResidual loses to Feature-only by
   `-1.14 pp` overall (`p=0.0133`, win rate `0.31`) and by `-2.89 pp` on the
   heterophily FSCC slice. Feature-only is the strongest method; all tested
   GraGE/static/random alternatives lose. The paper claim must be restricted to
   homophilic, feature-ambiguous citation regimes, with heterophily reported as
   a failure mode and motivation for future regime detection/fallback.

## Current Implementation State

Relevant directories:

```text
src/graca/      # original GraCA components
src/grage/      # GraGE edge-gate and hybrid score components
scripts/        # experiment runners
configs/        # dataset configs
tests/          # focused unit tests
paper_tables*/  # historical generated tables
results*/       # historical result files
```

Important existing GraGE components:

```text
src/grage/edge_gate_influence.py
src/grage/unrolled_hypergradient.py
src/grage/hybrid_score.py
src/grage/pruning.py
```

Important experiment/table scripts:

```text
scripts/run_grage_experiments.py
scripts/run_grage_edge_gate_experiments.py
scripts/run_grage_hybrid_sweep.py
scripts/build_grage_tables.py
scripts/build_grage_edge_gate_tables.py
scripts/build_grage_hybrid_tables.py
```

## Experimental Priorities

When designing new experiments, prioritize:

1. GraGE/hybrid score vs Feature-only under matched pruning budgets.
2. Edge-gate or hypergradient signal quality beyond feature cosine.
3. Seed stability across Cora, CiteSeer, PubMed and selected heterophily data.
4. Ablations separating:
   - feature prior
   - gradient signal
   - positive/negative score split
   - degree preservation
   - support/score split
   - unrolled hypergradient
5. No-leakage practical setting first; oracle only as diagnostic.
6. Paper-facing evidence: tables, effect sizes, stability, failure modes.

## Decision Rules

Use these rules when interpreting results:

1. If GraGE/hybrid consistently beats Feature-only with statistical support,
   the AAAI paper direction remains viable. This condition is met by
   StabilityResidual-GraGE on homophilic citation FSCC (20-seed confirmation:
   +1.59 pp, p<0.001, win rate 0.83). Residualization ablation and sensitivity
   analysis are complete. Heterophily validation is negative, so the claim must
   be regime-limited. GSL baseline comparison remains as a next step.
2. If GraGE only beats Random-Matched but not Feature-only, the method needs a
   stronger technical contribution before paper writing.
3. If edge-gate/hypergradient signals have poor bad-edge detection but accuracy
   improves, inspect whether improvements are actually caused by feature prior
   or budget effects.
4. If results are dataset-specific, identify the graph regime where the method
   works instead of overclaiming generality.
5. If results do not support the claim, revise the method or reframe the paper
   before producing more tables. This condition currently applies after
   `2026-06-04-fscc-confirmation-rerun`.

## Automation Workflow

New experiments must be represented as:

```text
experiments/<exp_id>/prompt.md
```

Expected output contract:

```text
experiments/<exp_id>/result.md
experiments/<exp_id>/metrics.json
experiments/<exp_id>/logs/
```

Preferred execution path for long experiments:

```bash
bash scripts/submit_exp_tmux.sh <exp_id>
```

This uses one fixed remote tmux session:

```text
graca_claude
```

Each submitted experiment gets its own tmux window inside that session. This
keeps management simple while still allowing explicit parallel experiments.

Claude Code runs with:

```bash
--dangerously-skip-permissions --permission-mode bypassPermissions --effort max
```

Check status from Codex:

```bash
bash scripts/check_exp_status.sh <exp_id>
```

Blocking execution path:

```bash
bash scripts/submit_exp.sh <exp_id>
```

Use blocking mode only for short tasks or when the user explicitly wants Codex
to wait for completion.

## Remote Server

Remote SSH:

```bash
ssh -p 15600 jyh@59.72.109.245
```

Remote project:

```bash
/home/jyh/workplace/ClaudeProjects/GraCA
```

The user may manually attach to tmux for supervision:

```bash
ssh -p 15600 jyh@59.72.109.245
tmux attach -t <session_name>
```

Manual supervision is only for observation or intervention when needed. The
normal workflow should be driven by Codex commands in the local Mac checkout.
