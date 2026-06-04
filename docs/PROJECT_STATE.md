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
   the AAAI paper direction remains viable.
2. If GraGE only beats Random-Matched but not Feature-only, the method needs a
   stronger technical contribution before paper writing.
3. If edge-gate/hypergradient signals have poor bad-edge detection but accuracy
   improves, inspect whether improvements are actually caused by feature prior
   or budget effects.
4. If results are dataset-specific, identify the graph regime where the method
   works instead of overclaiming generality.
5. If results do not support the claim, revise the method or reframe the paper
   before producing more tables.

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
