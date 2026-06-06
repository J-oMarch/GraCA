# Reviewer Risk Audit

## Runtime Risk

Current evidence: StabilityResidual is approximately **4.22x** Feature-only
wall-clock time for **1.69 pp** accuracy gain on homophilic citation FSCC.

Component breakdown shows the overhead is concentrated in:
1. Probe/model-view training: 0.25s
2. Stability scoring (multi-view + residualization): 2.88s

This is an accuracy-cost tradeoff, not an efficiency claim.  The paper should
frame the runtime as a known limitation and present the tradeoff honestly.

## Other Reviewer Risks

See `paper_draft/rebuttal_risks.md` for the full risk matrix.
