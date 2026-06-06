# Notes: Confidence Risk Audit

This experiment is P0 reviewer-risk reduction, not method search.

Current known risk:

- In `2026-06-05-ambiguity-stability-evidence`, `Feature+Stability` beats
  `Feature+Confidence` by only `+0.31 pp` on FSCC, with `p=0.198`.
- Aligned stability already beats random, shuffled, and node-permuted stability
  controls, but that does not fully answer whether stability is just confidence.

Required paper-facing outcome:

- If stability remains informative after confidence matching, use it to support
  the claim that stability is a related but distinct training-dynamics signal.
- If confidence explains most of the signal, keep the main result but shrink the
  mechanism claim and state the risk in `limitations.md` and
  `rebuttal_risks.md`.

Do not add datasets or new method variants unless required to complete the
confidence-control diagnostic.

Local Codex verification on 2026-06-05:

- `python -m py_compile scripts/run_confidence_risk_audit.py` passes.
- Local default Python lacks `numpy`.
- Codex bundled Python has `numpy/pandas` but lacks `torch`.
- Smoke/full execution therefore needs the remote experiment environment or a
  local environment with `requirements.txt` installed.
