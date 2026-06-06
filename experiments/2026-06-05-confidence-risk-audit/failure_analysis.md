# Failure Analysis: Confidence Risk Audit

## Failure Modes

- None under the configured decision rules.

## Reviewer-Risk Interpretation

- If confidence AUC is close to or exceeds StabilityResidual AUC, reviewers will
  reasonably frame stability as uncertainty under another name.
- If same-confidence-bucket AUC delta is negative or near zero, the residual
  stability signal does not survive confidence control.
- If partial correlation coefficient is non-positive, residual stability does not
  predict bad edges after controlling for feature risk and confidence.

## Required Paper Updates

The current claim can be maintained with explicit confidence discussion.

Update:
- `paper_draft/limitations.md`
- `paper_draft/rebuttal_risks.md`
