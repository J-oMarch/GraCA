# Notes

This is the innovation search lane. It may modify code substantially, but it
must keep the research direction:

> model training behavior should guide automatic graph evolution.

The output should not be a blind sweep. It should produce at least one clear
candidate method, one rejected variant, and enough evidence to decide whether
the candidate deserves a larger confirmation run.

Good candidates should be simple enough to explain in a paper:

- adaptive weighting by feature ambiguity,
- prediction-stability-guided edge risk,
- checkpoint-consistency of edge-gate gradients,
- explicit protective edge term from negative gradients.
