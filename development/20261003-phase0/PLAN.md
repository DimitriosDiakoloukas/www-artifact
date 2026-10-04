# Phase 0: engineering pilot (development only)

Written 2026-10-03 before the runs. Design: NEW_CAMPAIGN_PLAN.md Section D, with the Section O decision
(prediction-level feature influence primary; embedding-level on the endpoints of 8 pairs).

Matrix: SGCN, SIDNET x Bitcoin-Alpha, Epinions x T in {2, 8, 32} x seeds {0, 1}; spectral features;
one fixed training configuration (Adam 5e-3, weight decay 1e-5, at most 300 epochs, validation every 5,
patience 20 evaluations, clipping 1.0 for both); SIDNET restart = the authors' released per-dataset value
(Bitcoin-Alpha 0.35, Epinions 0.55). Epinions: sign flips on 25 queries with m = 4 per shell.

Purpose: check the pipeline and measure cost. No number from this study is evidence; none will be
reported as a result. Nothing here is tuned on test AUC.

Exit criteria (Section D): checkpoint round trip; record hashes verify; zero influence beyond the
receptive field in every record; cost table for the confirmatory matrix; validation AUC at reload equals
the selection value.

Amendment, 2026-10-03 (engineering; prompted by out-of-memory failures, not by any result): the six Epinions runs that ran out of memory
(SIDNET T=8 in the batched Jacobian; SGCN and SIDNET T=32 in training) are rerun with plain VJPs
(chunk 1) and per-step activation checkpointing (`--memory-efficient`); tests show both are identical
in outputs and gradients.
