# Completed pilot validation
- The protocol and target hashes still match LOCK.json. Its initial protocol commit is 3fb225c2;
  lock and implementation commits precede all new measurement records.
- All 23 tests pass across test_collective, test_collective_reporting, test_models, test_range,
  test_synthetic and test_revision.
  The five new tests check the variance identity and fixed-configuration nonmonotonicity, majority
  vertex derivatives against every finite flip, redundancy and saturation counterexamples, sampler
  constraints and empty eligible sets, graph distances and feature/topology construction.
- Every original full-test logit is bitwise unchanged before and after every bounded measurement
  process. Four network checkpoints contain all 400 primary cells and all 15 supplementary cells.
- verify.py independently replays all six checkpoints' full-test logits, their first selected gradient
  rows, 20 seeded real-network intervention draws and four true-Q trained-control draws, bitwise.
  The verifier result and core-source hash are in results/verification.json.
- Both bundled selected control checkpoints match the hashes in their completed training records.
- All seven report outputs regenerate byte for byte in a separate /tmp directory, including PDFs,
  PNGs, JSON, Markdown and the manifest. All referenced raw hashes are verified before collection.
- Negative checks in a disposable /tmp copy reject a missing raw draw and a byte-modified raw draw.
  Initial fixtures omitted LOCK.json and the temporary campaign root; the permanent reporting
  integration test supplies both and passes the intended missing/corrupted-draw checks.
  The earlier incomplete-run check also rejected an absent completion record. No original data changed.
- Core models and replication/measure.py are unchanged from the validated bbd0668 baseline. The new
  work is restricted to the collective pilot and its test module; the current paper and anonymous
  artifact are unchanged. Frozen campaign history is retained in the WWW mirror's merge ancestry.

The known-function controls are validated separately from learned-model success. All 27 settings recover
their analytic collective variance within the stated MC bound. Individual measurements intentionally
fail in some regimes. Both trained consensus controls learn strongly and recover distance three with
individual as well as collective diagnostics. The constrained network sampler's very low acceptance
is an inconclusive experimental outcome, retained in full. No acceptance score or general sufficiency
certificate follows from this pilot.
