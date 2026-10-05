# Reviewer-requested validation and manuscript revision
Written before computing either new study. These are exploratory extensions requested after the
manuscript review on 2026-10-05; they do not change the locked confirmatory or replication analyses.

## A. Exhaustive single-sign interventions
- Existing T=32, spectral-input checkpoints, seed 10000, all four encoders, on Bitcoin-Alpha and
  Bitcoin-OTC. These are the two smallest networks, selected for computational feasibility.
- Ten queries per checkpoint, drawn uniformly without replacement from its 100 stored queries
  by NumPy default_rng(20261005), sorted back into stored order. No queries selected by gradients,
  reach, correctness, confidence or finite effects.
- Negate EVERY training relation once. Evaluate all ten queries together for each negation,
  with topology, input features, checkpoint and SIDNET evaluation initialisation fixed.
  Enumerate all distances, including beyond three and unreachable relations.
- Define gradient and finite reach with the SAME equation, tau=0.1 and each quantity's own
  global maximum over all training relations. Undefined all-zero profiles remain undefined.
  Report per-query agreement, finite/gradient reach, absolute logit changes, prediction flips,
  effects beyond distance three and missed finite-threshold relations. Report threshold
  robustness at 0.05 and 0.2. No assumption that any outcome should pass.
- Check intact test logits bitwise. Save partial arrays with hashes and require exactly one
  intervention per training relation before collecting. GPU processes checkpoint progress and
  stop after about 120 seconds; resume from completed portions.
- This checks individual finite effects on 80 queries, eight checkpoints and two networks;
  it cannot rule out collective effects or generalise the audit to all six networks.

## B. Published objectives and decoders, tuned separately by depth
- Bitcoin-Alpha and Wiki-Elec, chosen to cover trading and voting at feasible sizes.
- Keep ordered SNAP relations: sum repeated ratings of the SAME ordered pair, discard
  zero-sum pairs and self-loops; do not merge reciprocal ratings. Remap node ids in sorted order.
  Uniform random 60/20/20 relation splits, persisted and hashed; seed equals run seed.
- SGCN: the pinned torch-geometric-signed-directed 1.1.1 complete SGCN model, its three-class
  positive/negative/non-edge decoder, and its native sign-entropy plus structural triplet
  objective, lambda=5 and the package's published defaults. Train its package forward/loss.
  Measure the positive-minus-negative decoder logit. Verify the differentiable encoder's
  binary forward against the unchanged package encoder before measuring.
- SIDNET: unchanged encoder and two-class concatenation decoder from snudatalab/SidNet
  commit c1064ddc1761bf047211f5f4fcdf364cd330d66c, native unweighted NLL, two layers,
  released restart (0.35 Bitcoin-Alpha; 0.45 Wikipedia), Adam with StepLR(10,0.99).
  Training initialisation is drawn afresh, as released; evaluation uses a seeded fixed draw
  so all measurements evaluate one function. Upstream source files are pinned by SHA-256.
- Hidden width 64; input width 64. SGCN uses its native signed spectral-feature routine;
  SIDNET uses the released directed-adjacency truncated-SVD construction (U times Sigma,
  n_iter=30, random_state=run seed). Features see training relations only and are frozen.
- T=2,8,32 propagation steps (SIDNET K=T/2 per layer). For EACH model/network/T select
  learning rate from {0.001,0.005,0.01} and weight decay from {0,0.00001} by mean validation
  AUC over tuning seeds 41000 and 41001, breaking ties by smaller learning rate then decay.
- Up to 300 epochs, validation every five epochs, patience 20 validation checks. Select the
  epoch on validation only. Then fit selected settings on evaluation seeds 42000..42004.
  Tuning does not compute test AUC or range. Evaluation reads test labels only after selection.
- Measure exact sign-sensitivity reach on 100 eligible test relations drawn uniformly with
  seed 7000+run seed; distance uses the undirected support of the directed training graph,
  measured to either query endpoint. A directed relation has one sign parameter.
- Report every evaluation cell's AUC, reach and ceiling with seed-level Student-t 95% intervals.
  Report paired depth differences. Do not equate performance with the undirected main study
  or compare this directed arm to the undirected short-walk baseline.
- These are published-model checks under documented validation/split/width choices, not
  reproductions of the original papers' accuracy or evidence about every signed GNN.

## Reporting
Keep all outcomes, failures and deviations. No original hypothesis is retroactively changed.
Generated summaries, tables and numerical macros come from stored records and RESULT_MAP.
Main claims describe individual sensitivity and the evaluated setup, regardless of outcomes.

