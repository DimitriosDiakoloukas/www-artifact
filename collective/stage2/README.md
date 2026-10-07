# Expanded WWW study

This directory contains the complete evidence for the expanded locality study. The scientific
contribution is a validity map and empirical separation of individual sensitivity, collective
prediction stability and computational fidelity. It does not rename standard conditional variance,
perturbation fidelity or adaptive inference as a newly invented method.

## Frozen designs and coverage
- PROTOCOL/LOCK: native random inputs and 144 known-function cells plus three saturation controls.
- NATIVE_COLLECTIVE_PROTOCOL/LOCK: 20 native T32 checkpoints, 20 queries each, three radii, two
  size-matched laws, eight draws (19,200 prediction slots).
- Matched constraints: four stored standardised checkpoints, 20 queries, four radii, three budgets,
  five laws and 16 draws (76,800 slots; infeasible requests and unchanged cycles are explicit).
- LARGE_CONTROL_PROTOCOL/LOCK: two encoders, independent consensus/correlated redundancy and three
  seeds, with 101 paths at distance three. All 12 selected weights and all 240-query audits remain;
  audit_v2 is the final strict bridge/fallback implementation. V1 attempts are superseded and preserved.
- COMPUTATION_PROTOCOL/LOCK plus policies/POLICY_LOCK: all 12 SIDNET checkpoints, independent
  development/policy-validation queries and 128 held-out test queries excluding prior sensitivity queries.
- LAYERWISE_PROTOCOL/LOCK and layerwise_policies/LAYERWISE_POLICY_LOCK: development-motivated follow-up
  with new validation queries, committed before either held-out candidate set existed.
- Every actual runtime policy/schedule: one warm-up, five synchronized repeats, batches 1/8/64,
  actual grouping/crops, peak allocations, CPU RSS and calibration ledger costs.

Read DEVIATIONS.md for source-preserving numerical changes, unsolved controls, policy-data reuse,
wrapper redistribution and constraints on interpretation. Every cold/resident inference comparison
recomputes embeddings per request batch; unchanged-snapshot cache hits are outside the benchmark.

## Regeneration and audit
`python3 collective/stage2/report.py --paper <paper>` regenerates all expansion assets through an
empty temporary directory after checking complete raw hashes. It never alters experiment records.
The full manuscript gate is `bash reporting/regen_check.sh <paper>`; use the pinned Python environment.
`results/integrity.json` verifies protocol commits, separate validation subsets and prospective policy
locks; rerunning integrity.py needs the full campaign git history. `results/native_verification.json`
contains all-60-checkpoint bitwise replay and four complete 100-query gradient replays.
`results/independent_replay-*.json` independently reconstruct matched edits, native joint edits and
full-union learned completions. No test result is used to change a policy.

Complete collectors: collect.py (matched/native), large_collect.py (all 12 learned runs),
computation_collect.py (all 204 policy/schedule outcomes and benchmarks). These distinguish seed
variation, completion sampling error, fidelity to the original prediction and accuracy on original
or resampled labels. Source/checkpoint/protocol hashes make mismatches fail loudly.


## External-review reporting revision (7 October 2026)

Read [REVIEW_ERRATA.md](REVIEW_ERRATA.md). The original all-relation metric remains alongside a
variable-sign normaliser and three thresholds for every learned run. Native reports include logit
RMSE/reference confidence, and computation reports include class counts and retained nodes.
`review_diagnostics.py` replays every saved radius-one/five-percent cycle and count-matched exchange
draw in the four standardised checkpoints, asserting achieved counts and reporting degree/shell bias.
`report.py` incorporates these analyses and four new disclosure tables in `stage_results.json`.
All analyses are explicitly post hoc; none changes the frozen predictions or makes an architecture-only,
natural-counterfactual or full-test computation claim. The complete original decision ledger retains
H1 specificity and all three unsupported H4 outcomes.
