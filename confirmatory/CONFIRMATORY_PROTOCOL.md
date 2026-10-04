# CONFIRMATORY PROTOCOL

Frozen 2026-10-03. Binding from the moment `PROTOCOL_LOCK.json` records this file's SHA-256, the commit
that contains it and the UTC time, which happens before any confirmatory run. The items fixed at the
freeze were taken from the development outputs named in Section 10, and only from them.

## 1. Questions and hypotheses

Measure validity on trust-chain tasks (Section 4 of the paper):

- **H1 (sign influence recovers the required distance).** Over solved cells (test AUC >= 0.9 in every
  seed) at T = 32, the prediction-level sign-flip R90 lies within 1 of the required edge distance
  e* = r* - 1 in every cell, and the Spearman correlation of e* and mean R90 across solved cells is
  >= 0.9. Reported beside it, not as a condition: for unsolved cells with e* >= 3, mean R90 against e* - 2
  together with their AUC. Interpretation rule (from the probe-2 outcome, Section 10): sign range is
  read as evidence of the distance a model uses only for models that solve the task, and is always
  reported together with accuracy.
- **H2 (the feature Jacobian understates used range).** Over solved signed-relay checkpoints with
  r* >= 4 (all seeds, both T), the prediction-level feature-Jacobian R90 is below r*, the node distance
  of the source, in more than half of them: one-sided exact binomial test, alpha 0.05.

Web trust networks:

- **H3 (depth does not extend range).** Per architecture, the depth-effect ratio
  (D_32 - D_8) / (D_unif,32 - D_8) of the prediction-level sign-flip D, pooled over networks, has a 95%
  bootstrap upper bound below 0.5 (resampling network-seed units). Holm over the four architectures.
- **H4 (deep models are functionally shallow).** Per architecture, the median over T = 32 checkpoints of
  the functional depth T_func(0.01) (smallest retained steps whose fixed-head test AUC is within 0.01 of
  the intact checkpoint) is <= 8. Sign test against 8 over checkpoints, Holm over architectures.

Reported without a hypothesis test (descriptive, fixed in advance): the learnability map (which
(task, architecture, r*) cells are solved); per network, the paired AUC difference between each
architecture and the local-evidence baseline (Leskovec et al. 2010 features) with a seed-level 95%
interval; range by newcomer stratum (smaller endpoint degree 1, 2-4, >= 5); range at initialisation;
feature-Jacobian, sign-gradient and embedding-level ranges; random-feature arm; distractor arm.

## 2. Datasets

Bitcoin-Alpha, Bitcoin-OTC, Wiki-RfA, Wiki-Elec, Slashdot, Epinions; raw and processed SHA-256 pinned in
`src/srange/data/snap.py`; undirected, sign of summed ratings; 60/20/20 edge splits with split seed = run
seed, stored and hashed. Trust-chain tasks: generator `trust-chain-v1`, seeds 1,000,000 + 10 * run seed
+ split index, M = 1000/500/1000, q = 6, h = 2.

## 3. Architectures and depth

Source tree at the locked commit; versions and departures in `PROVENANCE.md`. Propagation steps T:
SGCN and SLGNN layer_num = T; SIDNET L = 2, K = T / 2, restart per network as released by its authors
(Bitcoin-Alpha 0.35, Bitcoin-OTC 0.25, Wiki-RfA and Wiki-Elec 0.45, Slashdot and Epinions 0.55; 0.15 on
trust-chain tasks); BGSD num_layers = T, gate off, relation split on, retention bias 0. Hidden width 64;
symmetric pair head (node head for signed relay).

## 4. Matrix

- Native: 4 architectures x 6 networks x T in {2, 8, 32} x seeds, spectral features. Random features:
  Bitcoin-Alpha and Wiki-Elec, same architectures and T.
- Trust-chain, T in {8, 32}, b = 0: signed relay r in {2, 4, 8, 16}; balance completion r in
  {4, 8, 16, 32}; 4 architectures. At T = 8 the cells with r* = 16 under-reach by construction (expected
  AUC exactly 0.5 by Lemma 1) and serve as under-reaching controls. Sign-blind controls: BGSD and SIDNET with all signs +1, r* = 8, both tasks.
  Distractor arm: signed relay, b = 2, r in {4, 8}, T = 32, 4 architectures.
- Local-evidence baseline: 6 networks, every seed.
- Seeds: 10000, 10001, 10002, 10003, 10004 (5). The job list is generated mechanically from this section
  by `scripts/make_confirmatory_jobs.py`.

## 5. Training and checkpoint selection

Adam with the per-(architecture, network) learning rate and weight decay of
`development/20261003-phase1-selection/selection.json` (selection sha256
`a8bbc1dd1bc75d31eb05bfaed5d404f3826559f6cf8e14f5b401e775bfbd1e60`, also in the lock); trust-chain tasks use
lr 5e-3, weight decay 1e-5. Gradient clipping 1.0 for all; full batch; validation every 5 epochs;
patience 20 evaluations; at most 300 epochs (400 on trust-chain tasks); the validation-best checkpoint
(encoder and head) is stored by hash and is the only object measured. `--deterministic` throughout;
`--memory-efficient` (per-step recomputation; outputs and gradients identical, tests/test_models.py) on
Slashdot and Epinions, for every native run at T = 32, and for SLGNN everywhere. Test labels are read once per checkpoint, after it is
stored.

## 6. Measurement

Primary (confirmed by development/20261003-signvalidity, V1 held): prediction-level sign-flip influence; per query the profile over edge distance, D, R90 and T_k; per
checkpoint the mean over queries. Queries: 100 test edges (instances) drawn with seed 7000 + run seed
from those whose endpoints are in the message-passing graph; sign flips on the first 50 queries with
m = 8 per shell (Slashdot and Epinions: 25 queries, m = 4). Embedding-level feature range on the
endpoints of the first 8 queries (Slashdot and Epinions: the first 2). Secondary as listed in Section 1.
Truncation: retained steps T, T/2, ..., 1, 0 as feasible, fixed trained head; prediction-level feature
range on the first 25 queries.

## 7. Units, tests and multiplicity

The seed (one stored checkpoint) is the unit of replication; queries are nested within it. Families:
H1 (decision rules, no p-value); H2 (one test); H3 (four tests, Holm); H4 (four tests, Holm). Everything
else is descriptive with seed-level 95% intervals.

## 8. Failures and exclusions

A run that crashes is rerun once unchanged; a second crash is reported as a failed run. No run is
excluded for its AUC. Checkpoints whose feature-influence gain is zero for a query report that query's
feature range as undefined (`zero_gain_fraction`); sign measures are still reported. Diverged training
(non-finite loss) is reported as such.

## 9. What may change after the lock

Only fixes for code that crashes or fails an exit check (`scripts/check_study.py`, `tests/run.py`).
A run that runs out of memory may be rerun with `--memory-efficient`, which does not change its results;
this is logged but is not an amendment.
Every fix is logged in `confirmatory/AMENDMENTS.md` with time, reason and diff, and every affected run
is rerun in full. Hypotheses, thresholds, matrix, seeds, measures, query sampling, training settings and
analysis scripts do not change. If the cost table is exceeded during the runs, the cut order of
NEW_CAMPAIGN_PLAN Section L is applied mechanically and logged.

## 10. Development outputs this protocol rests on

- `development/20261003-phase0`: engineering pilot; exit checks pass (`scripts/check_study.py`); cost table.
- `development/20261003-falsification`: F1 for the prediction-level feature Jacobian (RESULT.txt,
  sha256 `b4e1a81917e7bd4d109966efa338f3e57174ecd71903cf8a3b4af57f13d6fa5b`); led to the sign measure.
- `development/20261003-signvalidity`: V1 holds (9 solved cells, each within 1 of e*, Spearman 0.914);
  V2 fails, the violating cells being partial solvers with AUC 0.81-0.83 (RESULT.txt, sha256
  `e6f5708bdeb98990aa2e297c4bd616762c576adbe5458f5d257cba2b064a78ac`).
- `development/20261003-phase1-selection`: the frozen hyperparameters.
- `development/20261003-detcheck*`: deterministic mode runs for every architecture and runner, and a
  full run reproduces bitwise across physical cards, including 16 GB and 32 GB V100s.

Confirmatory seeds, splits and trust-chain instances are disjoint from every development one.
