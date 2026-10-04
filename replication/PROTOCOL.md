# Replication protocol: decisive-evidence reach (pre-registered, 2026-10-04)

This protocol is written after the confirmatory campaign (lock bef7282), the exploratory studies
E1-E4 (exploratory/PLAN.md) and the post-hoc diagnostics listed in exploratory/POSTHOC.md. Those
results motivated every hypothesis below; none of the data tested here exists yet. The new data
are new networks for the planted chains (Bitcoin-OTC, Wiki-RfA), a required distance not used before
(r = 6), new seeds (30000-30004), the two largest networks (Slashdot, Epinions) and the random-feature
arm, neither of which has been scored with decisive reach, and a per-layer truncation of SIDNET that
has never been evaluated. Runs start only after `replication/LOCK.json` is committed; the runner and
the measurement script refuse otherwise (`replication/gate.py`).

## 1. The measure

For a query q (a target node on a chain, an edge on a network) and every message-passing edge f,
g_q(f) = |d logit_q / d s_f|, the sign gradient computed with one backward pass per query from the
stored checkpoint. Edge distance d(f) = the hop distance from q (the nearer endpoint for an edge
query) to the nearer endpoint of f. Only edges with finite d(f) < T count.

Decisive reach at threshold tau: rho_tau(q) = max { d : max_{f : d(f) = d} g_q(f) >= tau * max_f g_q(f) },
the farthest distance whose most influential edge carries at least tau of the query's most influential
edge. Queries with no counted edge, or a zero gradient everywhere, are dropped and their count is
reported. A run's value is the mean over its queries (the first 100 test queries on chains, the 100
test pairs stored in each network record). **Primary tau = 0.1**; tau = 0.05 and 0.2 are reported for
every result. Ceiling: the mean over queries of the farthest counted distance (the reach of a model
whose every edge is equally influential).

Every measurement regenerates the test logits from the checkpoint and checks them bitwise against the
record (`test_logits_sha256`). A mismatch fails the run.

## 2. Hypotheses

e* = r - 1 is the required edge distance of a signed-relay chain of radius r (b = 0). A run is
**solved** if test AUC >= 0.9 and at **chance** if test AUC < 0.6. A cell is solved if all 5 of its
seeds are solved.

- **RH1 (decisive reach finds the required distance).** In every solved planted cell, the cell mean
  of rho_0.1 lies in [e* - 1, e* + 1]. Holds if true in every solved cell and there are at least 3
  solved cells; fewer than 3 solved cells make RH1 inconclusive.
- **RH2 (reach separates solved from chance runs).** Over all 120 planted runs, rho_0.1 / e* is larger
  in solved than in chance runs: one-sided Mann-Whitney U, p < 0.05. Needs at least 5
  runs in each group, else inconclusive. Reported beside it: the fraction of chance runs with
  |rho_0.1 - e*| <= 1 (reach is necessary for solving, not sufficient).
- **RH3 (the mass summary falls short where decisive reach does not).** In every solved planted cell,
  the cell mean of the sign-flip R90 (the mass summary of the confirmatory protocol, recorded by the
  runner) is below e*. Holds if true in every solved cell (and at least 3 exist).
- **RH4 (real networks are local; held-out networks and features).** (a) Slashdot and Epinions, spectral
  features, T in {8, 32}, four architectures, five seeds: every (network, architecture, T) cell mean of
  rho_0.1 is <= 2. (b) Random-feature arm, Bitcoin-Alpha and Wiki-Elec, T in {8, 32}, four
  architectures, five seeds: every cell mean of rho_0.1 is <= 2. Each part holds if all 16 of its cells
  satisfy the bound. Disclosure: a smoke-test filter matched one RH4(b) checkpoint by accident (SIDNET,
  Bitcoin-Alpha, T = 8, seed 10000, random features) and printed its rho_0.1 = 1.05 before the lock.
  Threshold rationale: on the four networks scored post hoc the largest cell mean
  was 1.60, while solved planted chains on the same backbones gave rho_0.1 near e* >= 3.
RH2 is the only p-value; RH1, RH3 and RH4 are decision rules. H4 of the confirmatory protocol is not
re-decided: it failed for SIDNET as locked and is reported that way.

**Why SIDNET truncation is descriptive, not a hypothesis.** A per-layer truncation of SIDNET (k in
{16, 8, 4, 2, 1, 0} diffusion steps in each of the two layers, 2k retained steps; the confirmatory
schedule removed the earliest steps first, so at 16 retained steps the first layer had none and its
negative channel was the random seed M0) was drafted here as a hypothesis that SIDNET is functionally
shallow. Its smoke test, on one of the 30 checkpoints it would have used (SIDNET, Bitcoin-Alpha, T = 32,
seed 10000), showed test AUC 0.797 / 0.758 / 0.627 at 32 / 16 / 8 retained steps, and 0.798 / 0.798 /
0.799 / 0.795 at 32 / 16 / 8 / 4 retained steps with M0 set to zero. Having seen part of
its test data, we report it descriptively for all 30 checkpoints instead, with the same rows evaluated
with M0 set to its expectation (zero), and the confirmatory-order rows beside them.

Reported without a test: per cell AUC, rho at all three thresholds and the ceiling; the feature R90
(node units) against r on planted chains; rho_0.1 by newcomer stratum (smaller endpoint degree 1, 2-4,
>= 5) on all six networks at T in {8, 32} (the four smaller networks were already scored post hoc and
are marked as such); SIDNET truncation as described above, with T_func^layer(0.01) computed by
`analysis/native.py:t_func` and k = 16 required to reproduce the record's test logits bitwise.

## 3. Runs

Planted chains (`replication/run_chain_r.py`, identical to exploratory/run_chain_x.py apart from the
gate and the tree): `--task relay --variant planted-<net> --b 0 --T 32`, net in {bitcoin_otc, wiki_rfa},
r in {4, 6, 8}, arch in {SIDNET, BGSD, SGCN, SLGNN}, seeds 30000-30004, all other settings at the
runner defaults used by E3 (M = 1000/500/1000, hidden 64, lr 5e-3, weight decay 1e-5, at most 400
epochs, patience 20, restart 0.15). 120 runs. `--memory-efficient` on every run and `--chunk 1` on
Wiki-RfA (identical results, less memory).

Measurements (`replication/measure.py`), from stored checkpoints only: decisive reach on the 120 new
chain checkpoints; on the confirmatory network checkpoints for RH4 (160) and for the newcomer report
(the remaining 160 spectral T in {8, 32} checkpoints); SIDNET per-layer truncation (30, descriptive).

## 4. Failures, exclusions, changes

As confirmatory Sections 8 and 9. A crashed run is rerun once unchanged; an out-of-memory failure is
rerun once with `--chunk 1` (identical results); a second failure is reported as failed. No run is
excluded for its AUC. A measurement that runs out of memory is rerun once on a 32 GB card (identical
results). After the lock only fixes for code that crashes may change, each logged in
`replication/RERUNS.md` with its commit.

## 5. Analysis

`replication/analyze.py` (committed with this protocol) applies the rules above and writes
`replication.json` to the paper's `generated/` directory.

## 6. Smoke tests before the lock (development/20261004-replication-smoke)

- `measure.py sidnet` on SIDNET, Bitcoin-Alpha, T = 32, seed 10000 (values disclosed in Section 2).
- `measure.py native` on SIDNET, Bitcoin-Alpha, T = 8, seed 10000: spectral (rho_0.1 = 1.32, equal to
  the post-hoc value) and, by accident, random features (disclosed under RH4).
- `measure_native` on BGSD, Wiki-Elec, T = 2, seed 10000, random features (not a tested cell): logits
  match.
- `measure.py chains` on the E3 record SIDNET, planted Bitcoin-Alpha, r = 8, seed 20000: rho_0.1 = 7.14,
  equal to the post-hoc value.
- `run_chain_r.py` on planted Wiki-RfA, r = 8, SIDNET, seed 99, 5 epochs (memory and runner check).
- `analyze.py` on the outputs above and on synthetic outcomes (decision rules).
