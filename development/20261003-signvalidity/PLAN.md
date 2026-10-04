# Probe 2: does sign influence recover the required distance? Plan and decision rules

Written 2026-10-03, after the falsification probe (../20261003-falsification/RESULT.txt) found F1 for the
feature Jacobian and, in post-outcome inspection, sign-influence profiles that matched the required
distance. Because the sign measure was chosen after seeing that probe, this probe tests it on fresh
seeds and conditions it has not seen. Nothing below was chosen from a result of this probe.

## Runs (all T = 32, b = 0 unless stated, M = 1000/500/1000, seeds 10, 11, 12)

- Signed relay, r in {2, 4, 8, 16} (required edge distance r - 1).
- Balance completion, r in {4, 8, 16, 32} (r* = r/2, required edge distance r* - 1).
- Architectures SGCN, SLGNN, SIDNET (c = 0.15), BGSD: 2 tasks x 4 r x 4 architectures x 3 seeds = 96.
- Distractor arm: SIDNET, relay, b = 2, r in {4, 8}: 6.
- Training as in the falsification probe.

## Measure

Primary: sign-flip influence at the prediction level (50 queries, m = 8 per shell), its R90 in edge
distance, mean over queries then seeds. The target for a cell is the required edge distance
e* = r* - 1. Secondary (descriptive): sign gradient, prediction-level feature Jacobian, embedding-level
range, range at initialisation.

## Decision rules (a cell is (task, architecture, r, b); solved = test AUC >= 0.9 in every seed)

- **V1, tracks the requirement:** in every solved cell, |mean R90 - e*| <= 1; and over solved cells,
  the Spearman correlation between e* and mean R90 is at least 0.9 (needs at least 3 distinct e*).
- **V2, specific:** in every unsolved cell with e* >= 3, mean R90 <= e* - 2.
- **Outcome.** V1 and V2 hold: sign influence is the primary measure in the confirmatory protocol.
  V1 fails: neither measure recovers used range; the paper is re-scoped to that negative result and
  no range claim is made from either measure. V1 holds and V2 fails: sign influence is reported only
  together with accuracy (range is not read as use for unsolved models).
- **Reported whatever the outcome:** which cells are solved (the learnability map), the feature-Jacobian
  R90 in solved cells against e* + 1 (the node distance of the source), and the distractor arm.

Amendment, 2026-10-03 (engineering; prompted by out-of-memory failures, not by any result): SLGNN
balance-completion runs at T = 32 ran out of memory on 16 GB cards. They are rerun on 32 GB cards with
`--memory-efficient` (per-step recomputation in the backward pass; outputs and gradients identical, see
tests/test_models.py) and `--chunk 1` (plain vector-Jacobian products, identical values).
Amendment 2, 2026-10-03 (engineering; prompted by out-of-memory failures, not by any result): with
checkpointing, retaining the forward graph across the Jacobian's backward passes accumulates recomputed
buffers. In memory-efficient mode the forward pass is now rebuilt for each Jacobian row (`fresh`; values
identical, tests/test_range.py). The 12 SLGNN balance-completion runs are rerun with it.
