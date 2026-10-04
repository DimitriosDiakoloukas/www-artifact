# Falsification probe (NEW_CAMPAIGN_PLAN Section J): plan and decision rules

Written 2026-10-03 before any run of this study. Seen before writing: two smoke runs (relay r=2, BGSD,
T=4, small M: solved, prediction-level D 1.31, R90 2.02; balance r=4, SGCN, T=4: not learned) and the
Phase-0 Bitcoin-Alpha SGCN runs at T in {2, 8} (pipeline checks only). No threshold below was chosen
after seeing a probe result.

## Runs

- Signed relay, b = 0 (bare chains, padded to r_max = 16), M = 1000/500/1000, r in {2, 8},
  T in {8, 32}, architectures BGSD and SIDNET (restart c = 0.15), seeds {0, 1, 2}: 24 checkpoints.
- Sign-blind control: BGSD with every sign set to +1, r = 8, T = 32, seeds {0, 1, 2}: 3 checkpoints.
- Native: Bitcoin-Alpha, BGSD and SIDNET, T in {8, 32}, seeds {0, 1, 2}, spectral features: 12 checkpoints.
- Training as Phase 0 (Adam 5e-3, weight decay 1e-5, clipping 1.0, validation every 5, patience 20;
  at most 400 epochs on the relay task, 300 on the native graph).

## Measure

Primary: prediction-level feature influence (decision O), its mean distance D and 90% radius R90, mean
over the 100 test queries, then over seeds. Secondary, reported but not used in the rules: embedding-level
range, sign-gradient and sign-flip influence, range at initialisation.

## Decision rules (a cell is (architecture, r, T); "solved" means test AUC >= 0.9 in every seed)

- **F1, instrument fails:** some architecture solves r = 8 at some T, and in that cell mean R90 < 8, or
  mean D at r = 8 does not exceed mean D at r = 2 at the same T and architecture.
- **F2, depth dominates:** over the solved cells, the mean absolute change in D from T = 8 to T = 32 at
  fixed r is at least the mean change in D from r = 2 to r = 8 at fixed T.
- **F3, nothing uses long range:** no architecture solves r = 8 at T = 32.
- **F4, not specific:** the sign-blind control's mean R90 is at least 0.8 times the mean R90 of solved BGSD
  at r = 8, T = 32 (its AUC must also lie within 0.5 +- 0.05, by Lemma 4; otherwise the run is a leak).
- **Native depth effect:** on Bitcoin-Alpha, for each architecture, (D(T=32) - D(T=8)) / (D_unif(T=32) - D(T=8))
  at the prediction level. A value >= 0.5 for both architectures means depth does track range there.

Outcome: if F1, F2 or F4 holds, stop and redesign before any confirmatory compute (Section J). If F3
holds, the WWW claim changes and the paper is re-scoped before the freeze. Otherwise proceed.
Whatever the outcome, every number is reported as development evidence, never as a confirmatory result.
