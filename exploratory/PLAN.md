# Exploratory studies (outside the pre-registered claims)

Written 2026-10-03, after the confirmatory protocol was locked (commit bef7282) and before any of these
studies ran. They run after the confirmatory campaign, on their own seeds (20000-20004), write only under
`exploratory/`, and are reported in the paper as exploratory. Nothing here changes a confirmatory run,
hypothesis or analysis. Code lives under `exploratory/` and imports `src/srange` read-only; the locked
runners in `scripts/` are not modified.

## E1. Why the feature Jacobian falls short: a mechanism test

Development showed that on signed relay a model that provably uses the source at distance r places little
feature-Jacobian mass on it. Proposed mechanism: a model is sensitive to the stance channel of every node on
the chain, but in the data that channel is zero everywhere except at the source, so training never penalises
the sensitivity to intermediate nodes, and the Jacobian counts it as influence.

Variants of signed relay (b = 0, M = 1000/500/1000), r in {4, 8}, T = 32, SIDNET and BGSD:
- `signed`: the task as in the confirmatory protocol;
- `unsigned`: every sign +1, label y = s (no sign product); tests whether the shortfall is specific to signed
  propagation;
- `noisy-channel`: unsigned, and every non-source node carries an independent uniform +-1 in the stance
  channel, so sensitivity to intermediate stance values now hurts the loss.

Prediction of the mechanism: among solved cells, the feature-Jacobian R90 falls short of r in `signed` and
`unsigned` and is closer to r in `noisy-channel`. Reported whatever it shows: test AUC, feature-Jacobian D and
R90 (prediction and embedding level), and the mass at the source distance.

## E2. A global-attention contrast: SE-SGformer

SE-SGformer (Li et al., AAAI 2025) attends over all nodes, so its receptive field is the whole graph. Ported
from the official code (`liule66/SE-SGformer`, commit 6b995c6), with the shared pair head and objective of the
other architectures in place of its K-nearest-neighbour decoder (documented deviation). Dense attention limits
it to graphs of at most about 12,000 nodes: Bitcoin-Alpha, Bitcoin-OTC and Wiki-Elec at its own depth, and
both trust-chain tasks with M = 200/100/200 instances. The same reduced-M trust-chain cells are run for SIDNET
and BGSD so the comparison holds instance count fixed. Reported: test AUC, sign-flip and feature R90 against
the required distance, and range on the networks.

## E3. Chains planted in real trust networks (written before any E3 run)

Concern addressed: the trust-chain tasks are synthetic. Each instance's target end p_0 is attached by one
edge (random sign) to a uniformly drawn node of a real network (Bitcoin-Alpha or Wiki-Elec, full
undirected graph with its real signs); the rest of the chain is as in signed relay (b = 0), the source at
p_r. Real nodes carry noise features only. The chain is the only path from the target to its source, so
Lemmas 1 to 3 hold unchanged: nothing within radius r - 1 of the target predicts the label. Train,
validation and test graphs share the real backbone (it carries no label information) and have
independent chains. r in {4, 8}, T = 32, SIDNET and BGSD, M = 1000/500/1000.
Reported: test AUC; sign-flip and feature R90 against the required distance, now measured on graphs whose
neighbourhoods are real Web structure.

## E4. Does restart set the usable range? (written before any E4 run)

Development: only SIDNET (restart 0.15) learned signed relay at r = 8; nothing learned r = 16; BGSD
stopped at r = 4. Dose-response, everything else fixed:
- SIDNET restart c in {0.05, 0.15, 0.35}; BGSD retention-logit bias in {0, -2, -4} (initial retention
  about 0.5, 0.12, 0.02);
- signed relay b = 0, r in {8, 16}, T = 32;
- the same knobs on Bitcoin-Alpha and Wiki-Elec at T = 32 (spectral features, the frozen learning rate
  and weight decay of each architecture and network).
Reported: largest solved r and test AUC against the knob on the chains; sign-flip D and R90 against the
knob on the networks, with test AUC. Prediction (not a claim): lower restart or retention extends the
usable range on chains, while on the networks range stays short whatever the knob.

## E5. Distant pairs: does propagation help where local evidence is absent? (written before any E5 computation, 2026-10-04)

Concern addressed: trust propagation is meant for pairs of users with no direct evidence about each
other, but most test relations close a triangle in the training graph, where local features are strong.
A model that propagates trust should gain most, relative to local evidence, on distant pairs.

Strata: the hop distance between the two endpoints of a test relation in the message-passing (training)
graph: 2 (a common neighbour), 3, and >= 4; unreachable pairs are counted and reported separately. A test
relation is never in the message-passing graph, so the distance is at least 2.

Models: every confirmatory spectral checkpoint at T in {2, 8, 32} on all six networks (4 architectures,
5 seeds; 360 checkpoints), read from the store; the test logits of every checkpoint are regenerated and
must match the record bitwise. The local-evidence baseline is refit on the same split with the locked
code path; its overall test AUC must equal the stored local-baseline record, or the analysis stops.

Reported per (network, stratum): the number of test relations and the positive share; per architecture
and T, the test AUC on that stratum and its difference from the local baseline on the same relations,
with a seed-level 95% interval. A stratum with fewer than 50 relations of either sign is not scored.
Read descriptively: (a) in strata 3 and >= 4, does any architecture's interval lie above the local
baseline; (b) is the gain from T = 8 to T = 32 larger on distant than on near pairs. Nothing is retrained.

Deviation (2026-10-04, 10:30 UTC). The plan required the refit local baseline's overall test AUC to equal its
stored record. On Slashdot its logistic regression is not bitwise reproducible: repeated fits differ from the
record by 2e-9 to 5e-9 in AUC, with or without the thread settings of the original runs. The check now
requires agreement within 1e-6 and stores the difference per unit. Seen before this change: the counts and
AUCs of the four smaller networks and of one Slashdot seed (an interim collection).
