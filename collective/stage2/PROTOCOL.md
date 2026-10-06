# Collective dependence: stage 2
Exploratory expansion specified before new runs. The original pilot and all earlier registered analyses
remain unchanged. Report all outcomes. No acceptance score or causal interpretation is prescribed.

## Published systems with random features
Use the unchanged directed-data parser, pinned native SGCN and SIDNET objectives/decoders, deterministic
SIDNET indexed-sum backend and depth mappings from revision/native.py (validated in test_revision).
Both networks: Bitcoin-Alpha and Wiki-Elec. Fixed independent 64-dimensional Gaussian random features,
unit-norm columns, generator seed 80000 + run seed; no spectral preprocessing or test signs enter X.
Keep the native training-only objectives, frozen random X, hidden width 64, restart constants and SIDNET
StepLR from revision/PROTOCOL.md B. Ordered relations, sum same-direction ratings, discard zero sums and
self-loops; 60/20/20 split. Each run uses its own split seed, equal to its model seed.
Depths 2,8,32. For every model/network/depth select lr in {0.001,0.005,0.01}, decay in {0,0.00001}, by mean
validation AUC on tuning seeds 51000 and 51001; tie break smaller lr then decay. Maximum 300 epochs,
validation every five epochs, patience 20 checks. Select epochs on validation only. Tuning computes no
test metric. Evaluate selected settings on seeds 52000..52004, only after all tuning records are complete.
There are 144 tuning jobs and 60 evaluation jobs. Bounded resumable processes retain RNG, optimizer,
scheduler and best selection state. Record source, protocol, split, feature and checkpoint hashes.
Measure sign-gradient reach at tau 0.05,0.1,0.2 on 100 uniformly drawn eligible test relations using
seed 9000 + run seed; do not condition on correctness, confidence or gradient. Report ceilings/dropped
profiles and seed-level t intervals, paired T8/T32 differences, native/relaxed binary forward tolerance,
and full-test logits bitwise before/after gradients. Keep any failures in an execution ledger.

## Efficient signed-degree sampler and equal-sized interventions
Work on the original pilot's four frozen random-feature checkpoints and the same 20 queries. Mechanisms:
fixed-size unconstrained flips, prevalence-preserving equal positive/negative exchanges, and signed-degree-
preserving alternating four-cycles. Radii 0,1,2,3; nominal eligible-edge Hamming fractions 0.02,0.05,0.1;
16 independent draws per query/cell. Seed keys start at 20261007 and include checkpoint/query/radius/budget/
mechanism/draw. Preserve all signs at distance <=r and original features, weights and evaluation M0.
Fixed-size/exchange counts are even, m=2 floor(fraction*eligible_edges/2). If an exchange cannot select
m/2 signs from each class, report infeasible, never silently reduce its budget. Four-cycle target is
floor(m/4) edge-disjoint cycles; maximum 20000 proposals/draw, record achieved Hamming count. Propose a
positive edge (a,b), a negative incident edge (a,c), and a node d with positive (c,d) and negative (b,d);
require four distinct nodes and unused eligible edges. This improves acceptance, but is a stress sampler,
not uniform conditional sampling of signed-degree-equivalent graphs. Report proposal/acceptance/no-change
rates. Compare effects at the same realised count where supported; incomplete cycle budgets are not
matched comparisons. Supplement fixed-size and prevalence-exchange draws at each cycle draw's achieved
count, with independent RNG. Their paired size is a descriptive stress comparison, not an intervention-
distribution certificate or causal attribution. Retain original-prediction fidelity and pair variance.
Original full-test logits must match before/after, and collectors require complete coverage and hashes.

## Mechanistic known-function sweep
Expand exact functions using distance d in {1,3,5}, path count k in {1,9,31,101}, and redundant votes X_j=Y.
Compare majority and mean-of-path-parities solvers ell=2 majority(votes)+0.5Z and ell=2 mean(votes)+0.5Z.
They coincide on redundant Q, while off-support single flips and continuous gradients differ. Include all
path connectors. Calculate gradient/finite maxima, mass R90 and true-Q variance. State when each fails;
no changed outcome is considered an error in a correctly defined individual measure.
For consensus, enumerate configurations where k<=9 and otherwise draw 8192 with fixed seed 20261007;
stratify by vote disagreement. Add fixed unsigned distractor stars of b in {0,4,16} at the target, with
independent fair nuisance signs, coefficient 0.5/b for their mean when b>0 (one nuisance edge when b=0).
Topology and features remain independent of fair Y; no predictor below d exceeds chance in population.
Report both gradient and finite profiles for every input, including distractors. Saturation uses the
original cubic binary-preserving relaxation. Compute exact population completion variance where possible,
independent paired-completion checks otherwise, with explicitly named Q. These are known functions, not
claims about trained systems. Select any additional trained regimes in a separately frozen follow-up.

## Practical application
The development/validation/held-out computation-allocation design must be frozen separately after these
checks. It tests actual cropping/truncation with fixed global-node M0, boundary normalisation specified,
fidelity, AUC, calibration, runtime, memory and all diagnostic/policy overhead. Compare fixed shallow/deep
and cheap degree/triangle selectors. Do not promote this expansion into the main paper before that
application supplies a validated practical finding. Historical benchmark exposure must be disclosed.
