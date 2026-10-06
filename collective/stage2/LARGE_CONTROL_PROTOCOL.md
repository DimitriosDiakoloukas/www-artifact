# Prospective learned high-redundancy and consensus controls
Follow-up to the known-function mechanism sweep and the earlier learned nine-path control. Freeze before
any additional training. Both successful and unsuccessful learning are retained; diagnostic failures
are interpreted only for learned predictors that use the distant signal, not for unsolved models.

Standardised SGCN and SIDNET node classifiers, hidden width 32, independent 16-dimensional input features
(one unsigned source marker and fifteen Gaussian noise coordinates of sd 0.1), T8. SIDNET has two layers,
c=0.15, and fixed global-node evaluation M0; fresh training M0. Two laws: independent 101-vote majority,
or 101 identical copies of a fair latent label. Each source is reached by a distinct four-edge path;
terminal signs are at distance three. All connectors are positive; a fair local nuisance sign is label
independent. Topology, markers and noise are label independent, so every radius below three has population
chance accuracy. The graph generator follows collective/controls.py, with k=101,d=3. Redundancy replaces
all terminal votes by the fair majority label of the original independent draw, leaving noise untouched.

Model seeds 61010,61011,61012. Train/validation/test instance counts 256/128/256; data seeds
20262000+10*model_seed+split_index. Adam lr=0.005, decay=1e-5, gradient clipping 1. Maximum 200 epochs,
validation every five, patience ten checks. Select checkpoint on validation AUC only. No tuning on test.
Class-weighted BCE as the existing learned control. Record all dataset arrays' hashes, trace, selected
state, test AUC and threshold accuracy, with bounded optimizer/RNG resumability and all failed attempts.
For descriptions only, a solved cell has test AUC>=0.99 and threshold accuracy>=0.95; do not hide other cells.

Draw 20 test instances uniformly using seed 20261011+model_seed. Compute complete sign gradients at
all input relations; check that disconnected components have exactly zero influence. Exhaust every
single flip in each selected query's whole component, including all connectors and nuisance. For fast
finite evaluation, an isolated component may replace the full disjoint union only after its original
scores agree within 2e-6 and all retained SIDNET M0 rows are copied by original global node identity.
Keep full test logits bitwise unchanged before/after every bounded process. Report component equivalence
error and maximum original/perturbed logit tolerance implications; full-score gradients use the full union.

Known-Q completions: radii -1,0,1,2,3, sixteen independent draws per query. Below three, redraw terminal
votes independently for consensus, or redraw the shared fair latent label jointly for redundancy. Redraw
local nuisance only at radius -1. All connectors, topology, X, weights and global M0 remain unchanged.
Report original-prediction fidelity, paired variance, gradient/finite reach, and accuracy/AUC against the
NEW labels implied by each completion. At radius three the original completion is fixed. Exact known
solver and population lower-bound claims remain separate from empirical learning and sampling claims.
Every cell and source/protocol/checkpoint hash must be present before aggregation. This supporting
experiment does not claim a new architecture or faithful published-model benchmark.
