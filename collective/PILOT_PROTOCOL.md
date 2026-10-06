# Collective signed dependence: exploratory pilot
Specified before new measurements. Dates and records use UTC; the authors' working timezone is Athens.
This is a review-motivated pilot, not an independently preregistered replication. It extends the validated
WWW study without changing its registered decisions or the current manuscript's claims.

## Question and scope
When individual sign-sensitivity reach is short, do predictions remain stable under joint changes to
more distant training signs? Report either outcome, including failures and inconclusive estimates.
The first pilot uses existing RANDOM-feature SGCN and SIDNET checkpoints on Bitcoin-Alpha and Wiki-Elec,
T=32, seed 10000. These are the standardised encoders with the shared objective and decoder, not the new
published-objective spectral-input systems. The latter's random-input arm is a subsequent study.
These two networks cover trading and voting; these two models are established architectures; the first
registered seed is selected for feasibility, not accuracy or reach. targets.json freezes records/hashes.

## Network pilot
- Draw 20 of each checkpoint's 100 stored eligible queries uniformly without replacement, sorted into
  stored order, using NumPy default_rng(20261006). Never select queries by effect, correctness or confidence.
- Preserve topology, random features, weights, normalisation implementation and the evaluation function.
  Verify every original test logit bitwise before and after measuring. One sign parameter per undirected
  training relation is shared by both message-passing orientations.
- Distance is the minimum endpoint-to-relation hop distance in the training graph, as in the main study.
  Radii: -1 (preserve none), 0, 1, 2, 3. All relations beyond the radius, including unreachable relations,
  are candidates. Do not impose a propagation-distance cap on interventions.
- Primary mechanisms, each at strength 0.1 and 0.5, with 16 independent draws per query/radius/strength:
  (a) independently flip each eligible sign with the stated probability;
  (b) exchange k positive and k negative eligible signs, sampled uniformly without replacement, with
      k=floor(strength * min(number positive, number negative)). This preserves global sign prevalence;
      report the achieved Hamming fraction, which differs from the independent-flip parameter.
- Supplementary signed-degree-preserving mechanism: flip alternating signs on simple four-cycles wholly
  outside the preserved radius. Five uniformly selected pilot queries per checkpoint (the first five of
  the pre-specified sorted subset), radii 0,1,2, strength 0.1, eight independent draws. At most 5000 cycle
  proposals/draw, target floor(0.1 * eligible relations/4) edge-disjoint cycles. Report accepted cycles,
  achieved Hamming fraction and no-change draws. This is a constrained stress sampler, not uniform
  conditional sampling over signed-degree-equivalent graphs; infeasibility is an outcome.
- RNG seeds derive only from fixed checkpoint index, query slot, radius, mechanism, strength and draw.
  Save every changed logit and realised Hamming count, original logits/labels, query positions, distances,
  gradients and local strata. Resume independent query/radius/strength cells atomically in bounded
  processes of approximately 120 seconds. Save array hashes and require complete coverage to collect.
- Report probability/logit absolute change and RMSE, prediction disagreement, original-label AUC/Brier
  under interventions, and joint effects conditional on individual reach <= preserved radius. Undefined
  AUC for a single-label sample remains undefined, with class counts reported. Labels remain the observed
  test labels: stress-test performance does not estimate counterfactual user behaviour.
- Report common-neighbour count and minimum endpoint degree strata descriptively. One checkpoint per
  model/network and 20 queries cannot support seed-level or broad network-generalisation claims.

## Pair-variance diagnostic and uncertainty
For each fixed query/local configuration, pair independent completions and average
0.5 * (p_1-p_2)^2. This estimates variance under that cell's chosen stress distribution. Also report
changes from the original prediction: zero variance under a degenerate intervention does not imply
fidelity to the original. Sigmoid logits are model scores, not presumed calibrated posteriors.
The primary family has 4 checkpoints * 5 radii * 2 mechanisms * 2 strengths = 80 cells. Aggregate over
all 20 fixed queries and their eight independent pairs. Report a Hoeffding sampling-error interval
with simultaneous 95% coverage across these 80 means (Bonferroni), and unadjusted intervals descriptively.
Terms lie in [0,1/2]; the variance itself lies in [0,1/4]. These intervals condition on selected queries,
checkpoint and intervention distribution. They do not validate Q or cover query/seed sampling uncertainty.
Do not assume the observed-local-config curves are monotone. Population monotonicity requires nested
conditioning under one fixed Q; see METHODS.md.

## Known-function controls, before interpreting trained models
All unsigned topology, degrees, source markers and random noise are independent of the balanced label.
A target has informative relations at required edge distance d in {1,3,5}, with an independent nuisance
relation at distance zero. Fix nuisance amplitude 0.5 and label signal amplitude 2 (margin >=1.5).
- Sparse chain: independent signs along one path, label their product. Its exact polynomial solver has
  distant individual sensitivity and finite effects.
- Distributed consensus: odd k in {1,9,31,101} equal-length paths; connectors are positive, terminal signs
  are independent uniform, label their majority. Use the multilinear extension of majority for exact
  vertex gradients and binary majority for predictions. Many configurations have no pivotal single bit.
- Redundancy: k in {9,31,101}, same topology, every terminal sign equals a shared uniform latent label.
  Sample conditional completions from this true correlated Q, not independent off-support signs.
- Saturated chain: identical binary solver, but each sign is relaxed through (3s-s^3)/2. Its gradient
  vanishes at binary chain signs; finite and collective changes remain informative.
Compare individual reach, gradient and exhaustive single-flip mass radii, and pair variance. Nuisance
sensitivity ensures that zero far influence is distinguishable from undefined all-zero profiles.
Enumerate distributions exactly where feasible (k<=9); use 8192 fixed-seed configurations otherwise.
Paired-completion sampling also uses 8192 pairs per control/radius. Check its expectation against the
analytic conditional variance and its finite-sample error bound. Proofs state the precise control Q.

## Training one distributed regime
After known-function validation, fit SGCN and SIDNET on consensus k=9,d=3 with T=8, hidden width 32,
16 input features (source marker plus independent Gaussian noise), Adam lr=0.005, decay=1e-5, clipping=1,
maximum 200 epochs, validation every five epochs and patience ten. SIDNET restart=0.15; fresh training
M0 and fixed evaluation M0 as in the original code. Train/val/test graphs are independent draws with
256/128/256 instances, seeds 20261100/20261101/20261102; model seed=61000. Common node head and balanced BCE.
Select solely on validation AUC, then report held-out AUC, intact-logit hashes and measurements on 20
uniformly drawn held-out targets. Record failed learning without attributing it to measurement failure.
No native-objective claim applies to these node-task training controls.

## Reporting and next phases
Every outcome and implementation fix is retained. There is no preferred direction and no new pass/fail
hypothesis. Do not promote this pilot into the main paper before an expanded study establishes validity,
mechanistic explanations and a practical consequence. Next: depth-specific native-objective random arms;
controlled distance/path-count/redundancy/disagreement/branching sweeps; one global architecture if feasible;
then actual neighbourhood-cropping/step-budget selection, developed and tuned before held-out testing.
The computation study must compare fixed shallow/deep and degree/triangle rules, include diagnostic and
selector overhead, and measure fidelity, AUC, calibration, runtime and memory of the actual approximation.
