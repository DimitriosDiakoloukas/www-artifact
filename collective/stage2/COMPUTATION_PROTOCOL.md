# Prospective computation study
Specified before any crop/truncation measurements or policy fitting. This is an exploratory follow-up,
not a new preregistration of previously seen benchmarks. The historical pilot and model-validation
results have been inspected. There is no prescribed positive speedup or acceptance outcome.

## Models and partitions
Use the ten new native random-input SIDNET T32 checkpoints (two networks, seeds 52000..52004), and
both original standardised random-input SIDNET T32 pilot checkpoints as supplementary comparisons.
Frozen weights, X, evaluation M0 and training-only adjacency throughout. Independently select 160
uniformly drawn eligible validation relations (degree positive at both endpoints); first 80 are policy
development, next 80 policy validation, using seed 20261009 + model seed. Validation relations were
used for model epoch selection; this reuse is disclosed. No model test label enters policy fitting.
Select 128 uniformly drawn eligible test relations with seed 20261010 + model seed, excluding every
query in the earlier native sensitivity cohort or pilot's stored 100-query cohort. Test candidates
are measured only after all policies have been selected, hashed, committed and locked. No redesign
after held-out results. Report class counts, infeasibility and failures, including poor AUC precision.

## Actual candidate computations
Full: two layers with K=16 diffusion steps each. Truncated: K=2,4,8 per layer, unchanged weights/BN,
native decoder and original M0. Cropped: radius r=0,1,2,3, with K=16 per layer. Retain nodes at undirected
training-graph distance <=r+1 from either query endpoint; retain relations of distance <=r, plus their
original self-loops. Keep original source-degree normalisation for all retained operators (missing
messages are zero, not renormalised). X and M0 are indexed by global node identity, never redrawn.
The native released forward and full custom recurrence must agree within 2e-6 absolute/relative;
full logits from the released forward must remain bitwise unchanged before/after every bounded pass.
Cropping is an approximation even when individual sign reach is small. Record retained nodes/edges.

## Policies, frozen before test
Fixed baselines: full, crop1, crop2, K4, K8. Range proposal: crop at ceiling(mean development rho_0.1),
clipped to 0..3, using gradients on the first 20 development queries. Collective proposal: smallest r
in 0..3 for which a prevalence-preserving exchange of m=2 floor(0.05*eligible/2) signs gives development
probability RMSE <=0.01 over the same 20 queries and eight independently seeded draws each. Infeasible
exchange draws are reported; any radius with an infeasible nonzero request is ineligible. This is a
stress diagnostic, not a conditional-sufficiency certificate. Report its mean drift, variance and
class disagreement. Neither proposal is automatically certified for actual cropping.
Calibrated global policy: fastest candidate satisfying validation probability RMSE <=0.01, class
disagreement <=0.01, AUC loss <=0.005 and Brier increase <=0.005 relative to full, else full.
Structural tree: multi-output DecisionTreeRegressor(max_depth=3,min_samples_leaf=10,random_state=20261009)
fit on 80 development queries to absolute probability errors of all candidates, using log1p(minimum
and maximum endpoint degree), common-neighbour count, and endpoint positive-sign fractions. Choose
cheapest predicted-safe candidate at thresholds 0.001,0.005,0.01,0.02,0.05; select fastest threshold
meeting the same validation criteria, else full. Degree and triangle rules: crop1 for min-degree below
its development median, or common-neighbour count above its development median, respectively; validate
these rules against the same criteria, else full. Keep ungated proposal results and rejection outcomes.
The computation policies and inference mechanisms are baselines/calibration tools, not new architectures.

## Costs and reporting
Record measured latency (CUDA synchronisation, one warmup, median of five repeats), total and incremental
peak allocated CUDA memory, and preprocessing/gradient/completion/fitting times. Baseline full and
fixed truncation can share one embedding computation across queries: report batch sizes 1,8,64; crops
are computed for each query, and cache hits across identical snapshots are not assumed. Warm parameter
and graph residency is shared across methods, disclosed separately from incremental allocation.
Include structural feature extraction, graph slicing and policy prediction in online times. No timing
based only on edge or step counts. Offline diagnostics/calibration are reported explicitly with
break-even workloads; amortisation is a scenario, not an observed production speedup.
Held-out reports: fidelity RMSE/max error/disagreement, true-label AUC, Brier, 10-bin equal-width ECE,
latency, memory, retained steps/nodes/edges, and all overhead. Probability is sigmoid(native logit),
not post-hoc calibrated. Describe outcomes separately by checkpoint and network; paired seed summaries
are descriptive t intervals. Do not substitute stress stability for tested approximation fidelity.
