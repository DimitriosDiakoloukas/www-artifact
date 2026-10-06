# Layerwise computation follow-up
Exploratory design motivated by the PRIMARY DEVELOPMENT results: uniform K8 did not satisfy validation
fidelity, although strong sign sensitivities remained nearby. Specified before measuring any layerwise
candidate and before either study's held-out approximation results. The primary design and its failed
budgets remain intact. This is an adaptive development follow-up, not a retroactive preregistration.

Use all 12 SIDNET checkpoints from COMPUTATION_PROTOCOL. Keep first/last-layer step schedules (16,16),
(16,8),(16,4),(16,2),(8,16),(4,16), unchanged weights, X and global M0. Comparing (16,8) with (8,16)
holds total steps constant and tests the role of each layer's initialization. Source-degree operators,
BN evaluation and decoders remain unchanged. Full custom scores must agree with the source within 2e-6
and the original full-test function must remain bitwise unchanged before/after every bounded pass.

Reuse the first 80 primary development queries for development latency/magnitude summaries only; their
approximation results have already been inspected and this reuse is disclosed. Draw a NEW 80 eligible
validation queries, excluding ALL 160 primary development/policy-validation queries, using seed
20261012+model seed. These validation relations were also eligible for model epoch selection; they are
independent of computation-policy fitting, not of model selection. Use source labels only in this new
policy validation. Select the fastest schedule satisfying the primary criteria: probability RMSE<=0.01,
class disagreement<=0.01, AUC loss<=0.005 and Brier increase<=0.005, otherwise (16,16). Retain all candidates.
No parameter/model training or diagnostic threshold is fitted in this follow-up.

Commit and hash every schedule choice before evaluation on the SAME 128 held-out query positions
specified for the primary study. The primary held-out results have not been measured when this follow-up
is designed. Keep these results separate and identify their adaptive-development origin in the manuscript.
Do not revise either policy after held-out testing. Report fidelity, true-label AUC, Brier/ECE and measured
runtime/peak GPU allocation for batch sizes 1,8,64. Costs include the actual development and validation
execution ledger and online grouping. Model/graph residency is shared with the full baseline. Identical-
snapshot embedding-cache hits are not assumed; no production, future-snapshot or cache-saving guarantee
is inferred from this fixed-snapshot test. Record the full single-query/batched comparison and break-even
workload assumptions. Failed schedules are part of the result, not grounds for further held-out tuning.
