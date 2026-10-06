# Collective signed dependence: completed exploratory pilot

The first stage tests whether short individual reach survives joint changes to distant signs. The checkpoint and query choices, radii, mechanisms and draw counts were frozen before measurement. This is a small exploratory study of four standardized random-feature checkpoints, not a replication of native published objectives. It does not change the current manuscript or establish a submission score.

## Real networks

All original full-test logits match their stored hashes before and after measurement. There are 80 selected queries, 25,600 primary perturbed predictions and 480 supplementary predictions. The table uses strength 0.5. Probability RMSE is measured from the original prediction; disagreement counts changes to the predicted class. Exchange strength specifies the fraction of the smaller sign class exchanged, rather than an independent flip probability.

| Model / network | Mean individual reach | Independent: RMSE / disagreement beyond radius 1 | Exchange: RMSE / disagreement beyond radius 1 | Selected negatives / positives |
|---|---:|---:|---:|---:|
| SGCN / bitcoin_alpha | 0.30 | 0.2742 / 11.2% | 0.0523 / 1.2% | 4 / 16 |
| SIDNET / bitcoin_alpha | 1.00 | 0.1936 / 5.3% | 0.0561 / 1.9% | 4 / 16 |
| SGCN / wiki_elec | 1.30 | 0.3998 / 40.0% | 0.3915 / 35.9% | 11 / 9 |
| SIDNET / wiki_elec | 0.35 | 0.1732 / 15.0% | 0.0370 / 0.6% | 11 / 9 |

Short individual reach does not guarantee stability to the specified joint changes. Restricting to queries whose individual reach is at most one hop still gives measurable effects when signs beyond one hop are exchanged while preserving global sign prevalence. The following table reports that pre-specified subset for every checkpoint.

| Model / network | Queries with individual reach ≤ 1 | Exchange probability RMSE | Exchange disagreement |
|---|---:|---:|---:|
| SGCN / bitcoin_alpha | 20/20 | 0.0523 | 1.2% |
| SIDNET / bitcoin_alpha | 18/20 | 0.0591 | 2.1% |
| SGCN / wiki_elec | 9/20 | 0.3277 | 22.2% |
| SIDNET / wiki_elec | 20/20 | 0.0370 | 0.6% |

These are stress-law effects on one checkpoint and a fixed 20-query cohort per setting, not causal effects or population estimates across models and networks. Exchange and independent flip strengths produce different realised Hamming fractions. For strength 0.5 beyond radius 1, the exchanged fraction is approximately 9% on Bitcoin-Alpha and 22% on Wiki-Elec, versus 50% for independent flips. The smaller effects under exchange therefore do not isolate a benefit of preserving prevalence. A later comparison needs matched perturbation sizes.

The empirical squared change is also decomposed into variation across completions plus the squared drift of the mean completion prediction from the original. This descriptive decomposition was added after the first Bitcoin-Alpha results; it is not a registered hypothesis or a new uncertainty guarantee. Low completion variance can coexist with a large, consistent shift from the original prediction.

Each primary mean variance has a simultaneous 95% Hoeffding interval over the 80-cell family. Those intervals cover Monte Carlo error for the fixed queries and chosen stress sampler; they do not cover query sampling, seed variability or distribution misspecification. Pointwise observed-configuration curves need not decrease. The pilot cannot certify negligible dependence from a small point estimate alone. Full effects, AUC, Brier scores, achieved Hamming fractions, individual-reach subsets and local strata are in summary.json.

The signed-degree sampler preserves every node’s positive and negative degree by alternating four-cycle flips. Its achieved perturbation can be much weaker than the requested target; accepted cycles, proposal counts and no-change draws must accompany its effect estimates. Here it accepted only 57 cycles across 480 draws; 423/480 draws changed no signs. The achieved perturbation is too weak to establish stability under substantial signed-degree-preserving changes. An effective constrained sampler is required next.

## Known functions and learned distributed control

The controls separate binary dependence from the continuous relaxation used for gradients. Their unsigned graphs and features contain no label information within a radius below the required distance. The known solver has margin at least 1.5. Individual measures can miss majority consensus, redundant evidence and saturation, while true conditional completions recover the remaining prediction variance. This conclusion concerns the stated synthetic Q.

| Control, required distance 3 | Paths | Configurations where gradient reach misses distance 3 | Single flips miss |
|---|---:|---:|---:|
| chain | 1 | 0.0% | 0.0% |
| consensus | 9 | 50.8% | 50.8% |
| consensus | 31 | 71.9% | 71.9% |
| consensus | 101 | 84.8% | 84.8% |
| redundancy | 9 | 100.0% | 100.0% |
| saturated | 1 | 100.0% | 0.0% |

Consensus distributions with at most nine paths are enumerated exactly. Larger consensus fractions use 8,192 fixed-seed configurations. The redundant distribution is enumerated exactly. All 27 control settings also pass the independent paired-completion check against their analytic population variance.

| Trained consensus, nine paths at distance 3 | Selected epoch | Validation AUC | Held-out AUC | Gradient recovery of distance 3 | Single-flip recovery of distance 3 |
|---|---:|---:|---:|---:|---:|
| SGCN | 50 | 1.0000 | 0.9979 | 20/20 | 20/20 |
| SIDNET | 145 | 1.0000 | 0.9993 | 20/20 | 20/20 |

Both training controls use independent train, validation and test graphs, and checkpoint selection uses validation AUC only. They use the common node objective and are not tests of native link-prediction objectives. Their true-Q completions retain the control’s fixed connectors. The report also reconstructs each completion’s synthetic majority label and reports prediction accuracy and AUC against that changed label. This post hoc diagnostic uses the known control generator; real-network stress tests retain observed labels because their counterfactual truth is unknown. High AUC does not imply perfect classification at the fixed zero-logit threshold.

## Interpretation and next stage

The pilot establishes a reason to continue: individual reach can be short while joint changes beyond it alter predictions, including under prevalence-preserving exchanges. The known controls supply explicit failure cases, while both learned consensus controls recover distance three with gradients as well as finite flips. The network pilot measures sensitivity under specified stress distributions. It cannot identify causal changes in user behaviour or prove that a cropped computation preserves predictions.

Next, expand native-objective random-feature checks and controlled mechanism sweeps, then freeze a development/validation/held-out computation-selection experiment. Test the actual approximation against fixed shallow/deep and degree/triangle rules, including diagnostic overhead, fidelity, AUC, calibration, runtime and memory. The current paper and anonymous artifact remain the validated submission baseline until that evidence is complete.

See METHODS.md for the probability identities and assumptions, LITERATURE.md for prior work, and RESULT_MAP.md for the complete input/output mapping.
