# Next stage: prospective design, not additional evidence
The completed pilot supports an expanded study. This draft must be frozen before its new measurements;
none of the proposals below is reported as an experimental result in the current pilot or manuscript.

## Resolve the pilot's intervention limitations
Run the published-objective SGCN and SIDNET systems with random inputs, with depth-specific validation
selection, on Bitcoin-Alpha and Wiki-Elec. Use several seeds and a uniformly sampled cohort from the full
eligible evaluation set. Keep these results separate from the pilot's standardized-objective checkpoints.
Compare samplers at matched realised Hamming budgets: independent/fixed-size changes, prevalence
exchange and a more efficient alternating-cycle sampler. Report infeasibility for every query/radius,
including zero eligible signs or minority counts that cannot support the requested exchange. Do not
interpret failed constrained sampling as model stability. Retain original-prediction fidelity alongside
completion variance and distinguish the specified stress Q from a learned or true conditional Q.

## Separate dilution, redundancy and saturation
Vary one factor at a time: required distance, path count, correlation/redundancy, vote agreement and
unsigned distractor branching. Keep near topology, source markers and noise independent of the label.
Retain an exact solver and report failed learning separately from failed measurement.

A useful additional redundancy control compares two solvers on the same correlated support X_j=Y:

    ell_majority = 2 sign(sum_j X_j) + 0.5 Z
    ell_mean     = (2/k) sum_j X_j + 0.5 Z.

They coincide on the true Q and have the same conditional variance, but differ on off-support single
flips. With k=101, the mean solver's terminal gradient is 2/101 and its single-flip logit effect is 4/101;
both are below 10% of the local nuisance's corresponding effect (0.5 and 1). Their total distant mass
remains substantial, and joint latent-label changes remain decisive. Include every path connector in
the profile through its path product. This analytically specified comparison separates many weak
sensitivities from the zero pivotal sensitivities in the existing majority solver. It is a proposed
control, not a trained-model finding or a new general sensitivity theorem.

## Test an actual computation decision
Freeze development, policy-validation and held-out query partitions before policy selection. Disclose
historical benchmark exposure; held-out policy evaluation must not feed back into its design. Measure
actual neighbourhood cropping and propagation truncation, including fidelity to full predictions,
true-label AUC, Brier/calibration, runtime and peak memory. Compare fixed shallow/deep computation and
simple degree/triangle rules. Include the diagnostic's sampling and selector overhead.

For SIDNET cropping, preserve each retained node's original evaluation M0 by global node identity;
renumbering and drawing a new M0 changes the function. Specify boundary normalisation and account for
query batching/caching in runtime. Stress stability does not guarantee crop stability. Demonstrate an
actual accuracy/fidelity versus total cost improvement before treating the diagnostic as a useful
computation allocator. A published global-communication architecture is a supporting check if feasible.

Only after these checks should the paper's central narrative change. Its contribution should be the
validated mechanism and practical finding, with the standard variance estimator explicitly attributed
to prior sensitivity analysis. The current pilot supplies motivation and a reproducible foundation.
