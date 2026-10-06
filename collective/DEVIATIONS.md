# Pilot scope and analysis disclosures
The locked experimental choices are unchanged. No query, seed, radius, strength, draw count or mechanism
was selected or removed after inspecting results. All registered primary cells and all constrained
supplementary attempts are retained. No training or measurement execution failed.

The first pilot uses the existing standardized random-feature SGCN and SIDNET implementations, with the
shared training objective and decoder. This scope was explicit in the protocol before execution. It does
not replace a random-feature replication of the separately validated published-objective systems.

After the Bitcoin-Alpha results completed, the report added a descriptive empirical decomposition:

    mean_draws (p - p_original)^2
      = variance_draws(p; divisor K) + (mean_draws p - p_original)^2.

The equality is exact for the saved draws; averaging it over fixed queries separates completion
variability from mean prediction drift. Neither component is a new test or certificate. The paired
variance estimates and their pre-specified 80-cell simultaneous intervals are unchanged. This addition
was motivated by large changes from original predictions alongside small paired-completion variance.
It is a post hoc descriptive analysis and must remain labelled as such in any future manuscript.

A broader literature check identified variance-based global sensitivity analysis as a direct
mathematical predecessor. Under a product Q the population diagnostic equals an unnormalised group
total-effect variance. This strengthens the novelty constraint; no new generic variance estimator
or sufficiency theorem is claimed.

After the trained controls finished, their saved completion logits were additionally evaluated against
each completion's known synthetic majority label, reconstructed from the frozen generator and RNG keys.
Accuracy at the fixed zero-logit threshold and AUC are descriptive checks of learned-function behaviour;
they do not select a new checkpoint or threshold. They differ from original-label stress performance:
synthetic counterfactual labels are known, whereas real-network counterfactual user labels are unknown.
