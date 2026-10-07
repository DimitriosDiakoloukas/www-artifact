# Retrospective review corrections — 7 October 2026

These corrections concern reporting and interpretation. Frozen protocols, worker sources, policies,
measurements, numerical arrays and checkpoints have not changed. All new analyses below are post hoc,
prompted by an external review; they do not acquire confirmatory status by regenerating cleanly.

1. **Synthetic result scope.** The learned 101-path counterexamples establish diagnostic failure under
   specified synthetic laws. Native Web edits are bounded stress tests, rather than measurements of the
   population prevalence of hidden distant dependence. The title, abstract and conclusions now distinguish them.
2. **Normaliser and threshold.** Saved arrays reproduce all-relation misses of 71/90/99 gradient and
   27/57/78 finite profiles at thresholds .05/.1/.2, out of 100 qualifying SGCN queries. Restricting the
   normaliser to terminal votes plus nuisance gives 42/71/86 and 16/37/52. This excludes fixed connectors;
   it does not make independent single-vote flips valid under the correlated redundancy law. Incident fixed
   connectors supply 80 gradient and 79 finite maxima. At .1, finite misses split 10/40 consensus and 47/60
   redundancy. `large_collect.py` retains every run's sweep, including runs below the qualifying threshold.
3. **Mechanism association.** Known majority/redundancy/saturation functions illustrate mechanisms; they
   do not identify each learned mechanism. For the two qualifying consensus SGCN runs, Pearson correlation
   of absolute vote margin with maximum terminal finite effect divided by maximum all-relation effect is
   -.646861 and -.693516 (20 saved queries each). This precise definition reproduces -.65/-.69, rather
   than the review's second -.86 value; differently defined associations need not coincide. Gradients miss
   on 38/40 consensus queries and finite reach on 10/40. No new causal mechanism claim is made.
4. **Native scales.** `collect.py` now reports equal-seed logit RMSE alongside probability RMSE and each
   checkpoint's median absolute reference logit. Unrestricted Alpha logit RMSE reverses the probability
   ordering (.604 SGCN, .672 SIDNET). Native losses, decoders and confidence differ, so model-level responses
   cannot isolate architecture. Logit scale is also decoder-dependent.
5. **Original hypothesis ledger.** The primary H1 rule and H2 hold; H1's auxiliary specificity fails.
   H3 holds for all four implementations. H4 is supported only for the diffusion control: median 4,
   29/30 at most eight, Holm p=1.15484e-7. SGCN (median 8, 19/30, p=.300733), SLGNN-style (median 8,
   18/30, p=.361595) and SIDNET (median 32, 0/30, p=1) do not meet the locked decision rule.
   The frozen `replication/PROTOCOL.md` sentence naming only SIDNET's H4 failure is incomplete; it remains
   unchanged to preserve its historical lock. RH1, RH2, RH3 and both RH4 parts hold, with RH4 restricted to
   individual relative reach. The manuscript reports every decision and distinguishes evidence tiers.
6. **Constrained-edit selection.** `review_diagnostics.py` replays the unchanged seeded cycle and
   count-matched exchange draws, asserting the stored achieved counts/acceptance/proposals. It reports
   edited-relation instances, endpoint degree and shell coverage at radius one/five percent. Median minimum
   degree is 16 versus 6 on Alpha and 48 versus 28 on Wiki; these explicit cohorts/statistics differ from
   some review values but confirm the selection bias. Matching count does not isolate degree preservation.
   A degree/location-matched prediction control would require new interventions and forwards, not just
   reanalysis of the current response arrays.
7. **Computation interpretation.** Crop failures lead the practical test; 24-step schedules still propagate
   globally and test layer placement. Cohort class counts, retained crop size, no-saving full fallbacks and
   the batching penalty for a globally selected per-query crop are explicit. These checks do not replace
   full-test performance evaluation or guarantee fidelity on future snapshots/cache hits.
8. **Hardware field.** Frozen `pv.environment()` records the default CUDA device name, not necessarily
   the worker's `--device`. Therefore the name field alone cannot identify the executing card. Primary
   execution ledgers assign cuda:0–3 to 32 GB V100s and cuda:4–7 to 16 GB V100s. Keep raw records intact;
   consult those ledgers for execution assignment. The paper states both memory capacities. Earlier
   DEVIATIONS wording implying that raw environment names record both sizes is corrected here.
9. **Sources and assistance.** Added primary foundations for joint influence, majority/pivotality,
   saturation and signed robustness; fixed the ICLR proceedings URL. The OpenReview range preprint's indexed
   abstract was available, while its full page required browser verification, so the comparison is limited
   to its stated margin-aligned one-hop diagnostic. No unverified acceptance status is asserted.
   The author confirmed that only OpenAI Codex contributed tool assistance to this WWW work.
10. **Public mirror.** A current local artifact is insufficient: the public anonymous mirror must serve the
    revised commit. Verify its generated files after refresh; do not infer that state from a push alone.

Reproduction: run `collective/stage2/report.py --paper <copy>` and `final_audit.py --paper <copy> --out <path>`.
`RESULT_MAP.md` and full empty-directory regeneration cover the four new generated disclosure tables.
