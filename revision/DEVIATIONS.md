# Execution notes
The queries, checkpoints, exhaustive enumeration, thresholds and native tuning grid remain as specified.
During the exhaustive audit, relation-block sharding was added for the final, slower SLGNN-style
Bitcoin-OTC checkpoint after the other seven checkpoints finished. Four GPUs process disjoint blocks
using the same single-graph scalar forward; existing blocks are retained. Every process checks the
intact logits bitwise and the final collector requires complete, non-overlapping coverage.
Disjoint-graph batching was investigated only as a numerical-equivalence test: it changed logits by
up to 9.54e-7 and was not used to produce any reported intervention effect.

Before collecting the new results, bookkeeping was hardened to retain undefined all-zero native
gradient profiles as null (never zero) and to report their exclusion counts. Source hashes are
captured when each process imports its implementation. These changes do not alter training,
query selection, derivatives, finite effects, or the tuning decision.

The released SIDNET CUDA sparse-matrix product was not bitwise repeatable even under deterministic
PyTorch settings: repeated identical forwards differed by about 6e-7 with unchanged weights and
fixed M0. Both COO and CSR kernels showed this. The released encoder and decoder are retained,
but their matrix products are dispatched to deterministic indexed message sums implementing
the same weighted sums and recurrence. Independent dense-recurrence and gradient tests verify
the mathematics. All 72 initial COO SIDNET tuning records and their selection are retained in
revision/superseded-coo; they are excluded from the final selection. SIDNET tuning is repeated
on the identical grid and seeds using the stable backend. Four initial SIDNET evaluation attempts
failed the post-gradient bitwise check and produced no accepted records; logs are retained.
SGCN code, tuning settings and accepted records are unchanged.

The final repeated selection leaves all six SGCN model/network/depth settings identical to
the retained original calibration. Fifteen SGCN evaluation records had already completed
under those settings; they are retained with their original selection hash. The archived
selection and final selection document that only SIDNET calibration was replaced. No test
result was consulted in choosing hyperparameters.
