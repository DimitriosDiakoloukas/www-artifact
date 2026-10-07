# Implementation provenance and artifact identifiers

This anonymous note resolves the `PROVENANCE.md` references in preserved source docstrings.
The private campaign-history document is omitted from the review package; this replacement describes
algorithmic provenance without project or author identifiers. Frozen source files are retained byte for
byte because measurement records pin their hashes. “Old harness” in those docstrings means an earlier
implementation from which an operator or parser was ported; it is not a citation to a published model.

- **SGCN:** balance-based signed aggregation; see Derr, Ma and Tang, Signed Graph Convolutional Networks,
  ICDM 2018. Standardised code uses a shared decoder/objective. Directed native checks use the complete
  PyTorch Geometric Signed Directed 1.1.1 model, with its own loss and decoder.
- **SIDNET:** signed diffusion; see Jung et al., Information Sciences 2022. Directed native checks retain
  the released recurrence/objective/decoder, with source hashes, deterministic indexed-sum equivalence
  checks and fixed evaluation initialisation. They do not reproduce the authors' benchmark.
- **SLGNN-style:** a specified attention encoder whose fidelity to the published SLGNN equations was not
  verified. Interpret the standardised results as that implementation's behaviour.
- **BGSD:** artifact identifier for the signed diffusion control with learned retention. No expansion of
  the identifier or separate published architecture is claimed. Its full operator is specified in the paper.
- **Data:** SNAP parsers form an undirected sign from summed reciprocal ratings in the standardised arm;
  zero sums/self-loops are dropped. Native ordered-pair processing is separate. Pinned processed hashes
  enforce the stated parsers. Core sign parameters, gradients and receptive-field tests are included.

The manuscript, protocols, source hashes, retained deviations and `collective/stage2/REVIEW_ERRATA.md`
distinguish published algorithms, standardised implementations and later native checks. Timing metadata
has a historical default-device-name limitation; use the execution ledgers for the actual GPU assignment.
