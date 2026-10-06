# Implementation decisions recorded before measurements
The frozen protocol and targets remain unchanged. Independent stress draws are seeded by immutable
integer keys. A cell contains all 16 (or eight supplementary) draws and is written atomically.
Only complete, hash-verified checkpoint sets can be aggregated. The evaluation head is always called on
the same 20 queries; original full-test logits are verified separately before and after each process.
The source hash covers the measurement closure; collectors have their own source hashes.

Synthetic inputs use a source marker followed by 15 independent Gaussian noise columns with standard
deviation 0.1. No vote or label is included in features. Controls on independent sparse chains can be
integrated over their uniformly distributed path parity: absolute individual profiles are identical for
all internal sign configurations, and after conditioning on any proper prefix the remaining parity is
fair. Thus their analytic population quantities need no enumeration of all internal configurations.
Consensus configurations enumerate terminal votes and the independent nuisance; connectors are fixed +1.
The redundant control samples its shared latent label, respecting its correlated support. All connector
relations are included in gradient and single-flip profiles, even though they are constant under Q.

Trained distributed controls additionally use 16 true-Q completions per selected query at radii
-1,0,1,2,3,5, and exhaustive single flips on that query's independent component. Other components cannot
influence its output in evaluation (BatchNorm has fixed statistics and evaluation M0 is fixed).
Training state, optimizer and CPU/CUDA RNG are saved for exact continuation. Reporting these exploratory
measurements is descriptive; it introduces no new pass/fail rule or significance claim.
