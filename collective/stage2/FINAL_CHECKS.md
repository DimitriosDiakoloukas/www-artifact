# Final expansion verification — 2026-10-06

All experiment and runtime cohorts are complete. No result-dependent policy changes followed either
held-out computation study. The layerwise design follows primary development and remains an explicitly
exploratory, prospectively locked follow-up.

## Completed coverage

- 144 known-function configurations and three saturation controls; both earlier nine-path learned controls.
- 144 random-input native tuning runs, 60 evaluation weights and 60 complete 100-query gradient profiles.
- 20 deep native joint audits: 19,200 prediction slots; eight complete matched-constraint shards: 76,800 slots.
- Every one of the 12 larger learned-control training outcomes, 240 audited queries and final audit_v2 arrays.
  Seven runs qualify; the five partial solvers remain. Strict full-function numerical checks and three
  full-union fallbacks are retained alongside the superseded bridge attempt.
- Primary and layerwise candidate studies on the same separate 128-query test cohort for each of 12
  checkpoints. All 204 policy/schedule outcomes have measured runtimes at batch sizes 1, 8 and 64:
  612 batch measurements, each with one warm-up and five synchronised repeats.
- End-to-end development costs plus measured validation-only selection-process costs; full fallbacks
  have no claimed break-even even when timing noise makes a ratio slightly exceed one.

## Independent checks and gates

`results/integrity.json` verifies native profile hashes, validation-only native selection, original
protocol commit objects, disjoint calibration subsets, test exclusions and both prospective policy locks.
`results/native_verification.json` checks all 60 native test predictions bitwise and four complete
100-query gradient arrays. `results/independent_replay-0..3.json` reconstructs constrained and native draws
and the original full-union predictions and representative valid completions for every learned checkpoint.

`final_audit.py` independently rethresholds all 100 qualifying learned SGCN gradient vectors, recomputes
four native short-reach exchange RMSEs from raw draws and two selected-schedule RMSEs from all raw test
vectors. Its report contains 29 printed-number spot checks and seven raw re-derivations, all passing.
The native joint seed means are computed per checkpoint before taking an equal-seed average.
Four additional absolute-latency checks independently read the full and validation-selected
single-query timing records, convert seconds to milliseconds and average the five checkpoint medians.
Latency means and mean paired speedup ratios are distinct summaries.

The scoped test suite passes all 33 tests, including binary native fidelity, derivatives, constraints,
source-degree boundary handling, global initialisation preservation, law-specific conditional variance
and separation of fidelity from label performance. The provenance test that opens unrelated earlier
projects is excluded from this WWW audit and the anonymous artifact.

The complete regeneration command reproduces all 44 paper outputs byte for byte from an empty temporary
output directory, with zero stale, missing, unmapped or failed outputs. RESULT_MAP matches every output.
Removing either numbers.tex or stage_numbers.tex from a disposable paper copy fails the result-map gate;
a deliberately changed stage_numbers.tex fails as stale. Historical removed figures remain explicitly
marked historical and are not required outputs.

## Claim boundaries

Relative individual sensitivity is not predictive necessity or a collective locality certificate.
The finite proposition requires correctness with margin on all specified neighbours and does not
transfer to derivatives. Known conditional laws support mathematical dependence claims; Web edits
support law-specific stress responses. Signed-degree constraints and achieved-size controls are distinct.
Native findings concern two models on two networks. Calibration and batching costs are part of the
practical result; unchanged-snapshot embedding-cache hits and future-graph policy fidelity are untested.
Conditional variance, fidelity diagnostics, adaptive inference and signed walks are credited prior art.

## Discovery-first presentation revision

The manuscript leads with learned failures, then native collective responses, then computational
consequences. Main-text constraints, timing hardware and absolute latencies come from existing records;
the familiar short-walk comparison is supporting appendix evidence. No training, intervention,
measurement or locked policy source changed. The revised report regenerates all 44 assets from an
empty directory with zero problems. All scientific arrays and existing reported numerical results
remain unchanged; four latency macros are added.

The latest external review also proposes native short-reach signed-degree/coverage-matched edits and
full-test global schedule evaluation with paired uncertainty. Neither addition was run in this editorial
revision. Separate standardised constraint cohorts and 128-query computation cohorts are explicitly
identified in the paper.
