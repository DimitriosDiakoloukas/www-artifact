# Artifact: Distant Dependence Can Appear Local

Collective Audits of Signed Graph Neural Networks

Code, protocols, run records and analysis for the WWW 2027 submission. Every table, figure and number of the
paper is produced by code from the records here; `RESULT_MAP.md` lists each with its generating script,
inputs and SHA-256, and `paper/` holds the generated outputs themselves.

## Layout
- `src/srange/` models, data (SNAP downloads verified by pinned SHA-256), influence measures, provenance.
- `confirmatory/` protocol, its lock (`PROTOCOL_LOCK.json`), job list, all 860 run records and the 30
  local-baseline records (890 in `confirmatory/*/runs`), logs, reruns.
- `exploratory/` the written plan (`PLAN.md`, E1-E8), records, analyses, and the post-hoc diagnostics with
  their timeline (`POSTHOC.md`).
- `replication/` the pre-registered replication: `PROTOCOL.md`, `LOCK.json`, code, records, measurements,
  analysis, reruns.
- `revision/` the reviewer-requested exhaustive finite-flip audit and directed published-objective
  checks: protocol specified before computation, complete intervention arrays, validation-only tuning,
  evaluation records, logs and execution deviations (including superseded SIDNET calibration records).
- `development/` the development studies that shaped the protocol, with their plans and outcomes.
- `collective/` the completed pilot, unchanged frozen sources, raw interventions and smaller learned controls.
- `collective/stage2/` the expanded known-function and learned-control studies, native random-input models,
  matched constraints, joint audits, locked computation policies and every held-out/runtime outcome.
  `DEVIATIONS.md` retains initial failures, unsolved controls and the prospective layerwise follow-up.
- `analysis/`, `reporting/` the scripts that write the paper's outputs; `paper/` those outputs.
- `checkpoints/objects/` a sample of stored checkpoints, content-addressed by SHA-256 (see below).

## Reproducing
Python 3.12, `pip install -r requirements.txt` (the exact versions used). Set `SRANGE_STORE=<artifact>/checkpoints`
(or another writable directory containing the packaged `objects/`) and run `python3 tools/fetch_data.py` once:
it downloads the six SNAP networks there and checks the raw and processed files against their pinned SHA-256.
- Tests: `python3 tests/run.py`.
- Full paper regeneration: `bash reporting/regen_check.sh paper` (requires a CUDA GPU, the packaged checkpoints
  and the fetched SNAP data). It regenerates all outputs in a temporary directory and compares them with
  `paper/`; success prints "0 problems". It uses `python3` from the active environment; set
  `PYTHON=/path/to/python` to select another interpreter.
- Records verify against their own hash: `srange.provenance.verify_record`.
- Analyses from records: `python3 analysis/chain.py confirmatory/chain --seeds 5 --out <dir>`,
  `python3 analysis/native.py confirmatory/native --out <dir>`, `python3 exploratory/analyze.py --out <dir>`,
  `python3 replication/analyze.py --out <dir> --measure replication/measure`; compare with `paper/generated/`.
- Measurements from checkpoints: with `SRANGE_STORE=<artifact>/checkpoints`, the sampled checkpoints load by
  hash, e.g. `exploratory/posthoc_reach_max.py`, `replication/measure.py`, `reporting/shells.py`. The
  replication gate (`replication/gate.py`) refuses to write under `replication/` in this copy, because the
  rewritten paths change the hashes it checks; write re-measurements under `development/`, e.g.
  `python3 replication/measure.py chains --runs replication/planted/runs --only <run> --out development/check`.

## Checkpoints
Original-campaign checkpoints are retained for final release. This artifact includes
156: the solved SIDNET chains at r = 8 planted in Web networks (exploratory E3 and the
replication), the four architectures on Wiki-Elec at T = 32 (including the per-shell figure), all eight
checkpoints used in the exhaustive audit, all 120 directed native evaluation checkpoints (spectral/random),
and all four standardised checkpoints used in the collective comparisons. The two smaller learned
controls and all 12 larger learned checkpoints are additionally bundled as `selected.pt` under their
training directories. All new studies can therefore be regenerated from packaged weights.
The original deterministic campaign was verified on NVIDIA V100 cards. The new native checks retain
weights, fixed evaluation functions and checked logits; they do not claim bitwise agreement with
other hardware or the original authors' benchmarks.

## Reviewer-requested validation
The main campaign uses undirected relations and a shared objective/decoder. The new checks use ordered
relations, released SGCN and SIDNET objectives and decoders, and validation-only tuning separately at
T = 2, 8 and 32. They are exploratory review extensions, not a new independent preregistered replication.
The pinned SIDNET encoder/decoder sources are downloaded by `revision/native.py` on first use; the commit
and file hashes are recorded in its `UPSTREAM` dictionary. Its weighted sparse products use deterministic
indexed sums of the same recurrence. All affected tuning was repeated after non-repeatable CUDA sparse
rounding was detected; `revision/DEVIATIONS.md` records this and the retained failed attempts.

- Collect exhaustive arrays: `python3 revision/exhaustive.py --collect --out <dir>`.
- Recompute interventions: `python3 revision/exhaustive.py --index 0 --device cuda:0` (indices 0-7;
  checkpoint 5 uses `--part-shard 0/4` through `3/4`). Processes are bounded to 120 seconds and resume
  completed relation blocks. To recompute rather than retain existing blocks, use a disposable copy
  without `revision/exhaustive/`.
- Collect native results: `python3 revision/native.py --collect --out <dir>`.
- Check packaged native weights: `python3 revision/verify_native.py --index 0 --device cuda:0`
  (indices 0-59; add `--gradients` to repeat all 100 query reaches).
- Repeat the complete native tuning and evaluation: `python3 revision/run_native.py`, in a disposable copy
  without `revision/native/`. `REVISION_GPUS` controls the GPU worker count (default four). Training
  processes resume after 120 seconds; tuning must finish before test evaluation.

## Expanded study reproduction
- Complete expansion assets: `python3 collective/stage2/report.py --paper paper`. It verifies complete
  raw coverage and hashes, re-enumerates known laws, reselects native hyperparameters from validation-only
  records, and regenerates tables/figures without rewriting any experiment record.
- Native checkpoint replay: `python3 collective/stage2/verify_native.py --device cuda:0`.
  It checks all 60 random-input native weights and repeats all 100 gradients for four representative
  deep checkpoints; the recorded all-checkpoint audit is already bundled. Run it in a disposable copy.
- Seeded intervention and full-control replay: `python3 collective/stage2/verify.py --shard 0 --device cuda:0`
  (shards 0-3, about a minute per process). It writes only independent verification records.
- Fresh measurements use disposable copies after removing the relevant measurement directories;
  frozen worker sources, seeds, source closures, protocols and completion hashes are retained.
  New training checkpoints use the same content-addressed store. Do not fit new policies after viewing
  held-out results. The recorded policy commits preceded both test directories.
- `collective/stage2/results/integrity.json` records the original history/coverage/cohort audit. Full git
  history is withheld for anonymity, so replay of its commit-object checks requires the final release.
  Metadata identifiers in generated analyses are normalised through SANITIZATION; numerical arrays and
  model hashes are unchanged. This preserves byte-for-byte regeneration after anonymous metadata edits.

Timings use Tesla V100-SXM2 cards with 16 or 32 GB, CUDA 12.8 and PyTorch 2.8.
The environment GPU-name field names the default visible device; assigned CUDA indices are recorded
in execution ledgers. Absolute latency and per-checkpoint speed ratios are reported separately.

## Anonymisation
Machine paths and the host name were rewritten for review (`<repo>`, `$SRANGE_STORE`, `<venv>`, `<paper>`,
host `anonymous`). Run records carry their own SHA-256, so each rewritten record was re-hashed;
`SANITIZATION.json` maps its original `result_sha256` (the value quoted elsewhere, e.g. in measurement
outputs) to the new one. Test-logit and checkpoint hashes are unchanged. Three planning documents that
describe the authors' earlier work, and one test that compares code against it, are withheld. The git
history, which fixes the time order of every protocol lock, commit and run, will be released with the final
version; the lock files record the commit and UTC time of each lock.
