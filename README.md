# Artifact: How Far Does Learned Trust Propagate?

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
- `development/` the development studies that shaped the protocol, with their plans and outcomes.
- `analysis/`, `reporting/` the scripts that write the paper's outputs; `paper/` those outputs.
- `checkpoints/objects/` a sample of stored checkpoints, content-addressed by SHA-256 (see below).

## Reproducing
Python 3.12, `pip install -r requirements.txt` (the exact versions used). Set `SRANGE_STORE` to a writable
directory and run `python3 tools/fetch_data.py` once: it downloads the six SNAP networks there and checks the
raw and processed files against their pinned SHA-256.
- Tests: `python3 tests/run.py`.
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
All 2,460 checkpoints (7.7 GB) are kept and will be released with the final version. This artifact includes
24: the solved SIDNET chains at r = 8 planted in Web networks (exploratory E3 and the replication) and the four
architectures on Wiki-Elec at T = 32 (one of them is behind the per-shell figure). Training is deterministic;
on NVIDIA V100 cards a rerun of any job in a `jobs.txt` reproduces its checkpoint bitwise.

## Anonymisation
Machine paths and the host name were rewritten for review (`<repo>`, `$SRANGE_STORE`, `<venv>`, `<paper>`,
host `anonymous`). Run records carry their own SHA-256, so each rewritten record was re-hashed;
`SANITIZATION.json` maps its original `result_sha256` (the value quoted elsewhere, e.g. in measurement
outputs) to the new one. Test-logit and checkpoint hashes are unchanged. Three planning documents that
describe the authors' earlier work, and one test that compares code against it, are withheld. The git
history, which fixes the time order of every protocol lock, commit and run, will be released with the final
version; the lock files record the commit and UTC time of each lock.
