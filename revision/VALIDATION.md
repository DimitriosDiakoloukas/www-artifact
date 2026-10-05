# Verification of the reviewer-requested revision

The new studies are exploratory and were specified in revision/PROTOCOL.md before computation.
The protocol in commit 2dead31 (2026-10-05 19:57 UTC) is byte-identical to the final file; the first
retained native tuning record is dated 20:05 UTC. Execution changes and excluded failed attempts
are recorded in DEVIATIONS.md, without changing queries, thresholds, the tuning grid or decisions.

- Eight exhaustive audits are complete. Collecting requires non-overlapping coverage of every
  training relation and verifies all saved array hashes and intact checkpoint logits.
- All 144 final tuning records, 60 evaluation records, 72 superseded calibration records and
  eight exhaustive completion records verify against their self-hashes (284 records).
- Every new evaluation record uses the setting selected on validation for its model/network/depth.
  The retained SGCN settings are identical to the archived selection; no tuning record scores test AUC.
- All 60 native checkpoints were independently loaded through revision/verify_native.py and
  reproduced their test logits bitwise. Indices 10, 25, 40 and 55 (T=32, first evaluation seed,
  both models and networks) also reproduced all 100 stored query reaches and post-gradient logits.
- All 17 tests in test_models, test_range, test_synthetic and test_revision pass. The added checks
  compare SGCN against the unchanged package, SIDNET against an independently written dense
  recurrence, relaxed derivatives against numerical derivatives, and repeated CUDA SIDNET forwards
  before and after a gradient calculation. Undefined zero profiles remain undefined.
- reporting/check_result_map.py verifies all 34 paper outputs. reporting/regen_check.sh regenerates
  all 34 from an empty temporary directory and reproduces every byte, with zero problems.

Reproduction commands (from this repository, with the pinned requirements installed):

    python3 tests/run.py test_models test_range test_synthetic test_revision
    python3 revision/exhaustive.py --collect --out <paper>/generated
    python3 revision/native.py --collect --out <paper>/generated
    python3 revision/verify_native.py --index 10 --device cuda:0 --gradients
    python3 reporting/check_result_map.py --paper <paper>
    bash reporting/regen_check.sh <paper>

The paper reports observed outcomes and disagreements. These software and reproduction checks
are not acceptance criteria for the scientific hypotheses or a guarantee of conference acceptance.
