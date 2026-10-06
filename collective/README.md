# Collective dependence pilot for the WWW study
This pilot implements the first stage of the review-motivated research plan. It asks whether predictions
with short individual sign-sensitivity reach remain stable when distant training signs change jointly.
The current submission manuscript and anonymous artifact remain the validated baseline during this work.

Start with `results/report/REPORT.md` and the two figures there. `PILOT_PROTOCOL.md` and `LOCK.json` freeze
choices before measurements; `METHODS.md` states the assumptions and proofs. `IMPLEMENTATION.md` records
additional implementation details before execution. `ROADMAP.md` and `NEXT_STAGE.md` identify later stages, and
`LITERATURE.md` states the novelty boundary. `RESULT_MAP.md` maps every result to its inputs and generator.

Run from the campaign root using the WWW Python environment and cached SNAP/checkpoint store:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 tests/run.py test_collective test_collective_reporting
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 collective/controls.py --out /tmp/www-known-controls.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 collective/measure.py --index 0 --device cuda:0 --seconds 115
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 collective/train_control.py --arch SGCN --device cuda:4 --seconds 115
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 collective/collect.py --out /tmp/www-collective-report-check
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 collective/verify.py --device cuda:0 --out /tmp/www-collective-verification.json
```

Repeat bounded measurements until complete. Network indices 0–3 and both architectures are required.
Complete existing runs are verified and skipped. Distinct checkpoint indices can use distinct GPUs;
never start two writers for the same checkpoint directory. Training resumes preserve optimizer and
CPU/CUDA RNG. Complete records are required before report collection; incomplete results are rejected.

Raw draws, profiles, selected learned-control checkpoints and hashes are retained. Model scores are not
assumed calibrated. Real-network interventions are stress distributions, not estimated conditional laws
or causal counterfactuals. Sampling bounds do not validate those distributions. No experiment establishes
that graph cropping is safe: the later practical study must test its actual approximation.

For training from scratch, use a separate checkout or a copy under /tmp without the `runs/`, `results/`
and training result directories. Keep the frozen protocol, lock and code unchanged. This preserves the
original results while allowing the same commands to create fresh outputs. `training/CHECKPOINTS.json`
bundles the two selected control checkpoints; original network checkpoints remain in the WWW object store.
