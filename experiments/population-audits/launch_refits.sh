#!/usr/bin/env bash
# Fits, validation audits, selection and test audits for runs/refits-v1; safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/refits-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/refits.py "$@"; }
export -f run; export D OUT
for u in 0 1 4 5 2 3; do for l in 0.0 0.01 0.03 0.1 0.3; do
  run fit $OUT --unit $u --lam $l --workers 22 > $OUT/logs/fit-$u-$l.log 2>&1 || echo "FAILED fit $u $l"
done; done
for split in validation; do for u in 0 1 2 3 4 5; do for k in $(seq 0 21); do echo "$u $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run audit $OUT --unit $0 --split validation --shard $1 --shards 22 > $OUT/logs/validation-$0-$1.log 2>&1 || echo "FAILED validation $0 $1"'; done
for u in 0 1 2 3 4 5; do run select $OUT --unit $u > $OUT/logs/select-$u.log 2>&1 || echo "FAILED select $u"; done
for u in 0 1 2 3 4 5; do for k in $(seq 0 21); do echo "$u $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run audit $OUT --unit $0 --split test --shard $1 --shards 22 > $OUT/logs/test-$0-$1.log 2>&1 || echo "FAILED test $0 $1"'
echo ALL REFITS FINISHED
