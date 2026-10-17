#!/usr/bin/env bash
# Validation audits, full-validation selection, then test audits (runs/fullval-v1); safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/fullval-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/fullval.py "$@"; }
export -f run; export D OUT
for u in 2 3 4 5 0 1; do for k in $(seq 0 21); do echo "$u $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run audit $OUT --unit $0 --split validation --shard $1 --shards 22 > $OUT/logs/val-$0-$1.log 2>&1 || echo "FAILED val $0 $1"'
seq 0 5 | xargs -P 6 -I{} bash -c 'run select $OUT --unit {} > $OUT/logs/select-{}.log 2>&1 || echo "FAILED select {}"'
for u in 2 3 4 5 0 1; do for k in $(seq 0 21); do echo "$u $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run audit $OUT --unit $0 --split test --shard $1 --shards 22 > $OUT/logs/test-$0-$1.log 2>&1 || echo "FAILED test $0 $1"'
echo ALL FULLVAL FINISHED
