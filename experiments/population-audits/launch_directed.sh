#!/usr/bin/env bash
# Prepare, fit, validate, select and test the matched directed heads (runs/directed-v1); safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/directed-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/directed.py "$@"; }
export -f run; export D OUT
seq 0 5 | xargs -P 6 -I{} bash -c 'run prepare $OUT --unit {} > $OUT/logs/prepare-{}.log 2>&1 || echo "FAILED prepare {}"'
for u in 0 1 2 3 4 5; do for l in 0.0 0.01 0.03 0.1 0.3; do echo "$u $l"; done; done |
  xargs -P 10 -L 1 bash -c 'run fit $OUT --unit $0 --lam $1 > $OUT/logs/fit-$0-$1.log 2>&1 || echo "FAILED fit $0 $1"'
for u in 0 1 2 3 4 5; do for k in $(seq 0 9); do echo "$u $k"; done; done |
  xargs -P 12 -L 1 bash -c 'run audit $OUT --unit $0 --split validation --shard $1 --shards 10 > $OUT/logs/validation-$0-$1.log 2>&1 || echo "FAILED validation $0 $1"'
for u in 0 1 2 3 4 5; do run select $OUT --unit $u > $OUT/logs/select-$u.log 2>&1 || echo "FAILED select $u"; done
for u in 0 1 2 3 4 5; do for k in $(seq 0 9); do echo "$u $k"; done; done |
  xargs -P 12 -L 1 bash -c 'run audit $OUT --unit $0 --split test --shard $1 --shards 10 > $OUT/logs/test-$0-$1.log 2>&1 || echo "FAILED test $0 $1"'
echo ALL DIRECTED FINISHED
