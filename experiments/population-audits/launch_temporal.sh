#!/usr/bin/env bash
# Fits, validation selection, test audits and baseline for runs/temporal-v1; safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/temporal-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/temporal_audit.py "$@"; }
export -f run; export D OUT
for u in 6 7; do for f in three cover; do for l in 0.0 0.01 0.03 0.1 0.3; do echo "$u $f $l"; done; done; done |
  xargs -P 20 -L 1 bash -c 'run fit $OUT --unit $0 --family $1 --lam $2 > $OUT/logs/fit-$0-$1-$2.log 2>&1 || echo "FAILED fit $0 $1 $2"'
for u in 6 7; do run select $OUT --unit $u --workers 10 > $OUT/logs/select-$u.log 2>&1 & done; wait
for u in 6 7; do for k in $(seq 0 9); do echo "$u $k"; done; done |
  xargs -P 20 -L 1 bash -c 'run test $OUT --unit $0 --shard $1 --shards 10 > $OUT/logs/test-$0-$1.log 2>&1 || echo "FAILED test $0 $1"'
for u in 6 7; do run baseline $OUT --unit $u; done
echo ALL TEMPORAL FINISHED
