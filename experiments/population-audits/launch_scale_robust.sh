#!/usr/bin/env bash
# Robust three-walk fits, validation, selection and test audits for runs/scale-robust-v1; safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/scale-robust-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/scale_robust.py "$@"; }
export -f run; export D OUT
for net in slashdot epinions; do for l in 0.0 0.01 0.03 0.1 0.3; do
  run fit $OUT --net $net --lam $l --workers 22 > $OUT/logs/fit-$net-$l.log 2>&1 || echo "FAILED fit $net $l"
done; done
for net in slashdot epinions; do for k in $(seq 0 10); do echo "$net $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run validate $OUT --net $0 --shard $1 --shards 11 > $OUT/logs/validate-$0-$1.log 2>&1 || echo "FAILED validate $0 $1"'
for net in slashdot epinions; do run select $OUT --net $net > $OUT/logs/select-$net.log 2>&1 || echo "FAILED select $net"; done
for net in slashdot epinions; do for k in $(seq 0 10); do echo "$net $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run test $OUT --net $0 --shard $1 --shards 11 > $OUT/logs/test-$0-$1.log 2>&1 || echo "FAILED test $0 $1"'
echo ALL SCALE ROBUST FINISHED
