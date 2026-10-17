#!/usr/bin/env bash
# No-self head selection and test audits of the three control heads (runs/controls-v1); safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/controls-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/controls.py "$@"; }
export -f run; export D OUT
seq 0 5 | xargs -P 6 -I{} bash -c 'run noself $OUT --unit {} > $OUT/logs/noself-{}.log 2>&1 || echo "FAILED noself {}"'
for u in 2 3 4 5 0 1; do for k in $(seq 0 9); do echo "$u $k"; done; done |
  xargs -P 8 -L 1 bash -c 'run test $OUT --unit $0 --shard $1 --shards 10 > $OUT/logs/test-$0-$1.log 2>&1 || echo "FAILED test $0 $1"'
echo ALL CONTROLS FINISHED
