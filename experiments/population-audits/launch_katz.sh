#!/usr/bin/env bash
# Fit and audit the truncated signed Katz score (runs/katz-v1); safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/katz-v1; mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/katz_audit.py "$@"; }
export -f run; export D OUT
seq 0 5 | xargs -P 6 -I{} bash -c 'run fit $OUT --unit {} > $OUT/logs/fit-{}.log 2>&1 || echo "FAILED fit {}"'
for u in 2 3 4 5 0 1; do for k in $(seq 0 21); do echo "$u $k"; done; done |
  xargs -P 22 -L 1 bash -c 'run audit $OUT --unit $0 --shard $1 --shards 22 > $OUT/logs/audit-$0-$1.log 2>&1 || echo "FAILED audit $0 $1"'
echo ALL KATZ FINISHED
