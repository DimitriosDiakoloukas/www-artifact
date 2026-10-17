#!/bin/bash
# Fits, then hybrid audits of every test query; one thread per process.
cd "$(dirname "$0")/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
D=gnn/www-propagation/experiments/hybrid-auditor
LOG=$D/runs/v1/logs
mkdir -p $LOG
pids=()
for u in 0 1 2 3 4 5; do $PY -B $D/ungated.py fit --unit $u > $LOG/fit$u.log 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait $p || { echo FIT FAILED; exit 1; }; done
pids=()
for spec in 0:2 1:2 4:2 5:2 2:6 3:6; do
  u=${spec%:*}; k=${spec#*:}
  for ((j=0; j<k; j++)); do
    $PY -B $D/ungated.py audit --unit $u --shard $j --shards $k > $LOG/audit$u-$j.log 2>&1 & pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL DONE fail=$fail"
