#!/bin/bash
set -u
cd "$(dirname "$0")/.."
mkdir -p revision/logs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${PYTHON:-python3}
worker() {
 local worker_id=$1 gpu=$2 index status
 for index in "$worker_id" "$((worker_id+4))"; do
  while true; do
   CUDA_VISIBLE_DEVICES=$gpu "$PY" -u revision/exhaustive.py --index "$index" --seconds 120 >> "revision/logs/exhaustive-$index.log" 2>&1
   status=$?
   [ "$status" = 0 ] && break
   [ "$status" = 3 ] || return "$status"
  done
 done
}
pids=()
for worker_id in 0 1 2 3; do worker "$worker_id" "$((worker_id+4))" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"

