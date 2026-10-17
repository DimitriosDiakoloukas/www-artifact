#!/usr/bin/env bash
# Runs every shard of runs/v1; safe to rerun (completed lines are kept and skipped).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits
mkdir -p $D/runs/v1/logs
jobs=()
for unit in 2 3; do for k in $(seq 0 11); do jobs+=("$unit $k 12"); done; done
for unit in 4 5; do for k in $(seq 0 4); do jobs+=("$unit $k 5"); done; done
for unit in 0 1; do for k in $(seq 0 3); do jobs+=("$unit $k 4"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P "${WORKERS:-22}" -L 1 bash -c \
  'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B '"$D"'/full_audit.py run '"$D"'/runs/v1 --unit $0 --shard $1 --shards $2 > '"$D"'/runs/v1/logs/unit$0-shard$1.log 2>&1 || echo "FAILED unit $0 shard $1"'
echo ALL SHARDS FINISHED
