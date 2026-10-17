#!/usr/bin/env bash
# Test features, full-test metrics and sampled audits for runs/scale-v1; safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=gnn/www-propagation/experiments/population-audits; OUT=$D/runs/scale-v1; W=${WORKERS:-22}
mkdir -p $OUT/logs
run() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B $D/scale_audit.py "$@"; }
export -f run; export D OUT
for net in epinions slashdot; do
  seq 0 $((W-1)) | xargs -P $W -I{} bash -c "run features $OUT --net $net --shard {} --shards $W > $OUT/logs/features-$net-{}.log 2>&1 || echo FAILED features $net {}"
  run metrics $OUT --net $net > $OUT/logs/metrics-$net.log 2>&1
done
for net in epinions slashdot; do for k in $(seq 0 $((W-1))); do echo "$net $k"; done; done |
  xargs -P $W -L 1 bash -c 'run audit $OUT --net $0 --shard $1 --shards '"$W"' > $OUT/logs/audit-$0-$1.log 2>&1 || echo "FAILED audit $0 $1"'
echo ALL SCALE FINISHED
