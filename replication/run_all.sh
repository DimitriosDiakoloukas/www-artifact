#!/bin/bash
# Replication driver (PROTOCOL.md Sections 3-5): the queue, one rerun of failed runs, the chain
# measurements, one rerun of failed measurements on the 32 GB cards, then the analysis.
# Every step skips work whose output already exists, so the driver can be restarted.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/workspace/.venv/bin/python
LOG=replication/logs
mkdir -p $LOG
note() { echo "$(date -u +%Y-%m-%dT%H:%MZ) $*" | tee -a replication/driver.log; }

watchdog() {                       # pause (not kill) a process if / drops below 4 GB free
  while kill -0 "$1" 2>/dev/null; do
    if [ "$(df --output=avail / | tail -1)" -lt 4000000 ]; then
      kill -STOP "$1"; note "PAUSED $1: under 4 GB free"; return
    fi
    sleep 60
  done
}

note "queue start"
$PY replication/run_queue_r.py replication/jobs.txt --gpus 0,1,2,3,4,5,6,7 --logs $LOG > replication/queue.log 2>&1 &
Q=$!; watchdog $Q & wait $Q
note "queue done: $(grep -c '^done' replication/queue.log) done, $(grep -c '^FAIL' replication/queue.log) failed"

# Section 4: a failed planted run is rerun once on a 32 GB card with --chunk 1 (identical results).
grep 'run_chain_r.py' replication/jobs.txt | while read -r cmd; do
  id=$(echo "$cmd" | sed -E 's/.*--variant (planted-[a-z_]+).*--r ([0-9]+) --arch ([A-Z]+).*--seed ([0-9]+).*/relay-r\2-\3-\1-T32-s\4/')
  [ -f "replication/planted/runs/$id.json" ] || echo "$cmd" | sed -E 's/ --chunk 1//; s/$/ --chunk 1/'
done > replication/jobs_rerun.txt
if [ -s replication/jobs_rerun.txt ]; then
  note "rerunning $(wc -l < replication/jobs_rerun.txt) planted runs on gpus 0-3"
  $PY replication/run_queue_r.py replication/jobs_rerun.txt --gpus 0,1,2,3 --logs $LOG/rerun > replication/queue_rerun.log 2>&1
fi

note "chain measurements"
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$k $PY replication/measure.py chains --shard $k/8 --device cuda:0 > $LOG/measure-chains-${k}of8.log 2>&1 &
done
wait

# Section 4: a measurement without an output is rerun once on a 32 GB card.
note "measurement reruns on gpus 0-3 (only items without an output)"
for k in 0 1 2 3; do
  ( for mode in native sidnet chains; do
      CUDA_VISIBLE_DEVICES=$k $PY replication/measure.py $mode --shard $k/4 --device cuda:0
    done ) > $LOG/measure-rerun-${k}of4.log 2>&1 &
done
wait

note "counts: planted $(ls replication/planted/runs | wc -l)/120, chains $(ls replication/measure/chains | wc -l)/120," \
     "native $(ls replication/measure/native | wc -l)/320, sidnet $(ls replication/measure/sidnet | wc -l)/30"
$PY replication/analyze.py --out replication/analysis > $LOG/analyze.log 2>&1
note "analysis written: replication/analysis/replication.json"
