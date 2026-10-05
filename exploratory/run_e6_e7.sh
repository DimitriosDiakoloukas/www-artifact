#!/bin/bash
# E6-E8 driver (exploratory/PLAN.md): E6 training and the E7 audit in one queue, then E6 decisive reach,
# one rerun of failed measurements on the 32 GB cards, then the E6 and E7 summaries.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/workspace/.venv/bin/python
OUTD=$HOME/workspace/papers/www2027/generated
note() { echo "$(date -u +%Y-%m-%dT%H:%MZ) $*" | tee -a exploratory/e6e7_driver.log; }
note "queue start"
$PY exploratory/run_queue_x.py exploratory/jobs_e6.txt --gpus 0,1,2,3,4,5,6,7 --logs exploratory/logs/e6 > exploratory/queue_e6.log 2>&1
note "queue done: $(grep -c '^done' exploratory/queue_e6.log) done, $(grep -c '^FAIL' exploratory/queue_e6.log) failed"
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$k $PY exploratory/e6_measure.py --shard $k/8 --device cuda:0 > exploratory/logs/e6/measure-$k.log 2>&1 &
done
wait
note "E6 measurements: $(ls exploratory/e6-random/measure | wc -l)/$(ls exploratory/e6-random/runs | wc -l); reruns of missing items on gpus 0-3"
for k in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$k $PY exploratory/e6_measure.py --shard $k/4 --device cuda:0
    CUDA_VISIBLE_DEVICES=$k $PY exploratory/e7_audit.py --shard $k/4 --device cuda:0 ) > exploratory/logs/e6/rerun-$k.log 2>&1 &
done
wait
$PY exploratory/e6_measure.py --collect --out "$OUTD" > exploratory/logs/e6/collect.log 2>&1
$PY exploratory/e7_audit.py --collect --out "$OUTD" > exploratory/logs/e6/collect_e7.log 2>&1
note "summaries written: e6 $(tail -1 exploratory/logs/e6/collect.log); e7 $(tail -1 exploratory/logs/e6/collect_e7.log)"
