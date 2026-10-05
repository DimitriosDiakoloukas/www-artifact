#!/bin/bash
# E7b driver (exploratory/PLAN.md): 8 shards, one rerun of missing items on the 32 GB cards, then the summary.
cd "$(dirname "$0")/.."
PY=$HOME/workspace/.venv/bin/python
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$k $PY exploratory/e7_audit.py --subset uniform --shard $k/8 --device cuda:0 > exploratory/logs/e7b/shard$k.log 2>&1 &
done
wait
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY exploratory/e7_audit.py --subset uniform --shard $k/4 --device cuda:0 > exploratory/logs/e7b/rerun$k.log 2>&1 &
done
wait
$PY exploratory/e7_audit.py --subset uniform --collect --out $HOME/workspace/papers/www2027/generated > exploratory/logs/e7b/collect.log 2>&1
echo done > exploratory/logs/e7b/done
