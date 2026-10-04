#!/bin/bash
# Pause the confirmatory queue (no new launches; running jobs continue) if free space on / drops below 4 GB.
while kill -0 3052509 2>/dev/null; do
  free_kb=$(df --output=avail / | tail -1)
  if [ "$free_kb" -lt 4000000 ]; then
    kill -STOP 3052509
    echo "PAUSED queue 3052509 at $(date -u +%H:%M) with $((free_kb/1024)) MB free"
    exit 0
  fi
  sleep 60
done
echo "queue finished; watchdog exiting"
