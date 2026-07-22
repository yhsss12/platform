#!/usr/bin/env bash
RUN=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/coffee_preparation_compare/auto_mg_vs_ts_seed17201
while true; do
  pids=$(pgrep -af 'robomimic/scripts/train.py --config .*/auto_mg_vs_ts_seed17201/' | awk '{print $1}' || true)
  if [ -n "$pids" ]; then
    echo "[$(date '+%F %T')] stopping BC-RNN training pids: $pids" >> "$RUN/logs/training_guard.log"
    kill $pids || true
    exit 0
  fi
  sleep 30
done
