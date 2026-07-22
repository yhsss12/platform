#!/usr/bin/env bash
RUN=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/coffee_preparation_compare/auto_mg_vs_ts_seed17201
while true; do
  pids=$(ps -eo pid,args | grep -E 'generate_dataset.py|robomimic/scripts/train.py|run_trained_agent.py' | grep -E 'auto_mg_vs_ts_seed17201|coffee_preparation_d0_seed17202_extra' | grep -v grep | awk '{print $1}' || true)
  if [ -n "$pids" ]; then
    echo "[$(date '+%F %T')] stopping post-repair pids: $pids" >> "$RUN/logs/post_repair_guard.log"
    kill $pids || true
  fi
  if [ -f "$RUN/ts_repair_export/summary.json" ] && [ -f "$RUN/shared_phygen_export/summary.json" ]; then
    sleep 60
    exit 0
  fi
  sleep 20
done
