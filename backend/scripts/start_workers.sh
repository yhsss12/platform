#!/usr/bin/env bash
set -euo pipefail

# 稳定入口：每个队列独立 worker 进程，互不抢占。
# 运行前请先确保 Redis 可用。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

QUEUE_ANNOTATION="${QUEUE_ANNOTATION:-gpu_queue}"
QUEUE_CONVERSION="${QUEUE_CONVERSION:-cpu_queue}"
QUEUE_COLLECT="${QUEUE_COLLECT:-collect_queue}"
QUEUE_BATCH="${QUEUE_BATCH:-io_queue}"

echo "[workers] backend root: ${ROOT_DIR}"
echo "[workers] annotation queue: ${QUEUE_ANNOTATION}"
echo "[workers] conversion queue: ${QUEUE_CONVERSION}"
echo "[workers] collect queue: ${QUEUE_COLLECT}"
echo "[workers] batch/export queue: ${QUEUE_BATCH}"

run_worker() {
  local queue="$1"
  nohup "${PYTHON_BIN}" worker/start_worker.py --queues "${queue}" > "worker-${queue}.log" 2>&1 &
  echo "[workers] started queue=${queue} pid=$!"
}

run_worker "${QUEUE_ANNOTATION}"
run_worker "${QUEUE_CONVERSION}"
run_worker "${QUEUE_COLLECT}"
run_worker "${QUEUE_BATCH}"

echo "[workers] all workers started"
