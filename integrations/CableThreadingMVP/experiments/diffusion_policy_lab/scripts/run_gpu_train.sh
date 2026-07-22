#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:-$ROOT/../../../../runs/cable_threading/jobs/ct_gen_20260615_102019_8f58/datasets/dataset.hdf5}"
OUT="$ROOT/outputs/gpu_run_v1"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python}"

echo "dataset: $DATASET"
echo "out: $OUT"
echo "device: cuda (resnet18)"

"$PYTHON" "$ROOT/scripts/train.py" \
  --dataset "$DATASET" \
  --out-dir "$OUT" \
  --vision-encoder resnet18 \
  --num-epochs 30 \
  --batch-size 8 \
  --image-size 128 \
  --learning-rate 1e-4 \
  --device cuda

echo "training done: $OUT/checkpoints/model_final.pt"
