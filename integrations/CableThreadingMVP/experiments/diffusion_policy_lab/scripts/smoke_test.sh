#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:-$ROOT/../../../../runs/cable_threading/jobs/ct_gen_20260615_102019_8f58/datasets/dataset.hdf5}"
OUT="$ROOT/outputs/smoke_run"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python}"

echo "dataset: $DATASET"
echo "out: $OUT"

rm -rf "$OUT"
mkdir -p "$OUT"

"$PYTHON" "$ROOT/scripts/inspect_dataset.py" --dataset "$DATASET"
"$PYTHON" "$ROOT/scripts/train.py" \
  --dataset "$DATASET" \
  --out-dir "$OUT" \
  --debug

"$PYTHON" "$ROOT/scripts/infer_smoke.py" \
  --checkpoint "$OUT/checkpoints/model_final.pt" \
  --dataset "$DATASET" \
  --device cpu

echo "smoke test passed"
