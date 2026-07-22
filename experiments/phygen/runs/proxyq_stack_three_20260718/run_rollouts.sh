#!/usr/bin/env bash
set -euo pipefail

PY=/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python
ROOT=/home/ubuntu/project/eai-idev2.1/third_party/mimicgen
EXP=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/proxyq_stack_three_20260718
DATA="$EXP/data"
SCRIPT="$EXP/scripts/stack_three_failed_conditioned_mimicgen_repair.py"

export MUJOCO_GL=egl
export PYTHONPATH="$EXP/scripts:$ROOT:${PYTHONPATH:-}"

for METHOD in baseline proxyq; do
    OUT="$EXP/outputs/${METHOD}_rollout"
    LOG="$EXP/logs/${METHOD}_rollout.log"
    mkdir -p "$OUT"
    echo "[$(date '+%F %T')] starting $METHOD" | tee -a "$EXP/logs/rollout_queue.log"
    "$PY" "$SCRIPT" \
        --config "$DATA/mg_config.json" \
        --source-hdf5 "$DATA/source_prepared.hdf5" \
        --success-hdf5 "$DATA/demo.hdf5" \
        --failed-hdf5 "$DATA/demo_failed.hdf5" \
        --candidate-plan "$EXP/outputs/$METHOD/phygen_main_candidate_plan.jsonl" \
        --output-dir "$OUT" \
        --max-failed-demos 100000 \
        --seed 9917 2>&1 | tee "$LOG"
    echo "[$(date '+%F %T')] completed $METHOD" | tee -a "$EXP/logs/rollout_queue.log"
done

"$PY" - <<'PY' > "$EXP/outputs/comparison.json"
import json
from pathlib import Path

root = Path('/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/proxyq_stack_three_20260718')
out = {'raw_success': 39, 'raw_failed': 61, 'raw_total': 100, 'methods': {}}
for method in ['baseline', 'proxyq']:
    summary = json.loads((root / 'outputs' / f'{method}_rollout' / 'summary.json').read_text())
    per_demo = summary.get('per_demo', {})
    hit_ranks = []
    repaired = []
    for demo, values in per_demo.items():
        if values.get('num_success', 0) > 0:
            repaired.append(demo)
            candidates = values.get('candidates', [])
            rank = next((i + 1 for i, row in enumerate(candidates) if row.get('success')), None)
            if rank is not None:
                hit_ranks.append(rank)
    out['methods'][method] = {
        'repaired': int(summary.get('repaired_count_in_evaluated', len(repaired))),
        'projected_total_success': int(summary.get('projected_total_success_if_unique_repairs', 39 + len(repaired))),
        'projected_total_success_rate': float(summary.get('projected_total_success_rate_if_unique_repairs', (39 + len(repaired)) / 100)),
        'candidate_success_rate': summary.get('candidate_success_rate'),
        'num_problematic': summary.get('num_problematic'),
        'elapsed_sec': summary.get('elapsed_sec'),
        'hit_ranks': hit_ranks,
        'repaired_demos': repaired,
    }
print(json.dumps(out, indent=2))
PY

echo "[$(date '+%F %T')] all rollouts complete" | tee -a "$EXP/logs/rollout_queue.log"
