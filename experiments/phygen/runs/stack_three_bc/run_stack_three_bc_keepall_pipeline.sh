#!/usr/bin/env bash
set -euo pipefail

PY=/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python
ROOT=/home/ubuntu/project/eai-idev2.1/third_party/mimicgen
EXP=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/stack_three_bc
GENROOT=/home/ubuntu/project/eai-idev2.1/mimicgen_generated/stack_three_bc_keepall_20260715
DATA=$GENROOT/stack_three_d0_seed16201_demo100_keepall/demo_src_stack_three_task_D0
GEN_PID_FILE=$GENROOT/logs/stack_three_d0_seed16201_keepall.pid
RUN=$EXP/auto_bc_keepall_seed16201
LOGDIR=$RUN/logs
mkdir -p "$LOGDIR" "$RUN"

export MUJOCO_GL=egl
export PYTHONPATH="$ROOT:$EXP/scripts:${PYTHONPATH:-}"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/pipeline.log"; }
count_demos(){ "$PY" - "$1" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    print(len(f['data']))
PY
}

log "waiting for generation pid"
if [ -f "$GEN_PID_FILE" ]; then
  GPID=$(cat "$GEN_PID_FILE")
  while kill -0 "$GPID" 2>/dev/null; do
    log "generation still running pid=$GPID"
    sleep 120
  done
fi
log "generation process finished"

for f in "$DATA/demo.hdf5" "$DATA/demo_failed.hdf5" "$DATA/mg_config.json"; do
  if [ ! -f "$f" ]; then log "missing $f"; exit 2; fi
done
S=$(count_demos "$DATA/demo.hdf5")
F=$(count_demos "$DATA/demo_failed.hdf5")
log "generated data: success=$S failed=$F total_trials=$((S+F))"

SEL=$RUN/selector
if [ ! -f "$SEL/pinn_utility_boundary_union_candidate_plan.jsonl" ]; then
  log "running selector"
  "$PY" "$EXP/scripts/stack_three_rp_residual_field_pinn_selector.py" \
    --target-failed-hdf5 "$DATA/demo_failed.hdf5" \
    --target-success-hdf5 "$DATA/demo.hdf5" \
    --checkpoint "$EXP/checkpoints/stack_three_failed_conditioned_pinn.pt" \
    --output-dir "$SEL" \
    --budget 5 \
    --pool-size 128 \
    --candidate-mode safe 2>&1 | tee "$LOGDIR/selector.log"
fi

REPAIR=$RUN/repair_export
if [ ! -f "$REPAIR/summary.json" ]; then
  log "running repair export"
  "$PY" "$EXP/scripts/stack_three_failed_conditioned_mimicgen_repair_export.py" \
    --config "$DATA/mg_config.json" \
    --success-hdf5 "$DATA/demo.hdf5" \
    --failed-hdf5 "$DATA/demo_failed.hdf5" \
    --candidate-plan "$SEL/pinn_utility_boundary_union_candidate_plan.jsonl" \
    --output-dir "$REPAIR" \
    --max-failed-demos 100000 \
    --export-success-hdf5 "$REPAIR/demo_repaired_success.hdf5" \
    --export-merged-hdf5 "$REPAIR/demo_ours_repaired_merged.hdf5" 2>&1 | tee "$LOGDIR/repair_export.log"
fi

log "repair summary"
"$PY" - <<PY | tee "$LOGDIR/repair_summary.txt"
import json
s=json.load(open('$REPAIR/summary.json'))
for k in ['raw_success','raw_failed','raw_total','raw_success_rate','repaired_count_in_evaluated','projected_total_success_if_unique_repairs','projected_total_success_rate_if_unique_repairs','num_candidate_success','candidate_success_rate','repaired_success_hdf5','merged_hdf5']:
    print(f'{k}: {s.get(k)}')
PY

CFGDIR=$RUN/bc_rnn_configs
RESULTS=$RUN/bc_rnn_results
mkdir -p "$CFGDIR" "$RESULTS"
log "writing bc-rnn configs"
"$PY" - <<PY
import json, os, robomimic
from pathlib import Path
base=json.load(open(os.path.join(robomimic.__path__[0], 'exps/templates/bc.json')))
configs={
 'success_only': '$DATA/demo.hdf5',
 'ours_repaired': '$REPAIR/demo_ours_repaired_merged.hdf5',
}
for name,data in configs.items():
    c=json.loads(json.dumps(base))
    c['experiment']['name']=f'stack_three_bc_rnn_{name}_keepall_2000ep'
    c['experiment']['validate']=False
    c['experiment']['save']['every_n_epochs']=500
    c['experiment']['save']['on_best_rollout_success_rate']=False
    c['experiment']['rollout']['enabled']=False
    c['train']['data']=data
    c['train']['output_dir']='$RESULTS'
    c['train']['num_epochs']=2000
    c['train']['batch_size']=32
    c['train']['seq_length']=10
    c['train']['dataset_keys']=['actions']
    c['train']['hdf5_cache_mode']='low_dim'
    c['train']['cuda']=True
    c['algo']['rnn']['enabled']=True
    c['algo']['rnn']['horizon']=10
    c['algo']['rnn']['hidden_dim']=400
    c['algo']['actor_layer_dims']=[]
    c['observation']['modalities']['obs']['low_dim']=['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object']
    c['observation']['modalities']['obs']['rgb']=[]
    Path('$CFGDIR', f'{name}_bc_rnn_2000ep.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
PY

TRAIN=$("$PY" - <<'PY'
import robomimic, os
print(os.path.join(robomimic.__path__[0], 'scripts/train.py'))
PY
)

for name in success_only ours_repaired; do
  if [ ! -f "$RESULTS/${name}.done" ]; then
    log "starting training $name"
    bash -lc "export MUJOCO_GL=egl; '$PY' '$TRAIN' --config '$CFGDIR/${name}_bc_rnn_2000ep.json'" > "$LOGDIR/${name}_train_2000ep.log" 2>&1 &
    echo $! > "$RESULTS/${name}.pid"
  fi
done

for name in success_only ours_repaired; do
  PID=$(cat "$RESULTS/${name}.pid")
  while kill -0 "$PID" 2>/dev/null; do
    log "training $name still running pid=$PID"
    sleep 180
  done
  touch "$RESULTS/${name}.done"
  log "training $name done"
done

ROLLOUT=$("$PY" - <<'PY'
import robomimic, os
print(os.path.join(robomimic.__path__[0], 'scripts/run_trained_agent.py'))
PY
)
EVAL_SEED=781
N_ROLLOUTS=50
for name in success_only ours_repaired; do
  CKPT=$(find "$RESULTS/stack_three_bc_rnn_${name}_keepall_2000ep" -type f -name 'model_epoch_2000.pth' | sort | tail -1)
  log "evaluating $name ckpt=$CKPT"
  "$PY" "$ROLLOUT" --agent "$CKPT" --n_rollouts "$N_ROLLOUTS" --horizon 400 --env StackThree_D0 --seed "$EVAL_SEED" > "$LOGDIR/${name}_eval_seed${EVAL_SEED}_n${N_ROLLOUTS}.log" 2>&1
  grep -A8 'Average Rollout Stats' "$LOGDIR/${name}_eval_seed${EVAL_SEED}_n${N_ROLLOUTS}.log" | tee -a "$LOGDIR/pipeline.log"
done

log "pipeline complete"
