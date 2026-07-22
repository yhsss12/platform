#!/usr/bin/env bash
set -euo pipefail
PY=/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python
EXP=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/stack_three_bc
GEN=/home/ubuntu/project/eai-idev2.1/mimicgen_generated/stack_three_bc_keepall_20260715
DATA=$GEN/stack_three_d0_seed16201_demo133_keepall/demo_src_stack_three_task_D0
PIDF=$GEN/logs/stack_three_d0_seed16201_demo133_keepall.pid
RUN=$EXP/auto_bc_keepall_seed16201_success133
LOGDIR=$RUN/logs
RESULTS=$RUN/bc_rnn_results
CFGDIR=$RUN/bc_rnn_configs
mkdir -p "$LOGDIR" "$RESULTS" "$CFGDIR"
export MUJOCO_GL=egl
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/pipeline.log"; }
count_demos(){ "$PY" - "$1" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1],'r') as f: print(len(f['data']))
PY
}
if [ -f "$PIDF" ]; then
  PID=$(cat "$PIDF")
  while kill -0 "$PID" 2>/dev/null; do log "generation demo133 running pid=$PID"; sleep 120; done
fi
log "generation demo133 finished"
for f in "$DATA/demo.hdf5" "$DATA/demo_failed.hdf5" "$DATA/mg_config.json"; do [ -f "$f" ] || { log "missing $f"; exit 2; }; done
S=$(count_demos "$DATA/demo.hdf5"); F=$(count_demos "$DATA/demo_failed.hdf5")
log "generated demo133 data: success=$S failed=$F total_trials=$((S+F))"
"$PY" - <<PY
import json, os, robomimic
from pathlib import Path
base=json.load(open(os.path.join(robomimic.__path__[0], 'exps/templates/bc.json')))
c=json.loads(json.dumps(base))
c['experiment']['name']='stack_three_bc_rnn_success133_keepall_2000ep'
c['experiment']['validate']=False
c['experiment']['save']['every_n_epochs']=500
c['experiment']['save']['on_best_rollout_success_rate']=False
c['experiment']['rollout']['enabled']=False
c['train']['data']='$DATA/demo.hdf5'
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
Path('$CFGDIR/success133_bc_rnn_2000ep.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
PY
TRAIN=$("$PY" - <<'PY'
import robomimic, os
print(os.path.join(robomimic.__path__[0], 'scripts/train.py'))
PY
)
log "starting success133 training"
"$PY" "$TRAIN" --config "$CFGDIR/success133_bc_rnn_2000ep.json" > "$LOGDIR/success133_train_2000ep.log" 2>&1
log "success133 training done"
ROLLOUT=$("$PY" - <<'PY'
import robomimic, os
print(os.path.join(robomimic.__path__[0], 'scripts/run_trained_agent.py'))
PY
)
CKPT=$(find "$RESULTS/stack_three_bc_rnn_success133_keepall_2000ep" -type f -name 'model_epoch_2000.pth' | sort | tail -1)
log "evaluating success133 ckpt=$CKPT"
"$PY" "$ROLLOUT" --agent "$CKPT" --n_rollouts 50 --horizon 400 --env StackThree_D0 --seed 781 > "$LOGDIR/success133_eval_seed781_n50.log" 2>&1
grep -A8 'Average Rollout Stats' "$LOGDIR/success133_eval_seed781_n50.log" | tee -a "$LOGDIR/pipeline.log"
log "pipeline complete"
