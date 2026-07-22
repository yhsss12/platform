#!/usr/bin/env bash
set -euo pipefail

PY=/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python
ROOT=/home/ubuntu/project/eai-idev2.1/third_party/mimicgen
EXP=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/coffee_preparation_compare
GENROOT=/home/ubuntu/project/eai-idev2.1/mimicgen_generated/coffee_preparation_compare_20260716
SRC_RAW=$ROOT/datasets/source/coffee_preparation.hdf5
SRC_PREP=$GENROOT/source/coffee_preparation_prepared.hdf5
BASE_CFG=$GENROOT/configs/demo_src_coffee_preparation_D0_1000trial_keepall.json
RUN=$EXP/auto_mg_vs_ts_seed17201
LOGDIR=$RUN/logs
mkdir -p "$GENROOT/source" "$GENROOT/configs" "$LOGDIR" "$RUN"

export MUJOCO_GL=egl
export PYTHONPATH="$ROOT:$EXP/scripts:${PYTHONPATH:-}"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/pipeline.log"; }
count_demos(){ "$PY" - "$1" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1],'r') as f: print(len(f['data']))
PY
}
merge_two(){ "$PY" - "$1" "$2" "$3" <<'PY'
import h5py, sys
from pathlib import Path

def sort_key(k):
    try: return int(k.split('_')[-1])
    except Exception: return 10**9
src_a, src_b, out = map(Path, sys.argv[1:])
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): out.unlink()
with h5py.File(src_a,'r') as a, h5py.File(src_b,'r') as b, h5py.File(out,'w') as o:
    for k,v in a.attrs.items(): o.attrs[k]=v
    gd=o.create_group('data')
    for k,v in a['data'].attrs.items(): gd.attrs[k]=v
    total=0; idx=0
    for src, label in [(a, 'base'), (b, 'extra')]:
        for key in sorted(src['data'].keys(), key=sort_key):
            src['data'].copy(key, gd, name=f'demo_{idx}')
            g=gd[f'demo_{idx}']
            g.attrs['merge_source']=label
            g.attrs['source_demo_key']=key
            total += int(g.attrs.get('num_samples', g['actions'].shape[0]))
            idx += 1
    gd.attrs['total']=total
print(out)
PY
}

log "disk before"
df -h / | tee -a "$LOGDIR/pipeline.log"

if [ ! -f "$SRC_PREP" ]; then
  log "preparing source dataset"
  "$PY" "$ROOT/mimicgen/scripts/prepare_src_dataset.py" \
    --dataset "$SRC_RAW" \
    --env_interface MG_CoffeePreparation \
    --env_interface_type robosuite \
    --output "$SRC_PREP" 2>&1 | tee "$LOGDIR/prepare_source.log"
fi

log "writing fixed-1000-trial config"
"$PY" - <<PY
import json
from pathlib import Path
src=Path('/tmp/core_configs/demo_src_coffee_preparation_task_D0.json')
out=Path('$BASE_CFG')
c=json.load(open(src))
c['experiment']['generation']['guarantee']=False
c['experiment']['generation']['keep_failed']=True
c['experiment']['generation']['num_trials']=1000
c['experiment']['max_num_failures']=100000
c['experiment']['render_video']=False
c['experiment']['num_demo_to_render']=0
c['experiment']['num_fail_demo_to_render']=0
out.write_text(json.dumps(c, indent=2), encoding='utf-8')
print(out)
PY

SEED=17201
BASE_FOLDER=$GENROOT/coffee_preparation_d0_seed${SEED}_trial1000_keepall
BASE_DATA=$BASE_FOLDER/demo_src_coffee_preparation_task_D0
if [ ! -f "$BASE_DATA/demo.hdf5" ] || [ ! -f "$BASE_DATA/demo_failed.hdf5" ]; then
  log "generating fixed 1000 trials seed=$SEED"
  rm -rf "$BASE_FOLDER"
  "$PY" "$ROOT/mimicgen/scripts/generate_dataset.py" \
    --config "$BASE_CFG" \
    --source "$SRC_PREP" \
    --folder "$BASE_FOLDER" \
    --seed "$SEED" \
    --auto-remove-exp 2>&1 | tee "$LOGDIR/generate_1000trial.log"
fi
C=$(count_demos "$BASE_DATA/demo.hdf5")
F=$(count_demos "$BASE_DATA/demo_failed.hdf5")
log "base generation complete: C_success=$C F_failed=$F total=$((C+F))"

SEL=$RUN/selector
if [ ! -f "$SEL/pinn_utility_boundary_union_candidate_plan.jsonl" ]; then
  log "running TS selector"
  "$PY" "$EXP/scripts/coffee_preparation_rp_residual_field_pinn_selector.py" \
    --target-failed-hdf5 "$BASE_DATA/demo_failed.hdf5" \
    --target-success-hdf5 "$BASE_DATA/demo.hdf5" \
    --checkpoint "$EXP/checkpoints/coffee_preparation_failed_conditioned_pinn.pt" \
    --output-dir "$SEL" \
    --budget 5 \
    --pool-size 128 \
    --candidate-mode safe 2>&1 | tee "$LOGDIR/selector.log"
fi

REPAIR=$RUN/ts_repair_export
if [ ! -f "$REPAIR/summary.json" ]; then
  log "running TS repair export"
  "$PY" "$EXP/scripts/coffee_preparation_failed_conditioned_mimicgen_repair_export.py" \
    --config "$BASE_DATA/mg_config.json" \
    --success-hdf5 "$BASE_DATA/demo.hdf5" \
    --failed-hdf5 "$BASE_DATA/demo_failed.hdf5" \
    --candidate-plan "$SEL/pinn_utility_boundary_union_candidate_plan.jsonl" \
    --output-dir "$REPAIR" \
    --max-failed-demos 100000 \
    --export-success-hdf5 "$REPAIR/demo_ts_repaired_success.hdf5" \
    --export-merged-hdf5 "$REPAIR/demo_ts_merged_all_success.hdf5" 2>&1 | tee "$LOGDIR/repair_export.log"
fi
R=$("$PY" - <<PY
import json
s=json.load(open('$REPAIR/summary.json'))
print(s['repaired_count_in_evaluated'])
PY
)
log "TS repair success R=$R"

EXTRA_CFG=$GENROOT/configs/demo_src_coffee_preparation_D0_extraR_success.json
"$PY" - <<PY
import json
from pathlib import Path
c=json.load(open('$BASE_CFG'))
c['experiment']['generation']['guarantee']=True
c['experiment']['generation']['keep_failed']=True
c['experiment']['generation']['num_trials']=int('$R')
c['experiment']['max_num_failures']=100000
Path('$EXTRA_CFG').write_text(json.dumps(c, indent=2), encoding='utf-8')
PY
EXTRA_SEED=17202
EXTRA_FOLDER=$GENROOT/coffee_preparation_d0_seed${EXTRA_SEED}_extra${R}_success_keepall
EXTRA_DATA=$EXTRA_FOLDER/demo_src_coffee_preparation_task_D0
if [ "$R" -gt 0 ] && [ ! -f "$EXTRA_DATA/demo.hdf5" ]; then
  log "generating MimicGen extra R=$R successes seed=$EXTRA_SEED"
  rm -rf "$EXTRA_FOLDER"
  "$PY" "$ROOT/mimicgen/scripts/generate_dataset.py" \
    --config "$EXTRA_CFG" \
    --source "$SRC_PREP" \
    --folder "$EXTRA_FOLDER" \
    --seed "$EXTRA_SEED" \
    --auto-remove-exp 2>&1 | tee "$LOGDIR/generate_mg_extraR.log"
fi

DATASETS=$RUN/datasets
mkdir -p "$DATASETS"
MG_MERGED=$DATASETS/demo_mimicgen_C_plus_R.hdf5
TS_MERGED=$DATASETS/demo_tsphygen_C_plus_R.hdf5
if [ ! -f "$MG_MERGED" ]; then
  log "merging MimicGen C+R dataset"
  merge_two "$BASE_DATA/demo.hdf5" "$EXTRA_DATA/demo.hdf5" "$MG_MERGED" | tee -a "$LOGDIR/pipeline.log"
fi
if [ ! -f "$TS_MERGED" ]; then
  log "merging TS-PhyGen C+R dataset"
  merge_two "$BASE_DATA/demo.hdf5" "$REPAIR/demo_ts_repaired_success.hdf5" "$TS_MERGED" | tee -a "$LOGDIR/pipeline.log"
fi
MGN=$(count_demos "$MG_MERGED")
TSN=$(count_demos "$TS_MERGED")
log "training datasets ready: MimicGen=$MGN TSPhyGen=$TSN"

CFGDIR=$RUN/bc_rnn_configs
RESULTS=$RUN/bc_rnn_results
mkdir -p "$CFGDIR" "$RESULTS"
log "writing BC-RNN configs"
"$PY" - <<PY
import json, os, robomimic
from pathlib import Path
base=json.load(open(os.path.join(robomimic.__path__[0], 'exps/templates/bc.json')))
items={'mimicgen':'$MG_MERGED', 'tsphygen':'$TS_MERGED'}
for name,data in items.items():
    c=json.loads(json.dumps(base))
    c['experiment']['name']=f'coffee_prep_bc_rnn_{name}_CplusR_2000ep'
    c['experiment']['validate']=False
    c['experiment']['save']['every_n_epochs']=500
    c['experiment']['save']['on_best_rollout_success_rate']=False
    c['experiment']['rollout']['enabled']=False
    c['experiment']['rollout']['horizon']=800
    c['train']['data']=data
    c['train']['output_dir']='$RESULTS'
    c['train']['num_epochs']=2000
    c['train']['batch_size']=32
    c['train']['seq_length']=10
    c['train']['dataset_keys']=['actions']
    c['train']['hdf5_cache_mode']='low_dim'
    c['train']['cuda']=True
    c['algo']['optim_params']['policy']['learning_rate']['initial']=0.001
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
for name in mimicgen tsphygen; do
  log "training $name"
  "$PY" "$TRAIN" --config "$CFGDIR/${name}_bc_rnn_2000ep.json" > "$LOGDIR/${name}_train_2000ep.log" 2>&1
  log "training $name done"
  # delete intermediate checkpoints except final to save disk
  find "$RESULTS/coffee_prep_bc_rnn_${name}_CplusR_2000ep" -type f -name 'model_epoch_*.pth' ! -name 'model_epoch_2000.pth' -delete || true
done
ROLLOUT=$("$PY" - <<'PY'
import robomimic, os
print(os.path.join(robomimic.__path__[0], 'scripts/run_trained_agent.py'))
PY
)
for name in mimicgen tsphygen; do
  CKPT=$(find "$RESULTS/coffee_prep_bc_rnn_${name}_CplusR_2000ep" -type f -name 'model_epoch_2000.pth' | sort | tail -1)
  log "evaluating $name ckpt=$CKPT"
  "$PY" "$ROLLOUT" --agent "$CKPT" --n_rollouts 50 --horizon 800 --env CoffeePreparation_D0 --seed 781 > "$LOGDIR/${name}_eval_seed781_n50.log" 2>&1
  grep -A8 'Average Rollout Stats' "$LOGDIR/${name}_eval_seed781_n50.log" | tee -a "$LOGDIR/pipeline.log"
done
log "pipeline complete"
df -h / | tee -a "$LOGDIR/pipeline.log"
