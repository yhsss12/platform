#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/nvidia:/home/ubuntu/.mujoco/mujoco210/bin"
PY=/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python
ROOT=/home/ubuntu/project/eai-idev2.1/third_party/mimicgen
EXP=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/coffee_preparation_compare
RUN=$EXP/auto_mg_vs_ts_seed17201
GENROOT=/home/ubuntu/project/eai-idev2.1/mimicgen_generated/coffee_preparation_compare_20260716
DATA_DIR=$GENROOT/coffee_preparation_d0_seed17201_trial1000_keepall/demo_src_coffee_preparation_task_D0
SUCCESS=$DATA_DIR/demo.hdf5
FAILED=$DATA_DIR/demo_failed.hdf5
CONFIG=$DATA_DIR/mg_config.json
LOGDIR=$RUN/logs
mkdir -p "$LOGDIR" "$RUN/datasets" "$RUN/bc_rnn_configs"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/pipeline.log"; }
count_demos(){ $PY - "$1" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1],'r') as f: print(len(f['data'].keys()))
PY
}
C=$(count_demos "$SUCCESS")
F=$(count_demos "$FAILED")
log "resume from selector: C_success=$C F_failed=$F total=$((C+F))"
log "running TS selector"
PYTHONPATH="$EXP/scripts:$ROOT:${PYTHONPATH:-}" $PY $EXP/scripts/coffee_preparation_rp_residual_field_pinn_selector.py \
  --target-failed-hdf5 "$FAILED" \
  --target-success-hdf5 "$SUCCESS" \
  --checkpoint "$EXP/checkpoints/coffee_preparation_failed_conditioned_pinn.pt" \
  --output-dir "$RUN/selector" \
  --budget 5 \
  --pool-size 128 \
  --candidate-mode safe 2>&1 | tee "$LOGDIR/selector.log"
log "running TS repair export"
PYTHONPATH="$EXP/scripts:$ROOT:${PYTHONPATH:-}" $PY $EXP/scripts/coffee_preparation_failed_conditioned_mimicgen_repair_export.py \
  --config "$CONFIG" \
  --success-hdf5 "$SUCCESS" \
  --failed-hdf5 "$FAILED" \
  --candidate-plan "$RUN/selector/pinn_utility_boundary_union_candidate_plan.jsonl" \
  --output-dir "$RUN/ts_repair_export" \
  --export-success-hdf5 "$RUN/ts_repair_export/demo_ts_repaired_success.hdf5" \
  --export-merged-hdf5 "$RUN/ts_repair_export/demo_ts_merged_all_success.hdf5" \
  --max-failed-demos 100000 2>&1 | tee "$LOGDIR/ts_repair_export.log"
R=$($PY - <<PY
import json
s=json.load(open('$RUN/ts_repair_export/summary.json'))
print(s.get('repaired_count_in_evaluated',0))
PY
)
log "TS repair complete: R=$R"
log "STOP_AFTER_TWO_REPAIR_LINES: TS repair done; waiting for user before MimicGen extra generation / dataset construction / BC-RNN training"
exit 0
EXTRA_CONFIG=$GENROOT/configs/demo_src_coffee_preparation_D0_extra_success_${R}.json
$PY - <<PY
import json
src='$GENROOT/configs/demo_src_coffee_preparation_D0_1000trial_keepall.json'
out='$EXTRA_CONFIG'
c=json.load(open(src))
c['experiment']['name']='demo_src_coffee_preparation_D0_extra_success_${R}'
c['experiment']['generation']['guarantee']=True
c['experiment']['generation']['keep_failed']=True
c['experiment']['generation']['num_trials']=int('$R')
c['experiment']['generation']['max_num_failures']=100000
json.dump(c, open(out,'w'), indent=4)
PY
EXTRA_FOLDER=$GENROOT/coffee_preparation_d0_seed17202_extra${R}_success_keepall
log "generating extra MimicGen successes R=$R seed=17202"
cd "$ROOT"
$PY mimicgen/scripts/generate_dataset.py \
  --config "$EXTRA_CONFIG" \
  --source "$GENROOT/source/coffee_preparation_prepared.hdf5" \
  --folder "$EXTRA_FOLDER" \
  --seed 17202 \
  --auto-remove-exp 2>&1 | tee "$LOGDIR/generate_extra_mimicgen.log"
EXTRA_DIR=$(find "$EXTRA_FOLDER" -maxdepth 1 -type d -name '*task_D0' | head -1)
EXTRA_SUCCESS=$EXTRA_DIR/demo.hdf5
log "merging datasets"
MIMICGEN_DATA=$RUN/datasets/demo_mimicgen_C_plus_R.hdf5
TSPHYGEN_DATA=$RUN/datasets/demo_tsphygen_C_plus_R.hdf5
$PY - "$MIMICGEN_DATA" "$SUCCESS" "$EXTRA_SUCCESS" <<'PY'
import h5py, sys, shutil, tempfile, os
out, a, b = sys.argv[1:4]
shutil.copy2(a, out)
with h5py.File(out,'a') as fo, h5py.File(b,'r') as fb:
    data=fo['data']; idx=len(data.keys())
    for k in sorted(fb['data'].keys(), key=lambda x:int(x.split('_')[-1])):
        fb.copy(f'data/{k}', data, name=f'demo_{idx}'); idx+=1
    data.attrs['total']=idx
PY
cp -f "$RUN/ts_repair_export/demo_ts_merged_all_success.hdf5" "$TSPHYGEN_DATA"
log "dataset sizes: mimicgen=$(count_demos $MIMICGEN_DATA) tsphygen=$(count_demos $TSPHYGEN_DATA)"
log "DATASETS_READY: stopping before BC-RNN training as requested"
exit 0
make_cfg(){
  local name=$1 data=$2 out=$3 expdir=$4
  $PY - "$name" "$data" "$out" "$expdir" <<'PY'
import json, sys
name,data,out,expdir=sys.argv[1:5]
c={
  'algo_name':'bc',
  'experiment':{'name':name,'validate':False,'logging':{'terminal_output_to_txt':True,'log_tb':False},'save':{'enabled':True,'every_n_epochs':2000,'epochs':[]},'epoch_every_n_steps':100,'validation_epoch_every_n_steps':10,'rollout':{'enabled':False,'n':50,'horizon':800,'rate':2000,'warmstart':0,'terminate_on_success':True},'render_video':False,'keep_all_videos':False},
  'train':{'data':data,'output_dir':expdir,'num_data_workers':0,'hdf5_cache_mode':'low_dim','hdf5_use_swmr':True,'hdf5_load_next_obs':False,'hdf5_normalize_obs':False,'seq_length':10,'pad_seq_length':True,'frame_stack':1,'pad_frame_stack':True,'batch_size':32,'num_epochs':2000,'seed':1,'cuda':True},
  'algo':{'optim_params':{'policy':{'learning_rate':{'initial':0.001,'decay_factor':0.1,'epoch_schedule':[]},'regularization':{'L2':0.0}}},'loss':{'l2_weight':1.0,'l1_weight':0.0,'cos_weight':0.0},'actor_layer_dims':[],'rnn':{'enabled':True,'horizon':10,'hidden_dim':400,'rnn_type':'LSTM','num_layers':2,'open_loop':False,'kwargs':{'bidirectional':False}},'gmm':{'enabled':False,'num_modes':5,'min_std':0.0001,'std_activation':'softplus','low_noise_eval':True},'gaussian':{'enabled':False,'fixed_std':False,'init_std':0.1,'min_std':0.0001,'std_activation':'softplus','low_noise_eval':True},'vae':{'enabled':False}},
  'observation':{'modalities':{'obs':{'low_dim':['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object'],'rgb':[],'depth':[],'scan':[]},'goal':{'low_dim':[],'rgb':[],'depth':[],'scan':[]}}},
  'meta':{'hp_base_config_file':'bc_rnn','created_by':'phygen_compare_pipeline'}
}
json.dump(c, open(out,'w'), indent=4)
PY
}
make_cfg mimicgen_C_plus_R "$MIMICGEN_DATA" "$RUN/bc_rnn_configs/mimicgen_bc_rnn_2000ep.json" "$RUN/bc_rnn_mimicgen"
make_cfg tsphygen_C_plus_R "$TSPHYGEN_DATA" "$RUN/bc_rnn_configs/tsphygen_bc_rnn_2000ep.json" "$RUN/bc_rnn_tsphygen"
cd /home/ubuntu/project/eai-idev2.1/robomimic
log "training BC-RNN MimicGen group"
$PY robomimic/scripts/train.py --config "$RUN/bc_rnn_configs/mimicgen_bc_rnn_2000ep.json" 2>&1 | tee "$LOGDIR/train_mimicgen.log"
find "$RUN/bc_rnn_mimicgen" -name 'model_epoch_*.pth' ! -name 'model_epoch_2000.pth' -delete || true
log "training BC-RNN TS-PhyGen group"
$PY robomimic/scripts/train.py --config "$RUN/bc_rnn_configs/tsphygen_bc_rnn_2000ep.json" 2>&1 | tee "$LOGDIR/train_tsphygen.log"
find "$RUN/bc_rnn_tsphygen" -name 'model_epoch_*.pth' ! -name 'model_epoch_2000.pth' -delete || true
find_ckpt(){ find "$1" -name 'model_epoch_2000.pth' | head -1; }
MCK=$(find_ckpt "$RUN/bc_rnn_mimicgen")
TCK=$(find_ckpt "$RUN/bc_rnn_tsphygen")
log "evaluating MimicGen checkpoint $MCK"
$PY robomimic/scripts/run_trained_agent.py --agent "$MCK" --n_rollouts 50 --horizon 800 --env CoffeePreparation_D0 --seed 781 2>&1 | tee "$LOGDIR/eval_mimicgen.log"
log "evaluating TS-PhyGen checkpoint $TCK"
$PY robomimic/scripts/run_trained_agent.py --agent "$TCK" --n_rollouts 50 --horizon 800 --env CoffeePreparation_D0 --seed 781 2>&1 | tee "$LOGDIR/eval_tsphygen.log"
log "DONE"
