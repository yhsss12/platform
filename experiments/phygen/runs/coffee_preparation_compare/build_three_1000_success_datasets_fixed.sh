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
BASE_CONFIG=$GENROOT/configs/demo_src_coffee_preparation_D0_1000trial_keepall.json
SOURCE=$GENROOT/source/coffee_preparation_prepared.hdf5
LOGDIR=$RUN/logs
OUT=$RUN/three_1000_success_datasets
mkdir -p "$LOGDIR" "$OUT" "$GENROOT/configs"
LOG=$LOGDIR/build_three_1000_success_datasets_fixed.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }
count_demos(){ $PY - "$1" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1],'r') as f: print(len(f['data'].keys()))
PY
}
summary_repaired(){ $PY - "$1" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
print(s.get('repaired_count_in_evaluated', s.get('repaired_count', 0)))
PY
}
C=$(count_demos "$SUCCESS")
RTS=$(summary_repaired "$RUN/ts_repair_export/summary.json")
RSH=$(summary_repaired "$RUN/shared_phygen_export/summary.json")
if [ "$RTS" -ge "$RSH" ]; then
  HIGH_NAME=tsphygen; LOW_NAME=sharedphygen; RHIGH=$RTS; RLOW=$RSH
  HIGH_REP=$RUN/ts_repair_export/demo_ts_repaired_success.hdf5
  LOW_REP=$RUN/shared_phygen_export/demo_shared_repaired_success.hdf5
else
  HIGH_NAME=sharedphygen; LOW_NAME=tsphygen; RHIGH=$RSH; RLOW=$RTS
  HIGH_REP=$RUN/shared_phygen_export/demo_shared_repaired_success.hdf5
  LOW_REP=$RUN/ts_repair_export/demo_ts_repaired_success.hdf5
fi
G2=$((1000 - C - RHIGH)); G3=$((RHIGH - RLOW)); G4=$RLOW
log "counts: C=$C R_ts=$RTS R_shared=$RSH high=$HIGH_NAME:$RHIGH low=$LOW_NAME:$RLOW G2=$G2 G3=$G3 G4=$G4"
make_gen_config(){
  local name=$1 n=$2 out=$3
  $PY - "$BASE_CONFIG" "$name" "$n" "$out" <<'PY'
import json, sys
base,name,n,out=sys.argv[1:]
c=json.load(open(base))
c['experiment']['name']=name
c['experiment']['generation']['guarantee']=True
c['experiment']['generation']['keep_failed']=True
c['experiment']['generation']['num_trials']=int(n)
# Do not add max_num_failures here; this MimicGen config is key-locked and does not define it.
c['experiment']['render_video']=False
c['experiment']['num_demo_to_render']=0
json.dump(c, open(out,'w'), indent=4)
PY
}
run_gen(){
  local label=$1 n=$2 seed=$3
  local cfg=$GENROOT/configs/demo_src_coffee_preparation_D0_${label}_${n}_fixed.json
  local folder=$GENROOT/coffee_preparation_d0_${label}_${n}_seed${seed}_success_fixed
  make_gen_config "demo_src_coffee_preparation_D0_${label}_${n}_fixed" "$n" "$cfg"
  if [ "$n" -eq 0 ]; then return; fi
  if [ ! -f "$folder/demo_src_coffee_preparation_task_D0/demo.hdf5" ]; then
    log "generating $label success demos n=$n seed=$seed"
    cd "$ROOT"
    $PY mimicgen/scripts/generate_dataset.py --config "$cfg" --source "$SOURCE" --folder "$folder" --seed "$seed" --auto-remove-exp 2>&1 | tee "$LOGDIR/generate_${label}_fixed.log"
  else
    log "reuse existing $label $folder"
  fi
}
run_gen gen2 "$G2" 17202
run_gen gen3 "$G3" 17203
run_gen gen4 "$G4" 17204
G2_HDF5=$GENROOT/coffee_preparation_d0_gen2_${G2}_seed17202_success_fixed/demo_src_coffee_preparation_task_D0/demo.hdf5
G3_HDF5=$GENROOT/coffee_preparation_d0_gen3_${G3}_seed17203_success_fixed/demo_src_coffee_preparation_task_D0/demo.hdf5
G4_HDF5=$GENROOT/coffee_preparation_d0_gen4_${G4}_seed17204_success_fixed/demo_src_coffee_preparation_task_D0/demo.hdf5
log "building three 1000-demo hdf5 datasets"
rm -f "$OUT"/demo_*_1000_success.hdf5 "$OUT"/three_1000_dataset_build_summary.json
$PY - "$SUCCESS" "$HIGH_REP" "$LOW_REP" "$G2_HDF5" "$G3_HDF5" "$G4_HDF5" "$OUT" "$HIGH_NAME" "$LOW_NAME" "$C" "$RTS" "$RSH" "$G2" "$G3" "$G4" <<'PY'
import h5py, json, sys
from pathlib import Path
(success, high_rep, low_rep, g2, g3, g4, outdir, high_name, low_name, C, RTS, RSH, G2, G3, G4)=sys.argv[1:]
outdir=Path(outdir); C=int(C); RTS=int(RTS); RSH=int(RSH); G2=int(G2); G3=int(G3); G4=int(G4)
def keynum(k):
    try: return int(k.split('_')[-1])
    except Exception: return 10**12
def copy_first(src_path, data_out, start_idx, n, label, is_repaired):
    copied=0
    if n <= 0: return start_idx, copied
    with h5py.File(src_path,'r') as src:
        keys=sorted(src['data'].keys(), key=keynum)[:n]
        if len(keys) < n: raise RuntimeError(f'{src_path} has {len(keys)} demos, need {n}')
        for key in keys:
            src['data'].copy(key, data_out, name=f'demo_{start_idx}')
            g=data_out[f'demo_{start_idx}']; g.attrs['subset_source']=label; g.attrs['is_repaired']=bool(is_repaired)
            start_idx += 1; copied += 1
    return start_idx, copied
def build(out_path, parts):
    out_path=Path(out_path)
    with h5py.File(success,'r') as base, h5py.File(out_path,'w') as out:
        for k,v in base.attrs.items(): out.attrs[k]=v
        data=out.create_group('data')
        for k,v in base['data'].attrs.items(): data.attrs[k]=v
        idx=0; counts={}
        for path,n,label,is_rep in parts:
            idx,copied=copy_first(path, data, idx, n, label, is_rep); counts[label]=copied
        total=sum(int(data[k].attrs.get('num_samples', data[k]['actions'].shape[0])) for k in data.keys())
        data.attrs['total']=total; out.attrs['num_demos']=len(data.keys())
        if len(data.keys()) != 1000: raise RuntimeError(f'{out_path} has {len(data.keys())}, expected 1000')
    return counts
high_R=max(RTS,RSH); low_R=min(RTS,RSH)
high_out=outdir/f'demo_{high_name}_1000_success.hdf5'; low_out=outdir/f'demo_{low_name}_1000_success.hdf5'; mg_out=outdir/'demo_mimicgen_1000_success.hdf5'
high_counts=build(high_out, [(success,C,'gen1_success',False),(high_rep,high_R,f'{high_name}_repaired',True),(g2,G2,'gen2_shared_tail',False)])
low_counts=build(low_out, [(success,C,'gen1_success',False),(low_rep,low_R,f'{low_name}_repaired',True),(g3,G3,'gen3_low_tail',False),(g2,G2,'gen2_shared_tail',False)])
mg_counts=build(mg_out, [(success,C,'gen1_success',False),(g4,G4,'gen4_mimicgen_only',False),(g3,G3,'gen3_low_tail',False),(g2,G2,'gen2_shared_tail',False)])
summary={'base_success_C':C,'ts_repaired_R':RTS,'shared_repaired_R':RSH,'high_repair_group':high_name,'low_repair_group':low_name,'gen2_count':G2,'gen3_count':G3,'gen4_count':G4,'datasets':{high_name:str(high_out),low_name:str(low_out),'mimicgen':str(mg_out)},'counts':{high_name:high_counts,low_name:low_counts,'mimicgen':mg_counts},'final_demo_count_each':1000}
(outdir/'three_1000_dataset_build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
log "THREE_1000_DATASETS_READY: $OUT"
