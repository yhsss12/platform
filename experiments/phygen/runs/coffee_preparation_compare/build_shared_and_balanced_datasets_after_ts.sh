#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/nvidia:/home/ubuntu/.mujoco/mujoco210/bin"
PY=/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python
ROOT=/home/ubuntu/project/eai-idev2.1/third_party/mimicgen
EXP=/home/ubuntu/project/eai-idev2.1/experiments/phygen/runs/coffee_preparation_compare
RUN=$EXP/auto_mg_vs_ts_seed17201
LOGDIR=$RUN/logs
GENROOT=/home/ubuntu/project/eai-idev2.1/mimicgen_generated/coffee_preparation_compare_20260716
DATA_DIR=$GENROOT/coffee_preparation_d0_seed17201_trial1000_keepall/demo_src_coffee_preparation_task_D0
SUCCESS=$DATA_DIR/demo.hdf5
mkdir -p "$LOGDIR" "$RUN/shared_phygen_export" "$RUN/datasets_balanced"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/shared_dataset_builder.log"; }
count_demos(){ $PY - "$1" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1],'r') as f: print(len(f['data'].keys()))
PY
}
log "waiting for TS repair summary"
while [ ! -f "$RUN/ts_repair_export/summary.json" ]; do sleep 120; done
RTS=$($PY - <<PY
import json
s=json.load(open('$RUN/ts_repair_export/summary.json'))
print(s.get('repaired_count_in_evaluated',0))
PY
)
log "TS repair summary ready: R_ts=$RTS"
log "waiting for MimicGen/TS merged datasets from current pipeline"
while [ ! -f "$RUN/datasets/demo_mimicgen_C_plus_R.hdf5" ] || [ ! -f "$RUN/datasets/demo_tsphygen_C_plus_R.hdf5" ]; do sleep 120; done
log "running Shared-PhyGen universal repair export"
PYTHONPATH="$EXP/scripts:$ROOT:${PYTHONPATH:-}" $PY $EXP/scripts/universal_rprf_inference.py \
  --task coffee_preparation \
  --dataset "$DATA_DIR" \
  --checkpoint "$EXP/checkpoints/universal_rprf_pinn.pt" \
  --output-dir "$RUN/shared_phygen_export" \
  --budget 5 \
  --pool-size 128 \
  --candidate-mode safe \
  --export-success-hdf5 "$RUN/shared_phygen_export/demo_shared_repaired_success.hdf5" \
  --export-merged-hdf5 "$RUN/shared_phygen_export/demo_shared_merged_all_success.hdf5" \
  2>&1 | tee "$LOGDIR/shared_phygen_export.log"
RSH=$($PY - <<PY
import json
s=json.load(open('$RUN/shared_phygen_export/summary.json'))
print(s.get('repaired_count',0))
PY
)
C=$(count_demos "$SUCCESS")
R=$(( RTS < RSH ? RTS : RSH ))
log "Shared repair complete: R_shared=$RSH; balanced R=min($RTS,$RSH)=$R; base C=$C"
EXTRA_SUCCESS=$(find "$GENROOT" -path '*extra*_success_keepall/*task_D0/demo.hdf5' -type f | sort | tail -1)
if [ -z "$EXTRA_SUCCESS" ]; then echo "missing extra MimicGen success hdf5" >&2; exit 2; fi
$PY - "$SUCCESS" "$EXTRA_SUCCESS" "$RUN/ts_repair_export/demo_ts_repaired_success.hdf5" "$RUN/shared_phygen_export/demo_shared_repaired_success.hdf5" "$RUN/datasets_balanced" "$R" <<'PY'
import h5py, sys
from pathlib import Path
success, extra, ts_rep, sh_rep, outdir, r = sys.argv[1:]
r=int(r); outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
def demo_key(k):
    try: return int(k.split('_')[-1])
    except Exception: return 10**9
def merge(base, addon, out, n_add, addon_label):
    out=Path(out)
    if out.exists(): out.unlink()
    with h5py.File(base,'r') as b, h5py.File(addon,'r') as a, h5py.File(out,'w') as o:
        for k,v in b.attrs.items(): o.attrs[k]=v
        data=o.create_group('data')
        for k,v in b['data'].attrs.items(): data.attrs[k]=v
        total=0; idx=0
        for key in sorted(b['data'].keys(), key=demo_key):
            b['data'].copy(key, data, name=f'demo_{idx}')
            g=data[f'demo_{idx}']; g.attrs['is_repaired']=False; g.attrs['subset_source']='base_success'
            total += int(g.attrs.get('num_samples', g['actions'].shape[0])); idx+=1
        for key in sorted(a['data'].keys(), key=demo_key)[:n_add]:
            a['data'].copy(key, data, name=f'demo_{idx}')
            g=data[f'demo_{idx}']; g.attrs['is_repaired']=addon_label!='mimicgen_extra'; g.attrs['subset_source']=addon_label
            total += int(g.attrs.get('num_samples', g['actions'].shape[0])); idx+=1
        data.attrs['total']=total
merge(success, extra, outdir/'demo_mimicgen_C_plus_Rbalanced.hdf5', r, 'mimicgen_extra')
merge(success, ts_rep, outdir/'demo_tsphygen_C_plus_Rbalanced.hdf5', r, 'tsphygen_repaired')
merge(success, sh_rep, outdir/'demo_sharedphygen_C_plus_Rbalanced.hdf5', r, 'sharedphygen_repaired')
print('built balanced datasets with R=', r)
PY
cat > "$RUN/datasets_balanced/dataset_build_summary.json" <<JSON
{"base_success_C":$C,"ts_repaired_R":$RTS,"shared_repaired_R":$RSH,"balanced_R":$R,"mimicgen_dataset":"$RUN/datasets_balanced/demo_mimicgen_C_plus_Rbalanced.hdf5","tsphygen_dataset":"$RUN/datasets_balanced/demo_tsphygen_C_plus_Rbalanced.hdf5","sharedphygen_dataset":"$RUN/datasets_balanced/demo_sharedphygen_C_plus_Rbalanced.hdf5"}
JSON
log "ALL_THREE_DATASETS_READY: $RUN/datasets_balanced"
