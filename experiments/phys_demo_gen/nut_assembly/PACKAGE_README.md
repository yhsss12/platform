# Nut Assembly PINN 实验代码包

本压缩包包含 `experiments/phys_demo_gen/nut_assembly/` 下的**实验代码与文档**（不含 `outputs/` 运行产物）。

## 目录结构

```
nut_assembly/
├── README.md                    # V0–V2 总览
├── *.py                         # V0/V2 sim-in-loop、CEM、transport/grasp 搜索
├── v1_residual_model/           # V1-A/B/C/D + 方法验证
│   ├── README.md
│   ├── README_PINN.md
│   ├── build_training_dataset.py
│   ├── pinn_*.py / train_*.py / evaluate_*.py / run_pinn_*.py
│   └── repair_parameter_model/  # V1-E repair-parameter field
│       ├── README.md
│       └── build/train/eval/run_*.py
└── PACKAGE_README.md            # 本文件
```

## 环境依赖

- Python 3.10+
- PyTorch、numpy、scipy、h5py
- MuJoCo + RoboSuite（sim rollout 需要）
- HDF5 数据：`mnt/data/demo.hdf5`、`demo_failed.hdf5`
- RoboSuite 路径：`integrations/CableThreadingMVP`（`PYTHONPATH` + `MUJOCO_GL=egl`）

## 典型流程

### V1-E（repair-parameter field，推荐）

```bash
cd nut_assembly/v1_residual_model/repair_parameter_model
python build_repair_parameter_dataset.py
python train_pinn_repair_model.py --epochs 250
python evaluate_pinn_repair_model.py
python evaluate_repair_parameter_ranking.py
MUJOCO_GL=egl PYTHONPATH=../../../../integrations/CableThreadingMVP \
  python run_pinn_repair_guided_search.py --num-samples 1000 --top-k 20
```

### V1-D（trajectory PINN + 方法验证）

```bash
cd nut_assembly/v1_residual_model
python build_training_dataset.py --version v1c
python train_pinn_residual_model.py
python evaluate_pinn_group_split.py
python run_pinn_stability_check.py
python run_pinn_candidate_ranking.py
python run_pinn_guided_search.py
```

## 说明

- 未修改平台正式代码，未接前端，未使用 PINA
- 运行产物（模型权重、npz、视频、报告）需本地重新生成，位于 `outputs/`
- V1-D = trajectory residual predictor；V1-E = repair-parameter residual field
