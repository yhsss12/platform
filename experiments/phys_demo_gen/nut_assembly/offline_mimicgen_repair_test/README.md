# Offline MimicGen Repair Test

> **定位**：在实验目录内验证 V1-E PINN 能否作为 MimicGen **失败示范的 repair layer**，提高成功 demonstration 产出率。  
> 这是 **Offline** 测试；后续再做 **Online MimicGen Plugin**（不在本阶段）。

## 与 V1-D / V1-E guided search 的区别

| 阶段 | 目标 |
|------|------|
| V1-D | 轨迹 energy predictor + 方法验证 |
| V1-E guided search | 候选 θ 预筛选 + rollout 对比 |
| **Offline MimicGen Repair** | 将 repair 流程接到 MimicGen failed demos，产出 `repaired_dataset.hdf5` |

## 流程

1. 读取 `demo_failed.hdf5` 中 failed demo
2. baseline rollout → 提取 **failed context**
3. 按 `failure_type` 采样 **1000** 组 repair θ
4. V1-E PINN 预测 `E_total` / `success_prob`
5. 选择 **top-20** θ（对比 random / explicit energy top-20）
6. trajectory refiner 生成 refined waypoint → MuJoCo rollout
7. **success** → 写入 `repaired_dataset.hdf5`（仅 `success_flag=true`）
8. **failure** → 写入 `failures.json`

## 约束

- **不修改** HDF5 `object_poses`（原样复制）
- 不修改平台正式代码、不接前端
- 不使用 PINA
- 默认 **不跑 demo_3**（lift_failed 留待 V2-B5）

## 运行

```bash
cd experiments/phys_demo_gen/nut_assembly/offline_mimicgen_repair_test

MUJOCO_GL=egl PYTHONPATH=../../../../integrations/CableThreadingMVP \
  python run_offline_mimicgen_repair_test.py \
    --demo-keys demo_4,demo_2 \
    --num-samples 1000 \
    --top-k 20
```

前置条件：

- `mnt/data/demo_failed.hdf5`
- `outputs/v1_repair_parameter_model/model.pt`（先跑 V1-E 训练）
- demo_4 insertion 需 `outputs/cem_refinement/cem_refinement_report.json`

## 输出

`outputs/offline_mimicgen_repair_test/`

| 文件 | 内容 |
|------|------|
| `repaired_dataset.hdf5` | 仅含 repair 成功的 demonstration |
| `failures.json` | 失败 rollout 记录 |
| `offline_mimicgen_repair_report.json` | 成功率、rollout_budget、PINN vs random 对比 |

## 验收标准

1. demo_4 / demo_2 上 PINN top-20 能复现 repair success
2. `repaired_dataset.hdf5` 只含 `success_flag=true`
3. 不直接修改 `object_poses`
4. 报告含 `repair_success_rate`、`rollout_budget`、PINN vs random 对比

## 模块

| 文件 | 作用 |
|------|------|
| `config.py` | 路径与 demo 配置 |
| `repair_common.py` | context 提取、采样、PINN 打分 |
| `repair_rollout.py` | 带 actions/states 记录的 rollout |
| `repaired_hdf5_writer.py` | 写入 MimicGen 风格 HDF5 |
| `run_offline_mimicgen_repair_test.py` | CLI 入口 |

## 后续

- **Online MimicGen Plugin**：在 MimicGen 数据生成管线中在线调用 PINN repair layer
- **V2-B5**：demo_3 lift_failed 专项 refinement 后再纳入本测试
