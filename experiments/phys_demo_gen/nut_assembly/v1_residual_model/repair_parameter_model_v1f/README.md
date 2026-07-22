# V1-F Uncertainty-aware PINN Repair Parameter Model

> **定位**：在 V1-E 基础上扩充数据集、增加 lift-aware repair 参数与不确定性估计，重点提升 `demo_3` / `lift_failed` 修复能力与 top-K 效率。

## 与 V1-E 的区别

| 维度 | V1-E | V1-F |
|------|------|------|
| 样本量 | ~157 | 1000~3000（MuJoCo rollout） |
| θ 维度 | 24 | 32（+8 lift-aware 参数） |
| 能量输出 | 5 分量 | 7 分量（+E_grasp, E_lift） |
| lift 残差 | 无 | 5 项专项监督 |
| 不确定性 | 无 | aleatoric uncertainty (log_var) |
| demo_3 | grasp 路径 | 独立 lift refiner |

## 新增 lift-aware 参数

- `micro_lift_height` / `micro_lift_steps`
- `regrasp_shift` / `gripper_extra_close`
- `lift_speed_scale` / `lift_pause_steps`
- `contact_hold_steps` / `post_grasp_settle_steps`
- `lift_direction_bias` / `nut_follow_threshold`

## 新增 lift 残差

- `E_lift_follow` / `E_grasp_contact`
- `E_object_displacement` / `E_eef_nut_coupling` / `E_lift_stability`

## 运行流程

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model/repair_parameter_model_v1f

# 1. MuJoCo rollout 采样扩充数据集（约 1200+ 样本）
MUJOCO_GL=egl PYTHONPATH=../../../../integrations/CableThreadingMVP \
  python run_v1f_rollout_sampling.py --seed 0 --demo-3-budget 500

# 2. 构建 V1-F 数据集（合并 V1-E 历史 + rollout）
python build_v1f_repair_dataset.py

# 3. 训练 V1-F PINN
python train_pinn_v1f_model.py --epochs 300

# 4. 评估：V1-E vs V1-F vs explicit vs random
python evaluate_v1f_repair.py
```

## 输出

`outputs/v1f_repair_parameter_model/`

| 文件 | 内容 |
|------|------|
| `rollout_samples.jsonl` | rollout 原始记录 |
| `repair_parameter_dataset_v1f.npz` | 训练数据集 |
| `model_v1f.pt` | V1-F 模型 |
| `v1f_evaluation_report.json` | success@k / rollouts_per_success / LODO |

## 评估指标

- `success@1/3/5/10/20`
- `rollouts_per_success`
- `repair_success_rate` (budget 10/20)
- `best_E_total`
- `demo_3_lift` 改善对比
- leave-one-demo-out 泛化

## 约束

- 所有样本来自 MuJoCo / RoboSuite rollout
- **不修改** `object_poses` / `states`
- 不做 Online MimicGen Plugin（本阶段）

## 模块

| 文件 | 作用 |
|------|------|
| `v1f_repair_dataset.py` | 扩展 θ / 特征 / 目标 |
| `run_v1f_rollout_sampling.py` | rollout 采样 |
| `build_v1f_repair_dataset.py` | 数据集构建 |
| `pinn_v1f_repair_model.py` | uncertainty PINN |
| `train_pinn_v1f_model.py` | 训练 |
| `evaluate_v1f_repair.py` | 对比评估 |

父目录 lift 组件：

- `lift_energy_model.py`
- `lift_waypoint_refiner.py`
- `lift_sim_search.py`
