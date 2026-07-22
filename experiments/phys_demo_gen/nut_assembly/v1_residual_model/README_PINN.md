# V1-D：PINN-style Physics-Informed Neural Residual Energy Model

> **定位**：这是 **PINN-style residual energy model**（PyTorch），**不是 PDE PINN**，**不使用 PINA**，**不宣称最终泛化模型**。
>
> **V1-D** = trajectory residual predictor（输入 rollout 特征，预测轨迹能量）。  
> **V1-E** = repair-parameter residual field（输入 failed context + θ，见 `repair_parameter_model/README.md`）—— **PINN 作为核心修正模型的版本**。

## 背景

V1-C.5 group split 显示随机 split 偏乐观（LODO Pearson 0.797 vs random 0.996）。V1-D 将 Nut Assembly 的**显式物理残差方程**写入神经网络训练损失，形成 physics-informed residual energy model。

## 物理约束来源

来自 Nut Assembly 几何 / 插入 / 抓取 / 平滑残差（与 V0.5 energy model 一致）：

```
E_xy_phys        = final_nut_peg_xy / 0.03
E_transport_phys = min_nut_peg_xy / 0.03
E_yaw_phys       = min_yaw_error / 0.05
E_z_phys         = max(0, final_z_diff - 0.02) / 0.02
E_smooth_phys    = action_accel_max / 2.5

E_total_consistency = 3*E_xy + 3*E_transport + 2*E_yaw + 2*E_z + 0.2*E_smooth
```

## 模型

`PINNResidualEnergyModel`：45 维输入 → Softplus 非负 `E_components` + `E_total` + success / failure / outcome / grasp / lift 头。

## 损失

```
L = L_energy + 0.5*L_components + 0.5*L_success + 0.2*L_failure + 0.2*L_outcome
  + 0.3*L_grasp + 0.3*L_lift
  + 1.0*L_phys_components + 1.0*L_total_consistency + 0.5*L_margin
```

- `L_phys_components`：预测分量 vs 显式物理 residual
- `L_total_consistency`：E_total vs 加权分量之和
- `L_margin`：success 能量均值 < failed 能量均值 - margin（margin=1.0）

## 数据

使用 V1-C 数据集（45 维）：

```
outputs/v1_residual_model_v1c/training_dataset.npz
```

## 快速运行

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model

# 1. 训练 PINN（random split，保存 model.pt）
python train_pinn_residual_model.py --epochs 300

# 2. 评估：random + leave-one-demo-out group split（主要指标）
python evaluate_pinn_group_split.py --epochs 300

# 3. V1-C vs PINN 对比 + physics loss ablation
python compare_v1c_vs_pinn.py --ablation-epochs 150
```

## 输出目录

`outputs/v1_residual_model_pinn/`

| 文件 | 内容 |
|------|------|
| `model.pt` | PINN 权重 |
| `train_log.json` | 训练曲线 |
| `evaluation_report.json` | random split 指标 |
| `group_split_report.json` | **主要** LODO 泛化指标 |
| `predictions.csv` | random 模型全量预测 |
| `per_split_predictions.csv` | group split 预测 |
| `v1c_vs_pinn_comparison.json` | V1-C vs PINN |
| `physics_loss_ablation.json` | 物理损失消融 |

## 评估原则

1. **必须看 group split / leave-one-demo-out**，不要只汇报 random split。
2. 关注 demo_3 `lift_failed`、demo_2 `grasp_refined_success`、demo_4 insertion 在 holdout 下的表现。
3. physics ablation 应显示 full PINN loss 不劣于 no-physics。

## 局限与后续

- 当前泛化仍受 **demo 数量（demo_0–4）** 限制。
- 这是 **task-specific physics-informed residual model**，不是流体/PDE PINN。
- 后续可接 **PINA** 做框架化实现；当前 intentionally 纯 PyTorch。

## 方法验证（终端层）

在 **不接平台 / 不接前端** 的前提下，验证 PINN 是否可用于 **候选排序** 与 **sim-in-loop refinement 引导**：

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model

# 1. 多 seed LODO 稳定性（seeds 0–4，输出均值/方差）
python run_pinn_stability_check.py --epochs 150

# 2. 读取 V2-B2.6 / B3 / B4 候选，对比 explicit vs PINN 排序
python run_pinn_candidate_ranking.py

# 3. demo_4 / demo_2 / demo_3：random vs explicit_energy vs pinn_energy 搜索
MUJOCO_GL=egl PYTHONPATH=../../../integrations/CableThreadingMVP \
  python run_pinn_guided_search.py --max-evals 40
```

| 脚本 | 输出 |
|------|------|
| `run_pinn_stability_check.py` | `stability_report.json` |
| `run_pinn_candidate_ranking.py` | `ranking_report.json`, `ranking_predictions.csv` |
| `run_pinn_guided_search.py` | `pinn_guided_search_report.json` |

## 文件

| 文件 | 作用 |
|------|------|
| `pinn_residual_energy_model.py` | 模型 + physics losses |
| `pinn_inference.py` | 推理 / sim-in-loop 打分 |
| `train_pinn_residual_model.py` | 训练 |
| `evaluate_pinn_group_split.py` | random + group split 评估 |
| `compare_v1c_vs_pinn.py` | 对比 + ablation |
| `run_pinn_stability_check.py` | 多 seed LODO 稳定性 |
| `run_pinn_candidate_ranking.py` | 候选排序验证 |
| `run_pinn_guided_search.py` | sim-in-loop 搜索对比 |
