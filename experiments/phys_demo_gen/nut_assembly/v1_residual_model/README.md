# V1-A / V1-B / V1-C：PyTorch Residual Energy Model 原型

> **定位**：这是 **可学习 residual model 原型**，不是最终 PINN，也不使用 PINA。

## 版本概览

| 版本 | 数据集 | 样本量（参考） | 特征维 | 失败模式覆盖 |
|------|--------|----------------|--------|--------------|
| **V1-A** | demo_4 insertion + V0/V2-A/V2-B2.6 | 56 | 17 | insertion 为主 |
| **V1-B** | V1-A + V2-B3 transport_refinement | 112 | 33 | transport / alignment / insertion / grasp_failed |
| **V1-C** | V1-B + V2-B4 grasp_refinement | ~140+ | 45 | + lift_failed / grasp_improved_but_failed / refined_success |

## V1-C 目标

将 `outputs/grasp_refinement/` 的真实 sim rollout 接入训练数据，扩展 grasp-aware 特征与 grasp/lift 监督头，使模型能区分：

- `grasp_failed` / `lift_failed`
- `grasp_improved_but_failed` / `refined_success`
- demo_2 refined_success vs demo_3 lift_failed

**V1-C 仍不是最终 PINN / PINA。** 后续建议：

- **V2-B5**：demo_3 lift_failed 专项 refinement
- **V1-D**：trajectory residual predictor（PINN-style physics-informed losses）
- **V1-E**：repair-parameter residual field（`repair_parameter_model/`，PINN 核心修正模型）
- **Offline MimicGen Repair Test**：`offline_mimicgen_repair_test/`（V1-E PINN 接入 failed demo 修复流程）

## 数据来源（V1-C 新增）

| 文件 | 用途 |
|------|------|
| V1-B 全部来源 | HDF5 / CEM / sim_in_loop / transport_refinement |
| `outputs/grasp_refinement/grasp_refinement_report.json` | per-demo original / best / top-10 |
| `outputs/grasp_refinement/grasp_refinement_summary.csv` | 校验汇总 |
| `outputs/grasp_refinement/top_candidates.csv` | top 候选 rollout |
| `outputs/grasp_refinement/per_demo_best.json` | Level G1/G2/G3 验收 |

### V2-B4 样本来源标签

- `grasp_original_waypoint`
- `grasp_refined_success` / `grasp_improved_but_failed` / `grasp_no_improvement`
- `grasp_top_candidate`
- `lift_failed_candidate`

**注意**：demo_3 的 `grasp_improved_but_failed` / `lift_failed` **刻意保留**，不丢弃失败样本。

## 特征（V1-C，45 维）

**V1-B 33 维** + **Grasp search params（9）**：

- `grasp_xy_offset_x/y`, `pre_grasp_height`, `approach_height`
- `grasp_gripper_close_shift`, `grasp_gripper_hold_steps`
- `grasp_lift_height`, `grasp_lift_steps`, `grasp_speed_scale`

**Grasp extra（3）**：`nut_lift_delta`, `grasp_success_proxy_feat`, `lift_success_proxy_feat`

## 标签

**failure_type**：`success`, `transport_failed`, `alignment_failed`, `insertion_failed`, `grasp_failed`, **`lift_failed`**, `smoothness_issue`, `unknown_failed`

**outcome**：`success`, `refined_success`, `improved_but_failed`, **`grasp_improved_but_failed`**, `no_improvement`, **`grasp_no_improvement`**, `baseline`, `search_candidate`, **`candidate_ready`**, **`failed`**

**扩展 targets**：`grasp_success_proxy`, `lift_success_proxy`, `level_g1/g2/g3_pass`, `nut_lift_delta`, `nut_displacement_after_grasp`

## 模型与损失（V1-C）

- 结构：与 V1-B 相同 MLP（hidden=128, 3 层）
- 新增头：`grasp_success_logit`, `lift_success_logit`
- 损失：
  ```
  L = L_energy + 0.5*L_components + 0.5*L_success + 0.2*L_failure + 0.2*L_outcome
    + 0.3*L_grasp_success + 0.3*L_lift_success + 0.5*L_consistency
  ```

## 快速运行（V1-C）

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model

# 1. 构建 V1-C 数据集（需先完成 V2-B3 + V2-B4）
python build_training_dataset.py --dataset-version v1c

# 2. 训练
python train_residual_model.py --model-version v1c --epochs 300

# 3. 评估（含 V1-B vs V1-C 对比）
python evaluate_residual_model.py --model-version v1c
```

输出目录：`outputs/v1_residual_model_v1c/`

## V1-B 快速运行

```bash
python build_training_dataset.py --dataset-version v1b
python train_residual_model.py --model-version v1b --epochs 300
python evaluate_residual_model.py --model-version v1b
```

输出目录：`outputs/v1_residual_model_v1b/`

## V1-A 快速运行

```bash
python build_training_dataset.py --dataset-version v1a
python train_residual_model.py --model-version v1a --epochs 300
python evaluate_residual_model.py --model-version v1a
```

输出目录：`outputs/v1_residual_model/`

## 输出文件

| 文件 | 内容 |
|------|------|
| `training_dataset.npz` / `.jsonl` | 训练数据 |
| `training_dataset_summary.json` | 来源 / failure / outcome 分布 |
| `model.pt` | 模型权重 |
| `train_log.json` | 训练曲线 |
| `evaluation_report.json` | 评估指标 |
| `predictions.csv` | 逐样本预测（含 grasp/lift 头） |
| `v1b_vs_v1c_comparison.json` | V1-B vs V1-C 对比（V1-C 评估时） |

## 重要说明

1. **V1-C 是 grasp-aware multi-failure residual model**，样本来自 V0/V0.5、V2-A、V2-B2.6、V2-B3、**V2-B4**。
2. **不是最终 PINN / PINA**；physics consistency 仅通过加权 energy 分量约束。
3. demo_2 `refined_success` 与 demo_3 `lift_failed` 应在 `predictions.csv` 中可区分。
4. V1-A / V1-B 仍可作为 baseline，不代表完整任务泛化。

## V1-C.5：Group-split / Leave-one-demo-out 泛化验证

**随机 split 指标可能因 demo 级泄漏而偏乐观；group split 才是泛化验证。**

V1-C.5 用于判断模型是否能在未见 demo 上保持 energy / success / grasp / lift 识别能力，并决定是否进入 **V2-B5** 或 **PINA/PINN formalization**。

### 样本分组字段

每个样本 meta 含：

- `source_demo`: demo_0 … demo_4
- `source_failure_mode`: success / insertion_failed / transport_failed / grasp_failed / lift_failed
- `sample_source`: 与 `source` 相同（hdf5_baseline / grasp_refined_success / …）

### Split 策略

1. **leave_one_demo_out**（5 折）：每次 hold out 一个 demo 的全部样本
2. **failure_mode_holdout**（3 折）：holdout insertion_failed / transport_failed / grasp_lift_failed

### 运行

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model

# 全量 group split 训练 + 评估（每个 split 独立训练 300 epoch）
python evaluate_group_split.py

# 或单独训练某一 split
python train_group_split_model.py --split-id leave_one_demo_out__demo_3
```

输出目录：`outputs/v1_residual_model_v1c_group_split/`

| 文件 | 内容 |
|------|------|
| `group_split_report.json` | 全量 split 结果 + 泛化风险说明 |
| `group_split_summary.csv` | 各 split 指标汇总 |
| `per_split_predictions.csv` | 各 split 测试集预测 |
| `confusion_matrices.json` | failure_type / outcome 混淆矩阵 |
| `v1c_random_vs_group_split_comparison.json` | 随机 split vs group split 对比 |

### 解读要点

- 若 group split 指标**显著低于**随机 split → 报告会标记 **generalization_risk=high**
- 若 **demo_3 / lift_failed** 在 leave-one-demo-out 下不稳定 → 标记 **V2-B5 优先**
- V1-C.5 **仍不是 PINN / PINA**；只是泛化验收门禁

## V1-D：PINN-style Physics-Informed Model

详见 **`README_PINN.md`**。

- 显式 Nut Assembly 几何/抓取物理残差写入 loss（非 PDE PINN，非 PINA）
- **主要评估**：group split / leave-one-demo-out
- 输出：`outputs/v1_residual_model_pinn/`

```bash
python train_pinn_residual_model.py --epochs 300
python evaluate_pinn_group_split.py
python compare_v1c_vs_pinn.py --ablation-epochs 150
```
