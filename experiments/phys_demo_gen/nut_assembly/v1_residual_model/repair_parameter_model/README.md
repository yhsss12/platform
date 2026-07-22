# V1-E：PINN-guided Repair Parameter Selection

> **定位**：V1-E 将 PINN 从 trajectory energy predictor（V1-D）升级为 **repair-parameter residual field**。  
> 输入 failed demo 上下文 + 候选修正参数 θ，输出该 θ 经仿真修正后的物理残差与成功概率，用于 **候选预筛选、减少 sim-in-loop rollout 次数**。

## 与 V1-D 的区别

| 版本 | 输入 | 角色 |
|------|------|------|
| **V1-D** | rollout 后轨迹/能量特征 | trajectory residual predictor |
| **V1-E** | failed context + repair θ + mask | repair-parameter residual field |

显式 energy **不是主方法**，仅作 physics supervision / baseline。

## 快速运行

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model/repair_parameter_model

# 1. 构建 repair-parameter 数据集
python build_repair_parameter_dataset.py

# 2. 训练 PINNRepairParameterModel
python train_pinn_repair_model.py --epochs 250

# 3. 评估（random split + ranking）
python evaluate_pinn_repair_model.py
python evaluate_repair_parameter_ranking.py

# 4. PINN 预筛选 + 少量 sim rollout（需 MuJoCo）
MUJOCO_GL=egl PYTHONPATH=../../../../integrations/CableThreadingMVP \
  python run_pinn_repair_guided_search.py --num-samples 1000 --top-k 20
```

## 输出目录

`outputs/v1_repair_parameter_model/`

| 文件 | 内容 |
|------|------|
| `repair_parameter_dataset.npz` | 训练数据 |
| `repair_parameter_dataset.jsonl` | 可读样本 |
| `model.pt` | PINNRepairParameterModel |
| `train_log.json` | 训练曲线 |
| `evaluation_report.json` | random split 指标 |
| `repair_parameter_ranking_report.json` | 候选排序对比 |
| `repair_guided_search_report.json` | 预筛选 + rollout 对比 |

## 数据样本结构

- **A. failed demo context**：source_demo、failure_type、original 几何/能量指标  
- **B. repair θ**：insertion / transport / grasp-lift 参数 + `param_mask`  
- **C. targets**：rollout 后 success、E 分量、grasp/lift proxy、failure_type、outcome  

## Physics-informed loss

1. component supervised loss  
2. total consistency：`E_total ≈ 3E_xy + 3E_transport + 2E_yaw + 2E_z + 0.2E_smooth`  
3. success-energy margin  
4. monotonic repair constraint（θ 朝修复方向时预测能量不应升高）

## 验收标准

1. 数据集构建成功  
2. 模型训练完成  
3. ranking 中 PINN top-k **不低于 random**  
4. 至少一个 demo 上 PINN top-20 rollout 达到 refined_success 或 E_total 明显低于 random  
5. 若 PINN 不如 explicit，报告 per-pool 原因；论文定位为 **PINN-guided candidate pruning**

## 局限

- 非 PINA；非跨任务泛化模型  
- `outputs/lift_refinement/` 尚未存在时，demo_3 lift 样本来自 grasp_refinement  
- 不接平台 / 前端

## 首次运行结果摘要（157 样本）

| 指标 | PINN | Explicit baseline | Random |
|------|------|-------------------|--------|
| Pearson E_total | **0.986** | 0.564 | — |
| macro top-1 refined_success | 0.50 | 0.75 | 0.25 |

**Guided search（N=1000, top-K=20）**：

| Demo | PINN top-20 | random top-20 |
|------|-------------|---------------|
| demo_4 insertion | success, min E=**1.55** | fail, min E=4.43 |
| demo_2 grasp | **refined_success**, min E=1.12 | fail, min E=100.6 |
| demo_3 lift | fail（与 explicit/random 相近） | fail |

**结论**：V1-E 证明 PINN 可作为 **repair-parameter 预筛选器**（demo_4/2 在 20 次 rollout 预算下显著优于 random top-20）；explicit 在 demo_2 top-1 排序上仍略优，论文定位应为 **PINN-guided candidate pruning**，而非完全替代 explicit physics。
