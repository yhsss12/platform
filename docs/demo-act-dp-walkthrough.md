# 演示脚本：Diffusion Policy + ACT 训练闭环

> 演示环境保留任务（2026-06-19 稳定收尾）  
> - **ACT**：`train_20260619_163854_b9ab`（线缆穿杆 · 1 epoch · completed）  
> - **DP**：`train_20260619_154349_af03`（物块堆叠 · 2 epoch · completed）

访问入口：**训练中心** `/workspace/training` · **模型资产** `/workspace/resources/model-assets` · **评测** `/workspace/evaluation`

---

## 1. 训练中心整体说明（约 1 分钟）

**要讲什么：**

- 平台训练闭环：**选数据集 → 选模型类型 → 后端自动适配 → 启动训练 → 登记模型资产 →（可选）发起评测**
- 前端**不需要**手动调用 Adapter API；`create_training_job` 内部读取 HDF5 / manifest 自动生成配置
- 当前列表仅保留两条演示任务，便于讲解：
  - 线缆穿杆 + **ACT**（图像 + proprio）
  - 物块堆叠 + **Diffusion Policy**（纯 low_dim）

**操作：**

1. 打开训练中心，指出列表只有 2 条 **已完成** 任务
2. 分别点 **详情**，展示：进度 100%、Loss 曲线、日志、`生成的模型资产` 区域

**不要点：** 批量删除（除非刻意演示删除流程）

---

## 2. 物块堆叠 + Diffusion Policy（约 2 分钟）

**数据集：** `isaac_ds_20260617_224132_a452` · **物块堆叠数据_20260617_2237**  
**演示任务：** `train_20260619_154349_af03`

**要讲什么：**

- 物块堆叠 HDF5 为 **纯 low_dim**（eef / gripper / object），**无相机图像**
- 创建训练时选 **Diffusion Policy**，后端自动：
  - 解析 obs keys → `dp_adapted.yaml`
  - 推断 `action_dim`、`state_dim`、`horizon`、`n_obs_steps` 等
  - 启动 `train_dp.py`
- 产物路径：`runtime_outputs/training/jobs/{train_job_id}/config/dp_adapted.yaml`

**详情页演示：**

1. 打开 DP 任务详情 → **训练过程性能**（loss 曲线有数据）
2. **生成的模型资产** → **Final** · 状态 **就绪**
3. **发起评测** 按钮 **可点击**（蓝色链接）

**可选：现场新建（1 epoch）**

- 新建训练 → 数据集选 **物块堆叠** → 模型 **Diffusion Policy** → Epochs=1
- 无需 Adapter 弹窗；创建后列表出现新任务

---

## 3. 线缆穿杆 + ACT（约 2 分钟）

**数据集：** `ds_ct_gen_20260618_095819_8aa6` · **线缆穿杆数据_20260618_1022**  
**演示任务：** `train_20260619_163854_b9ab`

**要讲什么：**

- 线缆穿杆 HDF5 为 **mixed 观测**：`agentview_image`、`robot0_eye_in_hand_image` + proprio
- 创建训练时选 **ACT**，后端自动：
  - 生成 `act_adapted.yaml`（`image_keys`、`low_dim_keys`、`chunk_size=68`、`action_dim=7`、`state_dim=9`）
  - 启动 `train_act.py`
- ACT 需要 **图像观测**；纯 low_dim 数据集会在 **创建阶段 400 拒绝**（不会留下半残 job）

**详情页演示：**

1. 打开 ACT 任务详情 → Loss 曲线、日志含 `train_act.py` / `Epoch`
2. **Final** 模型资产 · **就绪**
3. **发起评测** 为灰色禁用，hover 提示：**当前评测后端暂不支持 ACT**

**反例（可选，10 秒）：**

- 新建训练 → **物块堆叠** + **ACT** → toast 显示  
  `ACT requires image observations, but dataset has only low_dim observations.`

---

## 4. 模型资产注册说明（约 1 分钟）

**路径：** `/workspace/resources/model-assets`

**要讲什么：**

- 训练完成后自动写入 `model_assets` 表 + runtime registry
- 每条 Final 资产包含：`modelType`、`checkpointPath`、`sourceTrainingJobId`、`status=ready/available`
- 演示保留资产：
  - **ACT Final**：`model__163854_b9ab_final`
  - **DP Final**：`model__154349_af03_final`

**操作：**

1. 打开模型资产页，筛选/浏览可见上述两条
2. 说明支持 **删除 / 批量删除**（演示环境勿删演示资产）

---

## 5. 评测入口说明（约 1 分钟）

**路径：** 训练详情 → **发起评测** · 或 **评测中心** `/workspace/evaluation`

| 模型 | 发起评测 | 说明 |
|------|----------|------|
| **DP Final** | ✅ 可用 | 跳转评测弹窗，默认 **模型评测**，预填 DP 资产 |
| **ACT Final** | ❌ 禁用 | Tooltip：当前评测后端暂不支持 ACT |

**DP 演示步骤：**

1. 打开 DP 训练详情 → 点击 **发起评测**
2. 确认进入评测创建弹窗，模式为模型评测，资产已选中

---

## 6. 当前限制（演示前必读）

1. **ACT 暂不支持评测** — 按钮故意禁用，非 bug  
2. **ACT 不支持 pure low_dim** — 物块堆叠等无图像数据集无法创建 ACT 任务  
3. **ACT 真实训练较慢** — 线缆穿杆 1 epoch（256×256 级图像）约 **7 分钟/GPU**；演示优先用已有 completed 任务  
4. **服务重启后** — 平台 startup 会 reindex；completed / ready 状态应保持稳定（已验收）

---

## 7. 快速链接（详情页）

| 任务 | 直达 |
|------|------|
| ACT 详情 | `/workspace/training?jobId=train_20260619_163854_b9ab` |
| DP 详情 | `/workspace/training?jobId=train_20260619_154349_af03` |

---

## 8. 关键产物路径（备查）

```
runtime_outputs/training/jobs/train_20260619_163854_b9ab/
  config/act_adapted.yaml
  checkpoints/act/checkpoints/model_final.pt

runtime_outputs/training/jobs/train_20260619_154349_af03/
  config/dp_adapted.yaml
  checkpoints/model_final.pt
```
