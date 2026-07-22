# Square_D0 / Nut Assembly — V0.5 物理约束轨迹能量模型

本目录是 **显式 Physics-Informed Trajectory Energy Model** 原型，用于对 MimicGen / RoboSuite HDF5 demo 计算结构化物理能量。**本阶段不训练神经网络，不使用 PINA，也不修改平台正式业务代码。**

V0.5 在 V0 基础上增加 **归一化能量（Normalized Physics Energy）** 与 **CEM Readiness Check**，使能量可直接作为后续 CEM waypoint refinement 的优化目标。

## 背景

基于 `mnt/data/residual_audit/` 的残差审计结论（审计程序见
`nut_assembly_residual_audit.py`）：

1. `final_nut_peg_xy_distance` 是最强 success / failed 区分项。
2. `min_nut_peg_xy_distance` 是第二强区分项。
3. `min_nut_peg_yaw_error`（方螺母四重对称 yaw）区分力强。
4. failed `demo_0–3` 为 `transport_failed`：螺母从未接近 peg。
5. failed `demo_4` 为 `insertion_failed`：已靠近 peg，但 final z 为正，螺母悬空未插入。
6. smoothness 指标区分力较弱，权重较低（`w_smooth = 0.2`）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `extract_features.py` | 从 HDF5 提取 nut/peg 几何残差、yaw、动作加速度等特征 |
| `energy_model.py` | Raw + Normalized 能量项、`score_candidate_trajectory` API |
| `run_energy_audit.py` | 批量审计，输出能量报告与贡献比 |
| `run_energy_sensitivity.py` | 对 failed demo_4 做虚拟残差修正敏感性验证 |
| `trajectory_parameterization.py` | 轨迹阶段划分、theta 参数化、proxy 特征重建 |
| `cem_refiner.py` | Residual-Guided CEM 优化器 |
| `run_cem_refinement.py` | 对 failed demos 批量运行 CEM proxy refinement |
| `robosuite_env_loader.py` | V2-B 环境加载、MuJoCo patch、state 复位 |
| `replay_original_demo.py` | V2-B1 原始 action replay |
| `run_rollout_smoke.py` | V2-B1 冒烟入口 |
| `refined_waypoint_builder.py` | V2-B2 theta → eef waypoint |
| `osc_action_converter.py` | V2-B2 waypoint → OSC actions |
| `rollout_refined_demo.py` | V2-B2 refined rollout 执行 |
| `run_refined_rollout.py` | V2-B2 demo_4 对比验证入口 |
| `replay_fidelity_utils.py` | V2-B1.5 三类 replay fidelity 检查 |
| `run_replay_fidelity_check.py` | V2-B1.5 诊断入口 |
| `sim_in_loop_refiner.py` | V2-B2.5 仿真闭环局部搜索 |
| `run_sim_in_loop_refinement.py` | V2-B2.5 demo_4 sim-in-loop 入口 |
| `run_sim_in_loop_repeatability.py` | V2-B2.6 重复性验证 |
| `run_sim_in_loop_ablation.py` | V2-B2.6 消融实验 |
| `transport_waypoint_builder.py` | V2-B3 transport theta + 搜索参数 → eef waypoint |
| `transport_sim_search.py` | V2-B3 transport 随机搜索与打分 |
| `run_transport_refinement.py` | V2-B3 demo_0–3 transport refinement 入口 |
| `grasp_waypoint_builder.py` | V2-B4 grasp 搜索参数 → eef waypoint |
| `grasp_sim_search.py` | V2-B4 grasp 随机搜索与打分 |
| `run_grasp_refinement.py` | V2-B4 demo_2–3 grasp refinement 入口 |
| `v1_residual_model/` | V1-A / V1-B / V1-C PyTorch residual energy model 原型 |

## 能量模型

### V0 Raw Energy（保留）

```
E_total = 3*E_xy + 3*E_transport + 2*E_yaw + 2*E_z + 0.2*E_smooth
```

### V0.5 Normalized Energy（CEM 优化目标）

归一化尺度：

| 参数 | 值 |
|------|-----|
| `xy_threshold` | 0.03 m |
| `transport_threshold` | 0.03 m |
| `yaw_threshold` | 0.05 rad |
| `z_success_target` | -0.021 m |
| `z_tolerance` | 0.02 m |
| `smooth_threshold` | 2.5 |

分项：

```
E_xy_norm        = final_xy / xy_threshold
E_transport_norm = min_xy / transport_threshold
E_yaw_norm       = min_yaw / yaw_threshold
E_z_norm         = max(0, final_z_diff - z_tolerance) / z_tolerance
E_smooth_norm    = action_acceleration_max / smooth_threshold

E_total_norm = 3*E_xy_norm + 3*E_transport_norm + 2*E_yaw_norm + 2*E_z_norm + 0.2*E_smooth_norm
```

贡献比（`contribution_*`）= 加权分项 / `E_total_norm`，用于解释不同失败类型的主导残差。

### 失败分类（按优先级）

- `min_xy <= 0.08 m` 且 `final_z_diff > 0.02 m` → `insertion_failed`
- `min_xy > 0.03 m` → `transport_failed`
- `min_xy <= 0.08 m` 且 `min_yaw > 0.05 rad` → `alignment_failed`
- smoothness 异常 → `smoothness_issue`
- 成功 → `success`

### CEM 候选评分 API

```python
from energy_model import score_candidate_trajectory

score = score_candidate_trajectory(traj_features)
# 返回 E_total_norm, components, failure_type, optimization_targets
```

`optimization_targets` 映射：

| failure_type | 优化参数 |
|--------------|----------|
| `transport_failed` | `transport_xy_offset`, `pre_align_pose`, `gripper_timing` |
| `alignment_failed` | `align_yaw`, `pre_insert_pose` |
| `insertion_failed` | `insert_z`, `insertion_speed`, `release_timing` |
| `smoothness_issue` | `speed_scale`, `waypoint_smoothing` |

## 运行

```bash
cd experiments/phys_demo_gen/nut_assembly

# 能量审计（含归一化能量与贡献比）
python3 run_energy_audit.py \
  --success ../../../mnt/data/demo.hdf5 \
  --failed ../../../mnt/data/demo_failed.hdf5 \
  --output-dir outputs

# 敏感性检查（failed demo_4 虚拟修正）
python3 run_energy_sensitivity.py \
  --failed ../../../mnt/data/demo_failed.hdf5 \
  --output-dir outputs

# V2-A CEM proxy refinement（failed demos）
python3 run_cem_refinement.py \
  --failed ../../../mnt/data/demo_failed.hdf5 \
  --output-dir outputs/cem_refinement \
  --run-success-sanity
```

## 输出

| 文件 | 内容 |
|------|------|
| `outputs/energy_summary.csv` | 每条 demo 的 raw / normalized 能量与贡献比 |
| `outputs/energy_report.json` | 完整报告、验收检查、`contribution_by_failure_type` |
| `outputs/success_vs_failed_energy_comparison.csv` | 归一化能量 success vs failed 对比 |
| `outputs/sensitivity_report.json` | demo_4 虚拟修正敏感性结果 |
| `outputs/sensitivity_summary.csv` | 敏感性场景汇总 |
| `outputs/cem_refinement/cem_refinement_report.json` | CEM before/after 能量与残差对比 |
| `outputs/cem_refinement/cem_refinement_summary.csv` | 每条 failed demo 的 CEM 汇总 |
| `outputs/cem_refinement/per_demo_iteration_history.json` | 每轮 best/mean/elite 能量收敛曲线 |

## V2-A：Residual-Guided CEM Proxy Refinement

V2-A 在 V0.5 归一化能量基础上，实现 **offline proxy-level CEM**，验证 `E_total_norm` 能否指导候选轨迹参数朝低能量方向收敛。

### 重要说明

- **这是 proxy refinement，不是最终仿真验证。**
- 不修改 HDF5 中的 `object_poses`，仅在内存中重建 proxy nut/eef 轨迹。
- 不训练神经网络，不使用 PINA，不修改平台正式代码。
- 下一步 **V2-B** 需接入 RoboSuite / MuJoCo rollout，将优化后的 theta 映射为可执行 waypoint。

### 可优化参数 theta

| 参数 | 含义 | 范围 |
|------|------|------|
| `transport_xy_offset` | grasp 后运输阶段 xy 偏移 | ±0.35 m |
| `pre_align_xy_offset` | t_min_xy 附近对齐微调 | ±0.08 m |
| `align_yaw_offset` | 插入前 yaw 修正 | ±0.785 rad |
| `insert_z_offset` | 插入窗口 z 偏移 | -0.12 ~ 0.02 m |
| `speed_scale` | 动作速度缩放（影响平滑性） | 0.5 ~ 1.2 |
| `gripper_close_shift` | 夹爪闭合时序偏移 | ±10 steps |
| `release_shift` | 释放时序偏移 | ±10 steps |

### Proxy 规则（`apply_theta_to_proxy_features`）

1. **transport window**（grasp → t_min_xy）：偏移 eef/target；夹持时 nut 刚性跟随 eef。
2. **insertion window**（t_min_xy → end）：`insert_z_offset` 作用于 eef/nut z proxy。
3. **align_yaw_offset**：在 t_min_xy 附近旋转 nut yaw proxy（四重对称误差重算）。
4. **speed_scale**：缩放 action 差分，影响 `E_smooth_norm`。
5. **gripper shifts**：平移 gripper_action proxy 时间线。

### CEM 配置（默认）

- `n_samples=128`, `elite_frac=0.1`, `num_iters=5`, `seed=0`
- 目标函数：`score_candidate_trajectory()` → 最小化 `E_total_norm`
- 初始均值：`suggest_initial_theta()` 按 nut→peg 方向与 z/yaw 残差给 hint

### V2-A 验收结果（参考）

| Demo | E_total_norm 变化 | 失败类型变化 |
|------|-------------------|--------------|
| failed demo_0–3 | 下降 57–75% | transport_failed → candidate_ready / lower_energy_candidate |
| failed demo_4 | 下降 ~90% | insertion_failed → candidate_ready |

## 验收标准（V0.5）

1. success 的 `E_total_norm` 均值显著低于 failed。
2. failed `demo_0–3` 仍为 `transport_failed`。
3. failed `demo_4` 仍为 `insertion_failed`。
4. `contribution_*` 能解释失败类型：`transport_failed` 由 xy/transport 主导，`insertion_failed` 由 z 主导。
5. sensitivity check 中，朝正确方向修正残差会降低 `E_total_norm`。
6. **Normalized energy 将作为后续 CEM waypoint refinement 的优化目标。**

## 路线图

| 版本 | 内容 |
|------|------|
| **V0** | 显式 raw 物理能量模型 |
| **V0.5** | 归一化能量 + 贡献比 + CEM 评分 API + 敏感性验证 |
| **V2-A（当前）** | Residual-Guided CEM proxy refinement，验证能量可引导参数收敛 |
| **V2-B（当前）** | RoboSuite / MuJoCo 真实 rollout 验证 refined theta |
| **V2-B1.5（当前）** | Replay Fidelity Calibration — 诊断 action replay 为何不能复现 success |
| **V2-B2.5（当前）** | Simulator-in-the-loop 局部搜索 — current-controller waypoint rollout |
| **V2-B2.6（当前）** | Repeatability + Ablation — 稳定性与 energy-guided 贡献验证 |
| **V2-B3（当前）** | transport_failed demo_0–3 sim-in-loop refinement |
| **V2-B4（当前）** | grasp_failed demo_2–3 grasp-stage sim-in-loop refinement |
| **V1-A（当前）** | PyTorch residual energy model 原型（demo_4 为主） |
| **V1-B（当前）** | Multi-failure-mode residual model（112 样本，33 维） |
| **V1-C（当前）** | Grasp-aware multi-failure residual model（156 样本，45 维，接入 V2-B4） |

## V2-B：RoboSuite / MuJoCo Rollout 验证

V2-B 在 V2-A proxy CEM 之后，将 `best_theta` 映射为 **eef waypoint / OSC action 修正**，在 RoboSuite `NutAssemblySquare`（Square_D0）环境中 open-loop rollout。

### 重要说明

- **V2-B 是真实 rollout 验证；V2-A proxy 成功不等于物理成功。**
- 不修改 HDF5 `object_poses`，不在 sim 中伪造 nut/peg 位姿。
- 不接入前端、不接入 PINA、不修改平台正式业务代码。
- theta 只作用于：eef waypoint、末端姿态、速度尺度、gripper 时序；`insert_z_offset` 作用于 insertion 窗口 z action。

### 新增文件

| 文件 | 作用 |
|------|------|
| `robosuite_env_loader.py` | 环境探测、MuJoCo 3.10 patch、state 复位、指标提取、mp4 |
| `replay_original_demo.py` | 原始 action replay |
| `run_rollout_smoke.py` | V2-B1 冒烟：success demo_0 + failed demo_4 |
| `refined_waypoint_builder.py` | theta → refined eef 4×4 waypoint（rollout-safe attenuation） |
| `osc_action_converter.py` | waypoint / insertion-focused → OSC_POSE 7-dim actions |
| `rollout_refined_demo.py` | refined theta rollout |
| `run_refined_rollout.py` | V2-B2：demo_4 原始 vs refined 对比报告 |

### 环境依赖

```bash
export PYTHONPATH=/path/to/eai-idev2.1/integrations/CableThreadingMVP:$PYTHONPATH
export MUJOCO_GL=egl   # 无显示器 offscreen 渲染
pip install imageio imageio-ffmpeg h5py numpy
```

- **robosuite**：使用 `integrations/CableThreadingMVP` 内 vendored 版本（非 pip 全局包）
- **MuJoCo 3.10**：实验脚本内 monkeypatch `mj_fullM` 签名（不改 integrations 源码）
- **mp4**：需要 `imageio-ffmpeg`

### 已知阻塞 / 限制

1. **旧 demo `model_file` XML 不兼容**：缺少 `robot0_right_center` site，无法 `reset_from_xml_string`；本阶段改用 `states[0]` 向量复位（不伪造 object_poses）。
2. **legacy OSC_POSE → composite controller**：open-loop replay 与录制 state 存在偏差（`replay_state_error_mean` ~0.2–0.4）；指标以 sim body 实时读取为准。
3. **insertion_failed rollout**：proxy 中 transport 修正会移动 proxy nut，但真实 sim 仅动 eef；对 `min_xy` 已较小的 demo_4 使用 **insertion-focused action**（保留 transport，修正 insertion z）。
4. **MuJoCo z 非线性**：过大 z action 修正可能导致 nut 弹飞；默认 `z_gain=0.55`。

### 运行

```bash
cd experiments/phys_demo_gen/nut_assembly

# V2-B1 冒烟
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_rollout_smoke.py

# V2-B2 refined demo_4（需先跑 V2-A 生成 cem_refinement_report.json）
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_refined_rollout.py

# V2-B1.5 replay fidelity 标定（扩展 rollout 前建议先跑）
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_replay_fidelity_check.py
```

### V2-B 输出

| 路径 | 内容 |
|------|------|
| `outputs/rollout_smoke/rollout_result.json` | B1 冒烟结果 + 环境探测 |
| `outputs/rollout_smoke/videos/*.mp4` | 原始 replay 视频 |
| `outputs/refined_rollout/refined_rollout_report.json` | demo_4 原始 vs refined |
| `outputs/refined_rollout/refined_rollout_summary.csv` | 指标汇总 |
| `outputs/refined_rollout/residual_before_after.json` | 残差前后对比 |
| `outputs/refined_rollout/videos/original_failed_demo_4.mp4` | 原始 failed replay |
| `outputs/refined_rollout/videos/refined_demo_4.mp4` | refined rollout |

### V2-B 验收（demo_4 第一阶段）

| 检查项 | 说明 |
|--------|------|
| 环境可加载 | `check_environment().available == true` |
| failed demo_4 replay 仍失败 | `failure_guess == insertion_failed` |
| refined 不伪造 object pose | `object_poses_modified == false` |
| refined E 下降 | `E_total_norm` 低于原始 failed replay |
| refined z 改善 | `final_z_diff` 朝 success 均值 -0.021 靠近 |
| outcome | `refined_success` 或 `improved_but_failed` |

## V2-B1.5：Replay Fidelity Calibration

在扩展 refined rollout 前，先标定 **open-loop action replay 的可信边界**。

### 三类检查

| 检查 | 方法 | 用途 |
|------|------|------|
| **final-state** | `set_state(states[-1])` → success checker | 确认 HDF5 末态在当前 env 是否成功 |
| **state-sequence** | 逐帧 `set_state(states[t])`，不执行 action | 确认 HDF5 states 轨迹本身是否可靠 |
| **action-replay** | `states[0]` 复位 → 逐步 `step(actions[t])` | 测量 controller replay 保真度 |

### 运行

```bash
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_replay_fidelity_check.py
```

### 输出

| 路径 | 内容 |
|------|------|
| `outputs/replay_fidelity/replay_fidelity_report.json` | 完整诊断报告 |
| `outputs/replay_fidelity/replay_fidelity_summary.csv` | 三类检查汇总 |
| `outputs/replay_fidelity/state_sequence_metrics_*.json` | 逐步 state-sequence 指标 |
| `outputs/replay_fidelity/action_replay_metrics_*.json` | 逐步 action replay 指标 + pose error |
| `outputs/replay_fidelity/videos/*.mp4` | state-sequence / action-replay 视频 |

### V2-B1.5 结论（标定结果）

| Demo | final-state | state-sequence | action-replay | 诊断 |
|------|-------------|----------------|---------------|------|
| success demo_0 | **success=true** | **success=true** | success 不稳定；`replay_state_error_mean` ~0.2–0.8 | **controller/action replay fidelity issue** |
| failed demo_4 | failed=true | failed=true | failed=true | **consistent_failure_as_expected** |

**核心结论：**

1. **success demo_0 的 HDF5 states 本身是成功的**（final-state + state-sequence 均 pass）。
2. **open-loop action replay 不能作为 full success validation 依据**：即使偶发 `success_flag=true`，state 轨迹仍与 HDF5 偏离（`replay_state_error_mean > 0.15`）。
3. **根因**：HDF5 录制时使用 flat `OSC_POSE`；当前 env 加载时 refactor 为 `CompositeController`；加上 closed-loop 逐步积分 vs 录制时闭环响应不一致。
4. **failed demo_4** 三类检查一致失败，HDF5 可靠编码失败轨迹。

### V2-B 可信边界与 refined rollout 解释

| 问题 | 答案 |
|------|------|
| 能否继续做 refined rollout？ | **可以**，但仅作 **residual improvement validation** |
| refined 结果含义 | 对比 sim 实时 nut-peg 残差 / `E_total_norm` 是否改善，**不是** full success validation |
| 不可信的部分 | open-loop action replay 的 `success_flag`（对 success demo 不稳定） |
| 可信的部分 | state-sequence 复现的末态残差；action replay 的逐步 `replay_state_error` / pose error 趋势 |

**controller 对比要点**（详见 report）：
- HDF5：`OSC_POSE`, `control_freq=20`, `action (T,7)`, range `[-1,1]`, `output_max=[0.05×3, 0.5×3]`
- Runtime：CompositeController 包装，`OSC_POSE` 参数一致，但积分路径不同

## V2-B2.5：Simulator-in-the-loop Local Refinement

V2-B1.5 确认 open-loop HDF5 action replay 存在 **controller_action_replay_fidelity_issue** 后，V2-B2.5 改为在当前 RoboSuite / MuJoCo 环境与 **当前 CompositeController** 下，用 **闭环 eef waypoint tracking** 做公平对比与局部搜索。

### 评价基线（不再使用 HDF5 原始 actions）

| Rollout | 说明 |
|---------|------|
| `original_waypoint_rollout` | failed demo_4 原始 `eef_pose` + `gripper_action`，`states[0]` 复位，闭环跟踪 waypoint |
| `refined_waypoint_rollout` | V2-A `best_theta` + `refined_waypoint_builder` 修正 waypoint，同一 `osc_action_converter` |
| `sim_in_loop_search` | 在 refined waypoint 基础上搜索 insertion 窗口 sim 参数（grid / random） |

### 搜索参数（demo_4 第一阶段）

`insert_z_offset`, `z_gain`, `insertion_steps`, `hold_steps`, `insertion_speed_scale`, `release_shift`, `pre_insert_pause` — 详见 `osc_action_converter.SEARCH_SPACE`。

### 运行

```bash
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_sim_in_loop_refinement.py

# 可选：grid 模式 / 限制评估次数
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_sim_in_loop_refinement.py --search-mode random --max-evals 120
```

### V2-B2.5 输出

| 路径 | 内容 |
|------|------|
| `outputs/sim_in_loop_refinement/sim_in_loop_refinement_report.json` | 完整报告 + best/top-10 |
| `outputs/sim_in_loop_refinement/sim_in_loop_refinement_summary.csv` | baseline vs best 汇总 |
| `outputs/sim_in_loop_refinement/top_candidates.csv` | Top-10 参数组合 |
| `outputs/sim_in_loop_refinement/residual_before_after.json` | original_waypoint vs best |
| `outputs/sim_in_loop_refinement/videos/original_waypoint_demo_4.mp4` | 当前 controller 原始 waypoint baseline |
| `outputs/sim_in_loop_refinement/videos/best_refined_demo_4.mp4` | 最优 sim-in-loop rollout |
| `outputs/sim_in_loop_refinement/videos/top_*.mp4` | Top-K 候选视频 |

### V2-B2.5 验收结果（demo_4，参考）

| 指标 | original_waypoint | refined (V2-A θ) | best sim-in-loop |
|------|-------------------|------------------|------------------|
| `success_flag` | false | **true** | **true** |
| `E_total_norm` | ~77 | ~1.82 | **~1.66** |
| `final_nut_peg_xy` | ~0.31 | ~0.009 | **~0.009** |
| `final_z_diff` | ~-0.020 | ~-0.010 | ~-0.019 |

**best sim 参数（参考）**：`insert_z_offset=-0.12`, `z_gain=0.7`, `insertion_steps=10`, `hold_steps=20`, `insertion_speed_scale=0.75`

### 解释边界

| 情况 | 含义 |
|------|------|
| `success_flag=true` 且 rollout 来自 waypoint+action（非 set final state） | **current-env refined success** — 可在当前环境下称为 refined 成功 |
| `success_flag=false` 但 E / 残差显著下降 | **residual_improvement_validation** |
| 使用 HDF5 原始 action open-loop replay | **不可**作为 full success baseline（V2-B1.5） |

后续扩展：transport_failed demo_0–3（V2-B3）。

## V2-B2.6：Repeatability + Ablation Study

V2-B2.5 在 **current-env** 下将 failed demo_4 改善为 `refined_success` 后，V2-B2.6 验证该结果是否**稳定**，并通过消融证明 **energy-guided refinement** 的贡献。

**范围说明**：本阶段仍是 **Nut Assembly 单任务、单失败模式（insertion_failed demo_4）** 验证；transport_failed demo_0–3 留待 **V2-B3**。

### 重复性验证

```bash
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_sim_in_loop_repeatability.py
```

- seeds: `[0, 1, 2, 3, 4]`
- max_evals: `[40, 80, 120]`
- 输出：`outputs/sim_in_loop_repeatability/`（report、summary、`videos/best_seed_*.mp4`）

### 消融实验

```bash
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_sim_in_loop_ablation.py --max-evals 40
```

| 方法 | 说明 |
|------|------|
| A `original_waypoint` | 原始 eef waypoint，无 theta / 无搜索 |
| B `v2a_theta_only` | V2-A best_theta refined waypoint，无 sim 搜索 |
| C `random_search_without_energy` | 随机搜索，按 random score 选 best（不用 energy） |
| D `energy_guided_search_full` | 随机搜索，按完整 `E_total_norm` 选 best |
| E–H | 去掉 z / xy+transport / yaw / smooth 分项的 energy 排序 |

输出：`outputs/sim_in_loop_ablation/`（report、summary、top_candidates、videos）

### V2-B2.6 消融结论（demo_4，40 evals，参考）

| 对比 | 结果 |
|------|------|
| full energy vs random | full energy **更优**（E 更低） |
| without_z vs full | **去掉 z 项显著损害** insertion 修正（E≈5.3，success=false） |
| without_xy_transport | demo_4 影响较小（符合 transport 已接近 peg 的预期） |
| without_smooth | 影响较小（符合 V0 审计） |
| V2-A theta vs original | theta **显著改善**（E 77→1.6，success=true） |

### V2-B2.6 重复性结论（demo_4，参考）

| 指标 | 值 |
|------|-----|
| 总体 refined success rate（15 runs） | **86.7%**（13/15） |
| `E_total_norm` 均值 ± 标准差 | 1.88 ± 0.63 |
| `final_z_diff` 距 target (-0.021) 均值 | **0.009 m** |
| max_evals=40 success rate | 60%（搜索预算不足时不稳定） |
| max_evals=80/120 success rate | 100% |

不稳定原因：`max_evals=40` 时随机采样覆盖不足，部分 seed 未找到有效 insertion 参数；增大 eval 预算后稳定达到 `refined_success`。

### 解释边界

| 术语 | 含义 |
|------|------|
| **current-env refined success** | 真实 sim rollout + waypoint/action，`success_flag=true`，非 set final state |
| **residual improvement validation** | success=false 但 E / 残差显著下降 |
| 不可使用 | proxy 结果冒充 sim；HDF5 raw action replay 作为 baseline |

## V2-B3：Transport_failed demo_0–3 Sim-in-loop Refinement

V2-B3 将 V2-B2.5 的 **current-controller 闭环 waypoint rollout + sim-in-loop search** 扩展到 `demo_failed.hdf5` 中 **transport_failed 的 demo_0–3**。

### 背景与目标

- **transport_failed demo_0–3 比 demo_4 insertion_failed 更难**：baseline `min_xy ≈ 0.28–0.32 m`，nut 从未接近 peg。
- 本阶段目标是 **扩展失败模式覆盖**，为 **V1-B residual model** 提供更均衡训练数据。
- **V1-A 当前模型不作为最终泛化模型**；需等 V2-B3 数据后再训练 V1-B。

### 新增文件

| 文件 | 作用 |
|------|------|
| `transport_waypoint_builder.py` | CEM `best_theta` + transport 搜索参数 → refined eef waypoint |
| `transport_sim_search.py` | random search、transport 导向打分、Level 1/2/3 验收 |
| `run_transport_refinement.py` | demo_0–3 批量 refinement 入口 |

### 搜索参数

`transport_xy_gain`, `transport_xy_offset_scale`, `pre_align_height`, `lift_height`, `approach_steps`, `transport_steps`, `hold_steps`, `gripper_close_shift`, `speed_scale` — 详见 `transport_sim_search.TRANSPORT_SEARCH_SPACE`。

### 评分（transport 导向）

```
score = 4*E_transport_norm + 4*E_xy_norm + 2*E_yaw_norm + 1*E_z_norm + 0.2*E_smooth_norm
```

### 验收等级（每条 demo）

| Level | 条件 |
|-------|------|
| **Level 1** | `final_nut_peg_xy` 相比 original_waypoint 降低 ≥ 50% |
| **Level 2** | `min_nut_peg_xy < 0.08 m` |
| **Level 3** | `min_nut_peg_xy < 0.03 m` 或 `success_flag=true` |

若未 success，仍输出 `improved_but_failed` 并记录真实残差改善。

### 运行

```bash
cd experiments/phys_demo_gen/nut_assembly

PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_transport_refinement.py

# 可选：限制评估次数 / demo
PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_transport_refinement.py \
  --max-evals 80 --seeds 0,1,2 --demo-keys demo_0,demo_1,demo_2,demo_3
```

### V2-B3 输出

| 路径 | 内容 |
|------|------|
| `outputs/transport_refinement/transport_refinement_report.json` | 完整报告 + per-demo 对比 |
| `outputs/transport_refinement/transport_refinement_summary.csv` | 汇总表 |
| `outputs/transport_refinement/top_candidates.csv` | 各 demo Top-10 候选 |
| `outputs/transport_refinement/per_demo_best.json` | 每条 demo 最优参数与 Level 验收 |
| `outputs/transport_refinement/videos/original_demo_*.mp4` | original waypoint baseline |
| `outputs/transport_refinement/videos/best_refined_demo_*.mp4` | best refined rollout |

### 约束

- 不伪造 `object_poses`
- 不以 set final state 冒充 success
- 不修改平台正式代码、不接前端、不使用 PINA

## V2-B4：Grasp_failed demo_2–3 Sim-in-loop Refinement

V2-B4 针对 V2-B3 暴露的 **grasp_failed** 问题：demo_2 / demo_3 中 nut 几乎未移动，transport waypoint refinement 无法补偿抓取阶段失败。

### 背景与目标

- **当前目标不是马上全任务 success**，而是让 nut 被稳定抓取、抬起并进入 transport 阶段。
- V2-B4 输出已接入 **V1-C grasp-aware residual model**（见 `v1_residual_model/README.md`）。
- 所有改善必须来自真实 MuJoCo / RoboSuite rollout（不伪造 object_poses、不 set final state）。

### 新增文件

| 文件 | 作用 |
|------|------|
| `grasp_waypoint_builder.py` | grasp 搜索参数 → refined eef waypoint + gripper 时序 |
| `grasp_sim_search.py` | random search、grasp 导向打分、Level G1/G2/G3 |
| `run_grasp_refinement.py` | demo_2–3 批量 grasp refinement 入口 |

### 搜索参数

`grasp_xy_offset_x/y`, `pre_grasp_height`, `approach_height`, `gripper_close_shift`, `gripper_hold_steps`, `lift_height`, `lift_steps`, `speed_scale`

### 评分（grasp 导向）

```
score = 4*grasp_distance_energy + 4*no_lift_penalty + 3*no_nut_motion_penalty
      + 2*E_transport_norm + 1*E_xy_norm + 0.2*E_smooth_norm
```

`grasp_success_proxy`：nut 抓取后位移、lift delta、eef-nut 距离三项中 **多数满足**。

### 验收等级

| Level | 条件 |
|-------|------|
| **G1** | `nut_displacement_after_grasp` 相比 original 提升 ≥ 50% |
| **G2** | `nut_lift_delta > 0.02` 或 `grasp_success_proxy=true` |
| **G3** | `min_nut_peg_xy` 降低 ≥ 30% 或进入 transport_improved |

Outcome：`refined_success` / `grasp_improved_but_failed` / `grasp_no_improvement`

### 运行

```bash
cd experiments/phys_demo_gen/nut_assembly

PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_grasp_refinement.py

PYTHONPATH=../../../integrations/CableThreadingMVP:$PYTHONPATH \
  MUJOCO_GL=egl python3 run_grasp_refinement.py \
  --max-evals 80 --seeds 0,1,2 --demo-keys demo_2,demo_3
```

### V2-B4 输出

| 路径 | 内容 |
|------|------|
| `outputs/grasp_refinement/grasp_refinement_report.json` | 完整报告 |
| `outputs/grasp_refinement/grasp_refinement_summary.csv` | 汇总 |
| `outputs/grasp_refinement/top_candidates.csv` | Top-10 候选 |
| `outputs/grasp_refinement/per_demo_best.json` | 最优参数与 Level 验收 |
| `outputs/grasp_refinement/videos/original_demo_*.mp4` | baseline |
| `outputs/grasp_refinement/videos/best_refined_demo_*.mp4` | best refined |

## V1-C：Grasp-aware Multi-failure Residual Model

详见 `v1_residual_model/README.md`。

- **定位**：在 V1-B 基础上接入 `outputs/grasp_refinement/`，扩展 grasp/lift 特征与监督头。
- **仍不是最终 PINN / PINA**；后续建议 **V2-B5**（demo_3 lift_failed 专项 refinement）或 **V1-D**（数据增强）。
- demo_2 `refined_success` 与 demo_3 `lift_failed` / `grasp_improved_but_failed` 均保留在训练集。

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model
python build_training_dataset.py --dataset-version v1c
python train_residual_model.py --model-version v1c --epochs 300
python evaluate_residual_model.py --model-version v1c
```

输出：`outputs/v1_residual_model_v1c/`（含 `v1b_vs_v1c_comparison.json`）

### V1-C.5 Group-split 泛化验证

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model
python evaluate_group_split.py
```

输出：`outputs/v1_residual_model_v1c_group_split/`（含 `v1c_random_vs_group_split_comparison.json`）

**注意**：V1-C 随机 split 指标可能偏乐观；V1-C.5 group split 才是泛化验证，用于决定是否进入 V2-B5 或 PINA/PINN formalization。

### V1-D PINN-style Physics-Informed Model

详见 `v1_residual_model/README_PINN.md`。

```bash
cd experiments/phys_demo_gen/nut_assembly/v1_residual_model
python train_pinn_residual_model.py --epochs 300
python evaluate_pinn_group_split.py
python compare_v1c_vs_pinn.py --ablation-epochs 150
```

输出：`outputs/v1_residual_model_pinn/`（**group split 为主要评估**）

## V1-A / V1-B：PyTorch Residual Energy Model 原型

详见 `v1_residual_model/README.md`。

## 说明

- 这是 **V0.5 显式物理能量模型**，不是已完成的 PINN。
- 暂不训练神经网络，暂不接入平台前端或 evaluation adapter。
- `E_total_norm` 量纲统一、可解释，适合作为 CEM 采样排序目标。
- V2-A 已验证 normalized energy 可驱动 proxy CEM 收敛；**V2-B 已在 RoboSuite 中验证 demo_4 refined rollout 可降低 `E_total_norm` 并改善 `final_z_diff`（`improved_but_failed`），但 open-loop replay 与录制 state 仍有偏差。**
