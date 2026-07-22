# CableThreadingMVP

从 dloBench 抽取的最小可运行 CableThreading 包（Minimum Viable Product）。

`results/` 是本地 CLI 的可再生输出目录，已从版本控制排除；平台任务产物仍应写入
`EAI_DATA_ROOT` 下的统一运行目录。

## 项目概述

CableThreadingMVP 是一个完整的线缆穿杆任务基准测试包，支持：

- **任务环境**：MuJoCo 仿真中的线缆穿杆任务（将线缆一端固定，拖拽另一端穿过两根桌面柱子之间的间隙）
- **支持机器人**：UR5e、Panda（Franka，推荐，成功率最高）
- **多种线缆模型**：rmb（刚体链）、flex（软体）、composite_cable（复合线缆）
- **专家策略**：10-15 阶段的脚本化 oracle，自动完成穿杆任务
- **数据采集**：批量生成专家数据集（NPZ/HDF5/LeRobot 格式）
- **策略评估**：加载 checkpoint 或使用专家/随机策略进行多轮 rollout
- **视频录制**：离屏渲染 MP4 视频

## 安装

### 方式一：conda 环境（推荐）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate cable-threading-mvp

# 安装项目
pip install -e .

# 安装 / 校验 robomimic BC 训练依赖（PYTHONNOUSERSITE=1 场景）
bash scripts/install_training_deps.sh
```

### 方式二：pip 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .

# 训练依赖（psutil / tqdm / sympy / matplotlib 等）
bash scripts/install_training_deps.sh
```

### 方式三：仅核心依赖

```bash
pip install numpy mujoco h5py imageio imageio-ffmpeg
```

## 快速开始

```bash
cd CableThreadingMVP

# 1. 冒烟测试
python run.py env-test

# 2. Panda + flex 线缆采集（100% 成功率）
python run.py expert --episodes 10 --robot Panda --cable-model flex

# 3. UR5e + rmb 线缆采集（~10% 成功率）
python run.py expert --episodes 20 --robot UR5e --cable-model rmb --horizon 250

# 4. 评估专家策略
python run.py eval --episodes 20 --policy scripted

# 5. 录制视频
python run.py video --episodes 1
```

## 任务性能

### 专家策略成功率

| 机器人 | 线缆模型 | horizon | 成功率 | 平均步数 | 说明 |
|--------|----------|---------|--------|----------|------|
| **Panda** | **flex** | 600 | **100%** (10/10) | ~150 | 最优配置，软体线缆物理特性好 |
| **Panda** | **composite_cable** | 400 | **100%** (10/10) | ~200 | 复合线缆稳定性高 |
| **Panda** | rmb | 400 | **50%** (5/10) | ~300 | 刚体链穿杆检测较难 |
| **UR5e** | flex | 400 | **30%** (3/10) | ~350 | UR5e 工作空间较小 |
| **UR5e** | rmb | 250 | **10%** (1/10) | ~250 | 默认配置，成功率较低 |

### 成功条件

`final_success` 要求**所有条件同时满足**：

1. **threaded_final**：线缆穿过杆间间隙（线段交叉检测）
2. **endpoint_region_final**：线缆末端到达目标区域（goal_tolerance=0.04m）
3. **endpoint_past_gap_final**：末端已越过杆缝
4. **straightened_final**：线缆直线度 ≥ 0.88（端点距离/弧长）
5. **settled_on_table_final**：90% 线缆点在桌面上
6. **anchor_stable_final**：线缆起点未漂移（anchor_tolerance=0.02m）
7. **peak_height_excess ≤ 0**：最高点不超杆顶

### 难度分级

| 难度 | anchor_angle_range | root_yaw_range | joint_noise_scale |
|------|-------------------|----------------|-------------------|
| easy | 0.3 | 0.3 | 0.01 |
| medium | 0.6 | 0.6 | 0.03 |
| hard | 1.0 | 1.0 | 0.05 |

## 输出格式

| 格式 | 命令 | 说明 |
|------|------|------|
| NPZ  | `expert --out` | 压缩 numpy，low-dim 数据 |
| HDF5 | `expert --hdf5-out` | robomimic 兼容，含图像（agentview + wrist） |
| LeRobot | `expert --lerobot-out` | LeRobot v3.0 格式 |
| CSV  | `eval --out` | 逐 episode 指标 |
| JSON | 自动生成 | `*.results.json`、`*.failures.json`、`*.manifest.json` |
| MP4  | `video --video-out` | 专家策略视频 |

## 完整示例

```bash
# Panda + flex 线缆，采集带图像的 HDF5
conda run -n robosuite python run.py expert --episodes 20 --robot Panda --cable-model flex \
    --out datasets/panda_flex.npz \
    --hdf5-out datasets/panda_flex.hdf5

# 评估随机策略
conda run -n robosuite python run.py eval --episodes 50 --policy random --out results/random.csv

# 评估学习策略（需要 robomimic）
conda run -n robosuite python run.py eval --episodes 20 --policy robomimic --checkpoint model.pth
```

## 目录结构

```
CableThreadingMVP/
├── run.py                    # 统一入口（env-test/expert/eval/video）
├── setup.py                  # 包安装配置
├── README.md                 # 本文档
├── environment.yml           # conda 环境定义
├── requirements.txt          # pip 依赖
├── configs/
│   └── cable_threading_task.yaml  # 任务配置参考
├── robosuite/                # robosuite 仿真框架（自包含，无外部依赖）
│   ├── controllers/          # 机器人控制器（OSC、IK、joint 等）
│   ├── environments/         # 仿真环境
│   ├── models/               # MJCF 模型
│   │   └── assets/robots/    # 机器人模型（仅保留 Panda + UR5e 网格）
│   ├── robots/               # 机器人运行时类
│   └── utils/                # 工具函数
├── examples/
│   ├── cable_threading/      # 专家策略、数据采集、工具函数
│   └── dlo/                  # 共享环境创建、任务注册
├── datasets/                 # 数据集输出目录
└── results/                  # 评估结果输出目录
```

**注意**：本文件夹完全自包含，可直接复制到其他机器使用，无需额外依赖或符号链接。机器人模型仅保留 Panda 和 UR5e（项目实际使用的两个），文件夹大小约 243M。

## 基准要求满足状况

| # | 要求 | 状态 | 说明 |
|---|------|------|------|
| 1 | 场景与模型 | ✅ | MJCF/XML、UR5e/Panda、4 种线缆模型、相机配置 |
| 2 | 环境接口 | ✅ | reset/step/render/close + is_success/get_observation |
| 3 | 数据生成 | ✅ | 专家策略、批量采集、成功/失败记录 |
| 4 | 数据记录 | ✅ | NPZ/HDF5(含图像)/LeRobot/MP4、manifest.json |
| 5 | 数据处理 | ✅ | ACT/robomimic/LeRobot 格式、字段定义文档化 |
| 6 | 评测接口 | ✅ | task_name 路由、checkpoint、CSV + JSON + video |
| 7 | 成功判定 | ✅ | 8 条件、可计算可复现 |
| 8 | 任务配置 | ✅ | YAML 配置参考文件 |

## 技术细节

### 专家策略阶段

**attachment 模式（默认）**：
1. approach_above_end (30步)
2. descend_to_grasp (24步)
3. attach (10步)
4. lift_clear (30步)
5. backoff_clearance (40步，条件跳过)
6. align_to_gap_entry (40步)
7. enter_gap (30步)
8. lower_after_gap (50步)
9. pull_through (90步)
10. lay_down_endpoint (60步)
11. table_straighten (60步)
12. press_to_table (40步)
13. release (12步)
14. retreat (30步)
15. settle_wait (60步)

**Panda+flex 简化模式**：跳过 lay_down_endpoint、table_straighten、press_to_table、backoff_clearance，避免线缆反弹穿过柱子。

### 观测字段

| Key | Shape | 说明 |
|-----|-------|------|
| robot0_eef_pos | (3,) | 末端执行器位置 |
| robot0_gripper_qpos | (2,) | 夹爪关节位置 |
| cable_end_pos | (3,) | 线缆末端位置 |
| pole_points | (6,) | 两根柱子位置 |
| endpoint_goal_pos | (3,) | 目标位置 |
| attachment_state | (1,) | 是否附着 |
| agentview_image | (H,W,3) | 主视角图像（HDF5/LeRobot） |
| robot0_eye_in_hand_image | (H,W,3) | 腕部相机图像（HDF5/LeRobot） |

### 控制参数

- **控制频率**：20 Hz
- **动作维度**：7（6 arm DOF + 1 gripper）
- **动作缩放**：0.04m（delta-EEF）
- **动作裁剪**：[-1.0, 1.0]
