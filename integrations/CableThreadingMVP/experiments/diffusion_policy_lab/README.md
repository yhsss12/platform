# 单臂线缆 Diffusion Policy 实验目录

独立于平台前端/后端的 DP 实验沙箱，使用 CableThreadingMVP 生成的 robomimic 兼容 HDF5（含双相机图像）。

## 目录结构

```text
experiments/diffusion_policy_lab/
  dp_lab/              # 核心库（dataset / model / trainer）
  scripts/             # CLI 入口
  configs/             # 默认超参
  outputs/             # 本地训练输出（gitignore）
```

## 环境

使用现有 conda 环境即可（需 PyTorch + h5py + torchvision）：

```bash
conda activate cable-threading-mvp
cd integrations/CableThreadingMVP/experiments/diffusion_policy_lab
```

可选额外依赖见 `requirements-lab.txt`（当前实现为纯 PyTorch DDPM，不强制安装 diffusers）。

## 快速开始

### 1. 检查数据集

```bash
python scripts/inspect_dataset.py \
  --dataset ../../../../runs/cable_threading/jobs/ct_gen_20260615_102019_8f58/datasets/dataset.hdf5
```

### 2. Smoke test（轻量 debug 模式，约十几秒）

```bash
bash scripts/smoke_test.sh
```

`--debug` 会自动使用 `tiny_cnn` 视觉编码器、64 训练窗口、2 epoch，不下载 ResNet 权重，不影响平台。

### 3. 正式训练（ResNet18，建议 GPU）

```bash
python scripts/train.py \
  --dataset /path/to/dataset.hdf5 \
  --out-dir outputs/cable_threading_dp_v1 \
  --num-epochs 100 \
  --batch-size 16 \
  --image-size 128 \
  --vision-encoder resnet18 \
  --device cuda
```

### 4. 推理 smoke

```bash
python scripts/infer_smoke.py \
  --checkpoint outputs/smoke_run/checkpoints/model_final.pt \
  --dataset ../../../../runs/cable_threading/jobs/ct_gen_20260615_102019_8f58/datasets/dataset.hdf5
```

### 5. 正式 GPU 训练

```bash
bash scripts/run_gpu_train.sh
# 日志：outputs/gpu_run_v1_train.log
```

### 6. 仿真 rollout 验证

```bash
MUJOCO_GL=egl python scripts/rollout_eval.py \
  --checkpoint outputs/gpu_run_v1/checkpoints/model_final.pt \
  --episodes 5 \
  --device cuda \
  --out-dir outputs/rollout_gpu_v1
```

`rollout_eval.py` 在单臂线缆仿真里加载 DP checkpoint，接口与 `RobomimicPolicyAdapter` 一致（`reset` / `act`），按 chunk 执行动作。

## 输出物

训练完成后在 `--out-dir` 下生成：

```text
checkpoints/model_final.pt    # 平台后续可对接的 DP checkpoint
config/train_config.json
logs/train.log
```

`model_final.pt` 内含 `state_dict`、`shape_meta`、normalizer 统计与超参，便于后续接到 `training_service`。

## 数据要求

- robomimic 格式 HDF5，`data/demo_*/obs/` 含：
  - `agentview_image` (T,H,W,3) uint8
  - `robot0_eye_in_hand_image` (T,H,W,3) uint8
  - `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`
- `data/demo_*/actions` shape (T, 7)
- 可选 `mask/train` 划分

## 与平台的关系

本目录**不修改** `backend/`、`src/` 或现有 `train_bc.py`。验证通过后，再将 `scripts/train.py` 的参数映射接入 `training_service._build_train_command()`。
