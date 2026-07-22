# pi0 / openpi 接入说明

## 环境变量

| 变量 | 说明 |
|------|------|
| `PI0_RUNNER_ENABLED` | 设为 `1` / `true` 启用 pi0 训练 runner |
| `OPENPI_ROOT` | openpi 源码根目录（需包含 `src/openpi` 或 `scripts/train.py`） |
| `OPENPI_PYTHON` | 用于运行 openpi 的 Python 可执行文件路径（建议使用 openpi 虚拟环境） |
| `OPENPI_BASE_CONFIG` | openpi 训练配置名，如 `debug_pi05`（smoke）、`libero` 等 |
| `OPENPI_TRAIN_SCRIPT` | （可选）覆盖默认 `scripts/train.py` 路径 |
| `OPENPI_INFER_SCRIPT` | （可选）覆盖默认 `scripts/infer.py` 推理脚本 |
| `PI0_TRAIN_MODE` | 默认 `openpi`；仅测试可设 `shim`（已废弃，勿用于生产） |

### 配置示例

```bash
export PI0_RUNNER_ENABLED=1
export OPENPI_ROOT=/path/to/openpi
export OPENPI_PYTHON=/path/to/openpi/.venv/bin/python
export OPENPI_BASE_CONFIG=debug_pi05
```

CI / 无真实 openpi 安装时，可使用仓库内 mock fixture：

```bash
export PI0_RUNNER_ENABLED=1
export OPENPI_ROOT=/path/to/eai-idev2.1/backend/tests/fixtures/mock_openpi
export OPENPI_PYTHON=python3
export OPENPI_BASE_CONFIG=pi0_mock
```

## 支持的数据格式

| 格式 | 训练 | 评测 |
|------|------|------|
| 平台 HDF5（含图像 obs） | 自动转换为 `artifacts/lerobot_dataset/` index | 通过 `Pi0PolicyAdapter` + 环境 camera obs |
| LeRobot / HF manifest | 直接使用 | 需 openpi 推理环境 |
| 仅 low_dim HDF5 | **不支持** | **不支持** |

转换层：`backend/app/services/pi0_hdf5_converter.py`  
训练入口：`backend/integrations/pi0_runner/run_openpi_train.py`

## 支持的任务类型

- **线缆穿杆** (`cable_threading`)：训练 + 评测（需图像观测，如 `agentview_image`）
- 其他任务：若数据集无图像 obs，创建训练/评测时会返回可读 400 错误

## 训练产物

- `config/openpi_platform_config.yaml`
- `artifacts/lerobot_dataset/dataset_index.json`
- `checkpoints/pi0/checkpoints/model_final.pt`（或 openpi 原生 checkpoint 目录）
- `artifacts/model_manifest.json` 含 `modelTypeId`、`openpiEnvironment` 等追溯字段

## 评测

- Resolver：`evaluation_request_resolver._resolve_trained_policy_for_model_asset` → `policy=pi0`
- Runner：`run.py eval --policy pi0 --checkpoint <path>`
- Adapter：`integrations/.../pi0_lab/policy_runtime.py`

## 故障排查

1. **模型类型页显示 pi0「runner 未接入」**  
   检查 `OPENPI_ROOT`、`OPENPI_PYTHON`；点击「刷新」触发 probe。

2. **创建训练任务 400：openpi 环境未配置，无法训练 pi0**  
   设置 `PI0_RUNNER_ENABLED=1`、`OPENPI_ROOT`、`OPENPI_PYTHON` 并完成 openpi 环境探测。

3. **创建训练任务 400：pi0 需要 LeRobot 格式**  
   确认数据集 HDF5 含图像 obs；平台会自动生成 lerobot index。若 `OPENPI_BASE_CONFIG` 为 libero 等真实配置，需提供 LeRobot 数据或改用 `debug_pi05` smoke。

4. **评测失败：ACT/pi0 需要图像观测**  
   线缆穿杆评测环境默认启用 camera obs；若自定义环境未提供 `agentview_image` 等键，会返回 `OBS_VALIDATION_FAILED` 而非 500。

5. **查看日志**  
   `runtime_outputs/training/jobs/<trainJobId>/logs/train.log`  
   `runtime_outputs/cable_threading/jobs/<evalJobId>/logs/`
