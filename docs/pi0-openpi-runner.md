# pi0 / openpi 训练 Runner 配置说明

平台通过外部 [openpi](https://github.com/Physical-Intelligence/openpi) 项目运行 pi0 真实训练，**不会**将 openpi 源码复制进本仓库。

## 环境变量

| 变量 | 说明 |
|------|------|
| `PI0_RUNNER_ENABLED` | 设为 `true` 启用 pi0 runner 探测 |
| `OPENPI_ROOT` | openpi 项目根目录（含 `src/openpi`） |
| `OPENPI_PYTHON` | 执行 openpi 的 Python 可执行文件，或 `uv run python` 等命令 |
| `OPENPI_BASE_CONFIG` | （可选）openpi 预置 TrainConfig 名称，如 `pi05_libero` |
| `OPENPI_TRAIN_SCRIPT` | （可选）覆盖默认训练入口脚本路径 |
| `PI0_USE_PLATFORM_SHIM` | 设为 `1` 时使用平台内置 shim 训练（smoke/无 GPU），**生产请关闭** |

探测成功时：

- `probe_training_capabilities().supportedTrainingBackends` 包含 `pi0`
- 模型类型 API 中 `pi0.trainingReady=true`
- 新建训练任务可选择 pi0

探测失败时：

- `pi0.trainingReady=false`
- `disabledReason` 示例：`未检测到可用 openpi 环境，请配置 OPENPI_ROOT 与 OPENPI_PYTHON`

## 安装 openpi（简述）

1. 克隆 openpi 到独立目录，按官方文档创建 Python/uv 环境并安装依赖。
2. 在本项目 `backend/.env` 中配置上述变量。
3. 重启后端，检查 `GET /api/workspace/training/capabilities` 是否包含 `pi0`。

## 当前支持的数据格式

- **第一版**：平台标准 **HDF5 模仿学习数据集**（`data/demo_*/obs/*` + `actions`）
- 必需：**图像观测键**（如 `agentview_image`）、**actions**
- 建议：manifest 提供 `taskType` / `taskDescription` 作为 language prompt
- 训练前会生成 `artifacts/pi0_dataset_index.json` 作为 openpi 侧数据索引

不支持：

- 纯 low_dim 无图像数据集
- 直接读取 HuggingFace LeRobot 原始格式（需先转换为平台 HDF5）

## 训练配置与任务参数

- **结构参数**（context_window、action_horizon、vision_encoder 等）来自「资源 - 模型类型」`structure_config`
- **训练参数**（Epochs、Batch Size、Learning Rate、Seed）在新建训练任务弹窗覆盖 `training_defaults`
- 平台生成 `config/openpi_platform_config.yaml`，由 `backend/integrations/pi0_runner/run_openpi_train.py` 调用 openpi

## Checkpoint 保存策略

与 ACT 类似，支持：

- Final（`model_final.pt`）
- Best Loss（若 runner 输出 `model_best.pt`）
- 按 epoch/step 中间 checkpoint（若 runner 输出）

Final 仅在 checkpoint 文件真实存在且任务 completed 后 `status=ready`，方可评测。

## Loss / Progress

- Runner 写入 `artifacts/metrics.jsonl`，字段：`epoch`、`step`、`trainLoss`、`learningRate`（有则写，无则省略）
- 日志第一行打印完整训练命令
- 训练完成后同步 `workspace_jobs`、`training_metric_summary`、`model_assets`

## 运行时目录结构

```
runtime_outputs/training/jobs/train_YYYYMMDD_HHMMSS_xxxx/
├── status.json
├── config/
│   ├── train_config.json
│   └── openpi_platform_config.yaml
├── logs/train.log
├── artifacts/
│   ├── metrics.jsonl
│   ├── pi0_dataset_index.json
│   └── model_assets_registry.json
└── checkpoints/pi0/checkpoints/model_final.pt
```

## 当前仍不支持

- 真实 openpi 全量预训练（仅 fine-tune / 平台 shim smoke）
- 非 HDF5 数据源直读
- pi0 在线推理服务（仅训练链路）
- 自定义 openpi TrainConfig 的可视化编辑（需在 openpi 项目或「模型类型」结构配置中间接体现）
