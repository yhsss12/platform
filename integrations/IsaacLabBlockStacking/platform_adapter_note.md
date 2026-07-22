# Platform Adapter Note — Isaac Lab Franka Stack Cube

## 任务概述

该任务是 Isaac Lab Stack Cube 任务模板，整理自 Isaac Lab / Isaac Lab Mimic 官方示例。

## 平台任务ID

`task_isaaclab_franka_stack_cube_v1`

## 运行依赖

该任务依赖完整 **Isaac Lab / Isaac Sim** 环境（含 `isaaclab.sh`、Isaac Sim 运行时及 `isaaclab` / `isaaclab_mimic` 包）。任务包内的 Python 文件为配置与脚本整理，不能脱离 Isaac Lab 工程独立运行。

## 专家策略来源

专家策略类型为 **Isaac Lab Mimic + seed demonstration** 生成链路，不是独立 scripted expert 或预训练 checkpoint。

典型流程：

1. `record_seed` — 录制 seed demonstration（HDF5）
2. `annotate_seed` — 标注子任务边界
3. `generate_mimic` — 基于 Mimic 扩增 demonstration

## 平台接入方式

平台采用 **adapter** 方式接入：

- 任务包落盘于 `integrations/IsaacLabBlockStacking/`
- Registry 中 `run_entry` / `generate_entry` / `check_entry` 指向 `run/platform_run.py`
- 后端 `isaaclab_franka_stack_cube_service` 通过 subprocess 调用统一入口，不在 FastAPI 进程内 import Isaac Lab

## 调用约定

平台调用时应执行 `run/platform_run.py`，根据 `--mode` 分发到对应流程，**不要**直接 import 任务包内全部源码。

推荐命令示例：

```bash
python run/platform_run.py --mode check
python run/platform_run.py --mode generate_mimic --output-dir <job_output_dir> --num-demos 10 --seed 0
```

## 数据中心登记

生成 demonstration 数据后，输出目录（HDF5 / 可选 Zarr）需由平台 job worker 写入 `dataset_manifest.json`，并在数据中心登记为数据集。

## HDF5 → Zarr 转换

若生成 HDF5，可通过 `run/convert_isaac_hdf5_to_zarr.py` 转换为 Diffusion Policy 等框架可用的 Zarr 格式：

```bash
python run/convert_isaac_hdf5_to_zarr.py --input <generated.hdf5> --output <output.zarr>
```

## 相关 Registry 字段

| 字段 | 值 |
|------|-----|
| sim_backend | isaac_lab |
| simulator | isaac_sim |
| expert_source | Isaac Lab Mimic seed demonstration |
| requires_external_runtime | true |
| requires_seed_demo | true |
