# NutAssembly MimicGen Worker (P5)

独立 MimicGen 数据生成 worker，用于 NutAssembly 任务。不修改 `CableThreadingMVP` vendored robosuite，不修改 MimicGen 核心源码。

## Conda 环境

```bash
conda env create -f runs/nut_assembly/debug/nut-assembly-mvp-environment.yml
conda activate nut-assembly-mvp
pip install -r integrations/NutAssemblyMimicGen/requirements.txt
pip install -e third_party/mimicgen
python $CONDA_PREFIX/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
python $CONDA_PREFIX/lib/python3.10/site-packages/robomimic/scripts/setup_macros.py
```

**重要**：MimicGen datagen 必须使用 `nut-assembly-mvp` 中的 upstream robosuite（site-packages），**禁止**在 PYTHONPATH 中包含 `integrations/CableThreadingMVP/robosuite`。平台后端会自动剥离 CableThreadingMVP 路径。

Fallback `robosuite_rollout` 在独立 subprocess 中使用 `cable-threading-mvp`，并显式设置 CableThreadingMVP PYTHONPATH。

## 环境检查

```bash
conda activate nut-assembly-mvp
python integrations/NutAssemblyMimicGen/env_check.py
```

检查项包括：`termcolor`、`robosuite.macros_private`、`robosuite.__file__`（非 CableThreadingMVP）、mimicgen、robomimic、mujoco。

输出：`runs/nut_assembly/debug/mimicgen_env_check.json`

## Source Demo

支持三种来源（优先级）：

1. 命令行/API `--source-demo-path`（绝对路径直接使用；相对路径基于项目根目录）
2. 环境变量 `NUT_ASSEMBLY_SOURCE_DEMO_PATH`
3. 默认依次尝试 `/mnt/data/demo.hdf5`、`<repo>/mnt/data/demo.hdf5`

若文件不存在，任务在 prepare_source 前即报 `source_demo_missing`。

## 独立运行

```bash
conda activate nut-assembly-mvp
python integrations/NutAssemblyMimicGen/run.py \
  --job-root runs/nut_assembly/jobs/na_test_p5 \
  --episodes 5 \
  --seed 0 \
  --generation-mode mimicgen_datagen \
  --render-video
```

## 生成模式

| `--generation-mode` | 行为 |
|---|---|
| `mimicgen_datagen`（默认） | 先 MimicGen prepare + generate；失败则 fallback `robosuite_rollout` |
| `robosuite_rollout` | 跳过 MimicGen，直接 partial_scripted rollout |

## 产物

```
runs/nut_assembly/jobs/<jobId>/
├── intermediate/prepared_source.hdf5
├── configs/mimicgen_nut_assembly_config.json
├── datasets/nut_assembly_generated.hdf5
├── manifest.json
├── results/generation_summary.json
└── videos/generate.mp4
```

## 平台接入

后端 `nut_assembly_service.py` 通过 subprocess 调用本 worker；MimicGen 路径使用 `nut-assembly-mvp` Python，rollout fallback 使用 `cable-threading-mvp` Python。
