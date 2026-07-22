# 代码与运行数据路径契约

本文定义平台后续模块迁移共同遵循的目录和环境变量。路径迁移期间保持旧布局可用，不在启用配置时自动搬运或删除文件。

## 1. 顶层结构

推荐代码目录和数据目录互为同级目录：

```text
eai-platform/
├── code/
│   └── eai-idev2.1/       # Git 仓库
└── data/                   # EAI_DATA_ROOT，不纳入 Git
    ├── runs/               # 每次任务的完整运行现场
    ├── assets/             # 长期保存和发布的数据集、模型及任务资产
    ├── cache/              # 可重新下载或生成的缓存
    ├── logs/               # 平台服务日志
    ├── state/              # 本地持久状态；PostgreSQL/MinIO 本体仍单独管理
    └── tmp/                # 上传、转换等短期中间文件
```

唯一的上级配置为：

```bash
EAI_DATA_ROOT=/absolute/path/to/eai-platform/data
```

允许使用相对路径；相对路径统一以代码仓库根目录为基准。生产部署建议始终使用绝对路径。

## 2. 目录职责

```text
runs/
├── collection/
├── data_generation/
├── training/jobs/
├── evaluations/jobs/
├── replay/
└── asset_pipeline/jobs/

assets/
├── datasets/
├── models/
├── task_assets/
└── external/              # 外部官方数据等不可由本平台重建的资产
```

- `runs/` 中保存任务配置、日志、指标、中间 checkpoint、视频和结果。
- `assets/` 中保存被平台正式登记、可复用或需要长期保存的内容。
- `cache/` 和 `tmp/` 必须允许独立清理，不得存放唯一副本。
- 数据库中的 MinIO URI、Agent 文件路径和平台本地路径属于不同命名空间，不得混用。

## 3. 兼容规则

未设置 `EAI_DATA_ROOT` 时，统一路径模块保持当前行为：

| 逻辑目录 | 旧路径 |
|---|---|
| runs | `runtime_outputs/` |
| assets | `runtime_assets/` |
| logs | `logs/` |
| state | `backend/data/` |
| tmp | `temp/` |

设置 `EAI_DATA_ROOT` 后使用新布局：

| 逻辑目录 | 新路径 |
|---|---|
| runs | `$EAI_DATA_ROOT/runs/` |
| assets | `$EAI_DATA_ROOT/assets/` |
| logs | `$EAI_DATA_ROOT/logs/` |
| state | `$EAI_DATA_ROOT/state/` |
| cache | `$EAI_DATA_ROOT/cache/` |
| tmp | `$EAI_DATA_ROOT/tmp/` |

现阶段设置变量本身不会让尚未迁移的业务模块改写到新路径。每个模块切换到 `app.core.platform_paths` 后才受该变量控制。

## 4. 代码使用规则

后端模块应从 `app.core.platform_paths` 获取路径，不再自行计算 `PROJECT_ROOT / "runtime_outputs"`：

```python
from app.core.platform_paths import platform_paths

job_root = platform_paths.training_jobs / job_id
```

规则如下：

1. 导入路径模块不得创建目录；业务真正开始写入时才创建需要的子目录。
2. API、Worker 和命令行任务必须使用相同的 `EAI_DATA_ROOT`。
3. 删除前使用规范化路径边界检查，禁止用字符串前缀判断目录归属。
4. 不把 `data_root` 自身作为普通任务目录删除。
5. 数据库新记录优先保存带逻辑命名空间的相对路径；兼容层继续解析历史 `runtime_outputs/...` 路径。
6. 不自动修改数据库历史路径，不在服务启动时自动搬运数据。

## 5. 模块迁移顺序

1. Workspace 通用路径、训练和评测。
2. 数据生成、数据集导入/构建、模型资产。
3. 产物扫描上传、索引重建和清理逻辑。
4. Isaac Lab、SAM3D、CableThreading、DualArm、NutAssembly。
5. PhyGen/PINN 及其余实验脚本。
6. Docker volume、迁移工具、历史路径兼容与最终清理。

## 6. 当前接入进度

- 已接入：Workspace 公共任务根目录、训练/评测任务目录解析、Workspace 运行目录删除边界。
- 已兼容：数据库中历史 `runtime_outputs/...` 相对路径和历史绝对路径。
- 已接入：训练任务写入根目录、训练状态同步、训练 checkpoint 本地解析。
- 已接入：训练模型资产扫描、产物上传扫描、远程训练数据根和训练引用检查。
- 已接入：Workspace HDF5 导入与标准数据集构建；新写入位于 `runs/datasets/`，旧目录继续合并读取。
- 已接入：外部模型资产导入；新模型文件位于 `assets/models/imported/`，删除安全边界兼容新旧目录。
- 已接入：Isaac Sim / Isaac Lab 数据生成、Isaac CLI 作业、SAM3D 作业根目录。
- 已接入：Isaac Lab 数据集注册表；新注册表位于 `assets/datasets/isaac_lab/`，旧注册表继续合并读取。
- 已接入：Isaac Lab 默认 Stack Cube seed；未显式配置时读取 `runs/isaac_lab/seeds/stack_cube_seed.hdf5`，旧路径保留回退。
- 远程训练：`TRAIN_NODE_L20_DATA_ROOT` 为空时继续使用远端 `<WORKDIR>/runtime_outputs`。
- 尚未接入：其余实验脚本、第三方集成内部默认路径、Docker volume 与最终历史数据搬迁。
