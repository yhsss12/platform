# 运行数据与代码路径审计报告

审计日期：2026-07-17  
审计范围：仓库根目录、前后端主流程、Worker、集成与实验脚本、Docker Compose、Git 跟踪/忽略状态。  
本次仅调查，不迁移文件，不修改现有路径行为。

## 1. 结论摘要

当前仓库已经形成了以 `runtime_outputs/` 为中心的运行目录约定，但还没有真正统一的“数据根目录”。运行数据、正式资产、缓存、日志、测试报告和第三方数据集分散在仓库内多个位置，且不少代码通过 `PROJECT_ROOT / "runtime_outputs"` 直接绑定代码仓库位置。

主要结论：

1. `runtime_assets/` 是当前最大的运行资产目录，约 3.4 GB；其中单个 MimicGen HDF5 文件约 3.4 GB。
2. `runtime_outputs/` 当前仅约 3 MB，但它是训练、评测、数据生成、回放和资产流水线的核心路径；历史清理脚本表明它曾达到数 GB。
3. `runtime_outputs/`、`runtime_assets/`、`backend/runtime_outputs/`、`mnt/data/` 当前未被根 `.gitignore` 覆盖，存在误提交风险。
4. 数据资产目录 `backend/data/assets/` 已支持 `DATA_ASSETS_ROOT` 环境变量，是现有代码中最接近可迁移设计的部分。
5. `runtime_outputs` 至少出现在 152 个非 `node_modules`/`dist` 文件中；其中后端 Python 文件 38 个，脚本、实验及集成代码文件 28 个。其余主要为测试和文档。
6. 多个数据库字段会保存本地路径或运行路径。迁移时必须兼容历史相对路径和绝对路径，不能只移动磁盘目录。
7. 删除任务的安全检查明确要求目标位于仓库内的 `runtime_outputs` 下。改变根目录时必须同步重写这些边界检查，否则删除会失败，或产生安全风险。
8. Docker Compose 当前只为 PostgreSQL、MinIO、Redis 使用命名卷，没有为后端统一挂载运行数据根目录。
9. `integrations/` 中同时存在源码和生成模型，例如 `DualArmCableManipulation/model_final.pth` 约 504 MB；不能把整个 `integrations/` 当作纯代码或纯数据整体处理。
10. 仓库当前已有大量用户未提交改动和删除记录。本次报告没有触碰这些内容，后续迁移必须与现有工作区变更隔离。

## 2. 当前数据分布

以下体积为审计时工作区的实际值，会随任务运行变化。

| 路径 | 约占用 | 当前用途 | Git 状态/风险 | 初步归类 |
|---|---:|---|---|---|
| `runtime_assets/` | 3.4 GB | MimicGen 源数据、核心数据、PINN 模型 | 未被根规则忽略，均显示为未跟踪文件 | 长期资产/模型 |
| `runtime_outputs/` | 3.0 MB | 训练、评测、生成、数据集构建、Isaac Lab、资产流水线 | 未被根规则忽略；有未跟踪文件 | 任务运行产物 |
| `mimicgen-main/datasets/` | 450 MB | MimicGen 官方源数据集 | 由嵌套仓库 `.gitignore` 忽略 | 外部依赖数据/长期资产 |
| `integrations/` | 881 MB | 多个集成工程、源码、结果及模型 | 当前大量内容未纳入外层 Git；内部规则忽略模型 | 混合目录，需拆分审计 |
| `external/` | 109 MB | Isaac Lab 等外部依赖 | 根 `.gitignore` 已忽略 `external/IsaacLab-main/` | 外部代码依赖 |
| `node_modules/` | 640 MB | 前端依赖 | 已忽略 | 可重建缓存 |
| `mnt/data/` | 3.0 MB | HDF5 样例、审计脚本与结果 | 未忽略，显示为未跟踪 | 混合临时数据 |
| `backend/runtime_outputs/` | 16 KB | 一份评测任务状态和元数据 | 未忽略，显示为未跟踪 | 错位运行产物 |
| `logs/` | 128 KB | 服务日志与 telemetry | 已忽略 | 可清理日志 |
| `backend/data/` | 4 KB | 数据资产根、设备配置 | 仅 `backend/data/assets/` 已忽略；设备配置未统一 | 平台持久数据 |
| `artifacts/` | 当前为空 | 验收截图、报告、测试产物 | 历史上部分文件被 Git 跟踪，当前工作区中大量为删除状态 | 测试/报告产物 |
| `debug_outputs/` | 当前为空 | 调试输出 | 未忽略 | 临时产物 |
| `mimicgen_generated/` | 当前为空 | MimicGen 生成数据 | 未忽略 | 生成数据 |
| `dist/` | 2.4 MB | 离线发布构建 | 未在现有根规则中明确忽略 | 发布产物 |

### 2.1 最大文件

| 文件 | 约大小 | 判断 |
|---|---:|---|
| `runtime_assets/mimicgen/nut_assembly/core/nut_assembly_d0.hdf5` | 3.4 GB | 大型数据资产，应移出代码仓库 |
| `integrations/DualArmCableManipulation/model_final.pth` | 504 MB | 生成模型，当前混在集成源码目录中 |
| `mimicgen-main/datasets/source/pick_place.hdf5` | 88 MB | 外部官方源数据 |
| `mimicgen-main/datasets/source/kitchen.hdf5` | 73 MB | 外部官方源数据 |
| `mimicgen-main/datasets/source/coffee_preparation.hdf5` | 72 MB | 外部官方源数据 |
| `runtime_assets/mimicgen/nut_assembly/source/nut_assembly.hdf5` | 34 MB | 长期源数据资产 |
| `runtime_assets/models/pinn/nut_assembly_pinn_v1/model.pt` | 约 1 MB | 正式模型资产 |
| `runtime_outputs/isaac_lab/seeds/stack_cube_seed.hdf5` | 3 MB | 种子数据；是否属于代码发行资源需要单独决定 |

## 3. 已存在的路径配置能力

### 3.1 数据资产根目录

`backend/app/db/data_assets_session.py` 已提供：

```text
DATA_ASSETS_ROOT=<自定义目录>
```

未配置时默认指向：

```text
backend/data/assets
```

这是可复用的迁移模式，但目前它只覆盖导入的数据资产及 MinIO 本地缓存，不能覆盖训练、评测、模型和日志。

### 3.2 HDF5 数据目录

标注、HDF5 服务和任务配置服务支持：

```text
HDF5_DATA_DIR=/tmp/hdf5_data
```

默认值位于 `/tmp`，不适合作为需要长期保存的数据根目录。该配置与 `DATA_ASSETS_ROOT` 之间目前没有统一关系。

### 3.3 部分流水线独立配置

`backend/app/core/config.py` 中已有：

- `ISAACLAB_OUTPUT_ROOT`，默认 `runtime_outputs/isaac_lab/jobs`；
- `ISAACLAB_STACK_CUBE_DEFAULT_SEED`，默认位于 `runtime_outputs/isaac_lab/seeds/`；
- `SAM3D_OUTPUT_ROOT`，默认 `runtime_outputs/asset_pipeline/jobs`；
- `TELEMETRY_FILE_LOG_DIR`，默认 `logs/telemetry`。

这些配置证明部分模块已有外置能力，但默认值和解析方式仍以仓库根目录为基准，而且不同模块各自定义，没有共同上级根目录。

### 3.4 外部模型缓存和工程路径

SAM3/SAM3D 相关配置包含多项机器绝对路径，例如：

```text
/home/ubuntu/project/asset3Drecons
/home/ubuntu/.cache/torch
/home/ubuntu/.cache/huggingface
/home/ubuntu/miniconda3/envs/...
```

它们属于外部运行环境和模型缓存，应纳入部署配置审计，但不应与平台生成任务目录机械合并。

## 4. `runtime_outputs/` 的主要使用者

### 4.1 后端主流程（高优先级）

以下模块直接以仓库根目录构造运行路径：

- `workspace_job_service.py`：Workspace 任务索引、重建、删除和运行目录安全检查；
- `training_service.py`、`training_job_sync_service.py`：训练任务、checkpoint、运行目录同步；
- `evaluation/job_paths.py`、`evaluation_service.py`：统一评测目录；
- `cable_threading_service.py`：线缆穿引任务；
- `dual_arm_cable_service.py`：双臂线缆任务；
- `isaacsim_franka_pick_place_service.py`、`isaaclab_franka_stack_cube_service.py`：数据生成任务；
- `workspace_dataset_import_service.py`、`workspace_dataset_build_service.py`：数据集导入与构建；
- `model_asset_import_service.py`、`workspace_model_asset_service.py`：模型资产导入与解析；
- `artifact_upload_service.py`：扫描本地任务产物并上传 MinIO；
- `workspace_reindex_service.py`、`workspace_dataset_backfill_service.py`：从磁盘回填数据库索引；
- `report_export`：评测报告输出；
- `sam3d_asset_paths.py`：资产生成流水线；
- `isaac_lab/job_paths.py`、`isaac_dataset_service.py`：Isaac Lab 任务和数据集登记。

这些模块必须优先改用统一路径入口，否则不同服务会在新旧目录同时产生数据。

### 4.2 Worker 和远程训练（高优先级）

- `artifact_upload_worker.py` 明确扫描 `runtime_outputs`；
- `training_remote_runner.py` 在远程机器拼接 `{remote_root}/runtime_outputs/training/jobs/...`；
- RQ Worker 与 API 进程必须看到相同数据根目录；
- 远程训练的 `remote_root` 语义目前更接近“远程代码根目录”，迁移后需要区分远程代码根与远程数据根。

### 4.3 集成和实验代码（中优先级）

NutAssembly、CableThreading、PhyGen/PINN 相关脚本大量直接引用：

```text
runtime_outputs/nut_assembly/...
runtime_outputs/cable_threading/...
runtime_outputs/phygen_.../...
runtime_assets/mimicgen/...
runtime_assets/models/pinn/...
```

其中有些是默认 CLI 参数，有些是模块级常量，还有些绑定到具体历史任务 ID。迁移时应优先让脚本接受环境变量或命令行参数，历史任务样例则保留为文档示例或 fixture。

### 4.4 测试和文档（低优先级但数量多）

`artifacts` 和 `logs` 的文本引用数量很高，主要来自验收脚本和文档，不都代表真实运行路径。迁移时应区分：

- 会实际创建文件的测试代码；
- 只验证 UI 不泄漏内部路径的字符串；
- 文档示例；
- 历史报告。

不能用全局文本替换完成迁移。

## 5. 路径持久化与兼容风险

PostgreSQL 模型中至少存在以下路径字段：

| 模型 | 字段 | 风险 |
|---|---|---|
| `WorkspaceJob` | `runtime_path` | 核心任务运行目录；已有相对/绝对路径兼容逻辑 |
| Workspace 资产记录 | `file_path`、`url_path` | 可能指向运行目录或发布资产 |
| `ArtifactStorageObject` | `local_path` | MinIO 上传前后的本地路径 |
| `DataAsset` | `file_path`、`minio_path` | 本地文件与对象存储 URI 并存 |
| 转换/同步相关模型 | `input_file_path`、`output_path` | 可能保存目录或外部 repo ID |
| `LabelTaskAsset` | `dataset_path` | 可保存多个服务器路径 |
| `Job` | `mcap_path` | 采集结果路径 |
| `Device` | `profile_path`、`script_path`、`stop_script_path` | 设备侧路径，不应误迁移为平台数据路径 |
| `ResourceDefinition` | `manifest_path` | 当前可能相对代码仓库根目录 |

风险点：

1. 当前部分服务把相对路径解释为相对于 `PROJECT_ROOT`。
2. 部分记录会写入 `job_root.relative_to(PROJECT_ROOT)`，迁移到仓库外后会变成绝对路径。
3. checkpoint resolver 会同时尝试 `PROJECT_ROOT / path` 和原始路径。
4. 重建索引逻辑依赖扫描固定目录，迁移后若只改写入端，会导致历史任务“消失”。
5. MinIO URI 不应参与本地磁盘路径迁移。
6. Agent/设备脚本路径属于采集端或服务器环境路径，需要保持原语义。

## 6. 安全边界风险

以下代码不仅拼路径，还以固定目录作为安全边界：

- Workspace 任务删除拒绝删除 `runtime_outputs` 根目录，并要求目标位于其下；
- 训练任务清理使用允许根目录列表；
- 评测任务路径通过 `resolve()` 后校验仍在评测根目录下；
- 数据资产文件浏览要求目标位于 `DATA_ASSETS_ROOT` 下；
- 模型资产导入目录删除也检查固定的 `runtime_outputs/model_assets/imported` 前缀。

未来统一路径模块必须同时提供：

- 规范化后的根目录；
- `is_relative_to`/安全包含判断；
- 禁止删除数据总根、运行总根和资产总根的保护；
- 对符号链接逃逸的防护；
- 明确区分平台路径、Agent 路径和 MinIO URI。

## 7. Git 与仓库迁移风险

### 7.1 当前忽略规则不足

根 `.gitignore` 已忽略：

- `node_modules/`、`.next/`；
- `logs/`；
- `backend/data/assets/`；
- `external/IsaacLab-main/`；
- 常规 Python/测试缓存。

但未明确忽略：

- `runtime_outputs/`；
- `runtime_assets/`；
- `backend/runtime_outputs/`；
- `mnt/data/`；
- `debug_outputs/`；
- `mimicgen_generated/`；
- 根 `artifacts/`（而且其中存在历史已跟踪文件）；
- `dist/`。

这意味着即使文件目前未提交，也可能在后续执行 `git add` 时误入仓库。

### 7.2 混合源码目录

`integrations/` 当前包含：

- 应属于代码的 Python、Shell、YAML 和 vendored 工程；
- `model_final.pth` 等模型；
- `results/`、日志、`egg-info`、`__pycache__` 等运行/构建产物。

因此后续应按文件类型和用途清理集成目录，不能移动整个目录。

### 7.3 嵌套仓库

`mimicgen-main/` 自带 `.git/`，是一个嵌套 Git 仓库；其 `datasets/` 由内部 `.gitignore` 忽略。外层仓库迁移时需要明确它是：

- Git submodule；
- 安装时下载的外部依赖；
- 还是随部署包复制的 vendored 目录。

当前外层 `git submodule status` 没有登记结果，因此不能把它视为已规范配置的 submodule。

## 8. Docker 与服务部署现状

`docker-compose.postgres-minio.yml` 使用：

- `postgres-data` → `/var/lib/postgresql/data`；
- `minio-data` → `/data`。

`docker-compose.yml` 使用：

- `redis-data` → `/data`。

当前未看到统一的平台本地数据目录挂载。将来如果 API、CPU Worker、GPU Worker、IO Worker 分容器运行，它们必须挂载同一宿主数据根，并在容器内使用一致路径；否则任务创建、产物扫描、上传和删除会看到不同文件系统。

PostgreSQL 和 MinIO 的命名卷本身已经与代码仓库分离，但备份和整机迁移仍需单独纳入数据迁移说明。

## 9. 建议的迁移优先级

本报告不实施迁移，建议后续按以下顺序推进：

### P0：先建立统一抽象

1. 新增唯一的 `EAI_DATA_ROOT`。
2. 建立集中路径模块，导出 `runs_root`、`assets_root`、`cache_root`、`logs_root` 等。
3. 为现有 `DATA_ASSETS_ROOT`、`ISAACLAB_OUTPUT_ROOT`、`SAM3D_OUTPUT_ROOT` 保留覆盖能力。
4. 补齐路径解析与安全边界单元测试。

### P1：迁移后端核心读写链路

1. Workspace Job、训练、评测、数据生成；
2. 数据集导入/构建、模型导入；
3. 任务重建、backfill、artifact upload；
4. 删除和清理安全检查；
5. API 与各类 Worker 共享配置。

### P2：迁移外部执行和实验流程

1. 远程训练区分代码根与数据根；
2. Isaac Lab、SAM3D、CableThreading、DualArm、NutAssembly；
3. PhyGen/PINN 脚本默认参数；
4. 移出 `integrations/` 中的模型和结果文件。

### P3：处理历史数据和仓库卫生

1. 编写只复制、不删除的迁移工具；
2. 迁移数据库路径或增加旧路径解析兼容层；
3. 更新 Docker volume、环境模板、`.gitignore` 和文档；
4. 验证完成后，再清理旧路径和兼容符号链接。

## 10. 下一步建议

下一步不宜立即搬文件。建议先设计并评审“统一路径模块及环境变量契约”，确定：

1. 新数据根目录内部结构；
2. 默认开发路径和生产路径；
3. 数据库保存相对路径时采用的命名空间，例如 `runs/...`、`assets/...`；
4. 旧 `runtime_outputs/...` 路径的兼容期限；
5. 哪些种子数据和小型 fixture 必须继续随代码发布。

确认这些契约后，再做第一批小范围代码改造，建议从 `workspace_runtime_paths.py`、训练和评测路径开始，并保持旧目录可读。

路径契约现已形成独立文档，后续实现以 [data-path-contract.md](../data-path-contract.md) 为准。

## 附录：审计口径

- 目录体积使用 `du -sh`，仅代表 2026-07-17 当前工作区。
- 文本引用使用 `rg`，排除了 `node_modules/`、`dist/`、`external/` 和嵌套仓库的 `.git/`；引用数量包含代码、测试和文档，因此用于衡量影响面，不等同于需要修改的代码文件数。
- Git 状态使用 `git status --short --untracked-files=all`、`git check-ignore -v` 和 `git ls-files`。
- 大文件扫描覆盖当前主要数据目录，并额外检查了常见模型/数据扩展名。
- 本次没有读取或记录 `.env` 中的具体秘密值。
