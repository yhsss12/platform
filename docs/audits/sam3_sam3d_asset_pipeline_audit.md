# SAM3 + SAM3D Objects 资产流水线集成审计报告

> **审计日期**：2026-07-02  
> **审计范围**：只读，未修改任何业务代码  
> **上游项目**：`/home/ubuntu/project/asset3Drecons`  
> **目标平台**：`/home/ubuntu/project/eai-idev2.1`

---

## 0. 执行摘要

| 维度 | 结论 |
|------|------|
| SAM3 可调用性 | ✅ 已有本地脚本 `run_sam3_box_select.py`，conda 环境 `sam3`，权重离线可用 |
| SAM3D 可调用性 | ✅ 已有 `demo.py` + `notebook/inference.py`，conda 环境 `sam3d-objects`，checkpoint 在 `checkpoints/hf/` |
| 端到端串联脚本 | ❌ **不存在**；当前为手工两步（SAM3 输出目录 → 修改 `demo.py` 路径 → 运行） |
| 仿真资产导出 (MJCF/USD/URDF) | ❌ **不存在**；上游仅有 GLB 生成库函数（未封装 CLI） |
| 平台现有 SAM 集成 | ❌ **不存在**；可复用 CableThreading / OpenPI / Isaac Lab 的外部 subprocess 模式 |
| 前端导入入口 | ⚠️ `/workspace/resources/assets` 已有页面，**导入按钮为 mock** |

---

## 1. asset3Drecons 目录结构摘要

```
/home/ubuntu/project/asset3Drecons/
├── sam3/                              # Meta SAM3 官方仓库 + 本地定制脚本
│   ├── sam3/                          # Python 包（model / agent / eval / train）
│   ├── scripts/
│   │   ├── run_sam3_box_select.py     # ★ 实际使用的分割入口（本地定制）
│   │   ├── eval/                      # 官方评测脚本
│   │   └── ...
│   ├── assets/images/                 # 测试图片（如 tai4.jpg）
│   ├── checkpoints/sam3/            # 本地权重 sam3.pt + HF 缓存
│   ├── outputs/                       # SAM3 运行产物（非 git）
│   │   └── tai4_prompt_box/           # 已有样例输出
│   └── 指令.md                        # 手工命令备忘
│
└── sam-3d-objects/                    # Meta SAM 3D Objects 官方仓库
    ├── demo.py                        # ★ 最小重建入口（已改为读 SAM3 输出）
    ├── load_splat.py                  # 加载 splat.ply + Gradio/GIF 预览
    ├── show_ply.py                    # Open3D 点云预览
    ├── splat.ply                      # 样例 Gaussian Splat 输出 (~57MB)
    ├── notebook/
    │   ├── inference.py               # ★ 核心 Inference API + load_image/mask
    │   ├── mesh_alignment.py          # 3DB mesh 对齐工具（非通用导出）
    │   ├── images/                    # 官方 demo 样例
    │   ├── meshes/                    # 含 human_object/3Dfy_results/0.glb
    │   └── gaussians/
    ├── sam3d_objects/                 # 推理 pipeline 源码
    │   └── pipeline/
    │       ├── inference_pipeline.py          # mesh/glb 后处理逻辑
    │       ├── inference_pipeline_pointmap.py
    │       └── layout_post_optimization_utils.py  # export_transformed_mesh_glb()
    ├── checkpoints/hf/                # pipeline.yaml + 各 stage 权重
    ├── environments/default.yml       # conda 环境定义（env 名 sam3d-objects）
    └── 3D指令                           # 手工命令备忘
```

### 1.1 未发现的目录/文件（记录缺口）

审计中**未发现**以下用户提及的标准化资产目录或文件：

| 期望路径 | 状态 |
|----------|------|
| `asset/demo1/` | ❌ 不存在 |
| `pointcloud.ply`（标准命名） | ❌ 不存在（仅有根目录 `splat.ply`） |
| `camera.json` | ❌ 不存在（`mesh_alignment.py` 支持 focal_length json 参数，但未形成目录约定） |
| `depth/`、`normal/` 导出目录 | ❌ 不存在 |
| `export_*.py` 独立导出脚本 | ❌ 不存在 |
| `run_pipeline.py` 一体化脚本 | ❌ 不存在 |

---

## 2. SAM3 当前可调用命令

### 2.1 环境与路径

| 项 | 值 |
|----|-----|
| Conda 环境 | `sam3` → `/home/ubuntu/miniconda3/envs/sam3` |
| 工作目录 | `/home/ubuntu/project/asset3Drecons/sam3` |
| 入口脚本 | `scripts/run_sam3_box_select.py` |
| 默认权重 | `checkpoints/sam3/sam3.pt` |
| 默认 BPE | `sam3/assets/bpe_simple_vocab_16e6.txt.gz` |
| 离线模式 | 脚本顶部强制 `HF_HUB_OFFLINE=1` |

### 2.2 CLI 参数（来自 `argparse`）

| 参数 | 说明 |
|------|------|
| `--image` | 输入图片路径（必填） |
| `--out` | 输出目录（必填） |
| `--prompt` | 文本提示（可选） |
| `--box X0 Y0 X1 Y1` | 单个正向 xyxy 像素框 |
| `--pos-box X0 Y0 X1 Y1` | 正向框，可重复 |
| `--neg-box X0 Y0 X1 Y1` | 负向框，可重复 |
| `--text-only` | 仅文本，不进入框选 |
| `--no-neg-interactive` | 跳过负框交互 |
| `--checkpoint` | 权重路径 |
| `--bpe` | BPE 词表路径 |
| `--device` | `cuda` / `cpu` |
| `--confidence-threshold` | 默认 0.5 |

### 2.3 命令示例（来自 `指令.md` + 脚本默认值）

**交互式（matplotlib 框选，不适合 Web 直调）：**

```bash
cd /home/ubuntu/project/asset3Drecons/sam3 && conda activate sam3
python scripts/run_sam3_box_select.py \
  --image /home/ubuntu/project/asset3Drecons/sam3/assets/images/tai4.jpg \
  --out /home/ubuntu/project/asset3Drecons/sam3/outputs/tai4_interactive
```

**文本 + 正向框（适合平台 API 传参）：**

```bash
cd /home/ubuntu/project/asset3Drecons/sam3 && conda activate sam3
python scripts/run_sam3_box_select.py \
  --image /home/ubuntu/project/asset3Drecons/sam3/assets/images/tai4.jpg \
  --prompt "中间部分的机械结构的复合块件" \
  --pos-box 430 80 1320 900 \
  --out /home/ubuntu/project/asset3Drecons/sam3/outputs/tai4_prompt_box \
  --confidence-threshold 0.05
```

**纯文本：**

```bash
python scripts/run_sam3_box_select.py \
  --image ... --prompt "..." --text-only --no-neg-interactive --out ...
```

### 2.4 SAM3 输出结构（实测 `outputs/tai4_prompt_box/`）

```
{out}/
├── masks/
│   └── mask_{000..011}.png
├── cutouts/
│   └── cutout_{000..011}.png      # RGBA，alpha 为 mask
├── overlay.png                     # 可视化叠加
├── combined_mask.png
└── detections.json                 # 含 prompt、boxes、每 mask 的路径与 score
```

`detections.json` 中最高分 mask 示例：`index=9, score=0.91015625`。

### 2.5 SAM3 集成注意事项

1. **交互框选依赖 matplotlib**（`select_box_with_matplotlib`），Web 前端框选应通过 `--pos-box`/`--neg-box` API 传参，不能依赖 stdin/弹窗。
2. **多 mask 输出**：一次运行可产生 N 个 mask；下游需让用户选择 `mask_index`。
3. **SAM3D mask 命名约定不一致**：`notebook/inference.load_single_mask(folder, index=1)` 查找 `{index}.png`（如 `1.png`），而 SAM3 输出为 `cutout_001.png`；当前 `demo.py` 使用 `cutouts/` 目录 + `index=1`（依赖手工放置的 `1.png` 或需平台侧复制/重命名）。

---

## 3. SAM3D Objects 当前可调用命令

### 3.1 环境与路径

| 项 | 值 |
|----|-----|
| Conda 环境 | `sam3d-objects` → `/home/ubuntu/miniconda3/envs/sam3d-objects` |
| 工作目录 | `/home/ubuntu/project/asset3Drecons/sam-3d-objects` |
| 推理配置 | `checkpoints/hf/pipeline.yaml` |
| 核心模块 | `notebook/inference.py` → `class Inference` |

### 3.2 命令示例（来自 `3D指令` + `demo.py`）

**下载 checkpoint（一次性）：**

```bash
python -m pip install modelscope
modelscope download --model facebook/sam-3d-objects \
  --local_dir /home/ubuntu/project/asset3Drecons/sam-3d-objects/checkpoints/
# 权重实际使用 checkpoints/hf/
```

**重建（当前 demo.py 已硬编码 SAM3 输出路径）：**

```bash
cd /home/ubuntu/project/asset3Drecons/sam-3d-objects && conda activate sam3d-objects
python demo.py
# 输出: ./splat.ply (Gaussian Splat)
```

**预览：**

```bash
python load_splat.py splat.ply --interactive          # Gradio 3D 查看器
python load_splat.py splat.ply --render-gif out.gif   # 旋转 GIF
python show_ply.py splat.ply                          # Open3D 点云窗口
```

### 3.3 Inference API 行为（`notebook/inference.py`）

```python
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
output = inference(image, mask, seed=42)
# 当前 demo 默认参数:
#   with_mesh_postprocess=False
#   with_texture_baking=False
#   with_layout_postprocess=False
# 因此 demo.py 仅导出 output["gs"].save_ply("splat.ply")
```

若开启 mesh/glb 后处理（需改 `Inference.__call__` 参数或通过 `InferencePipeline` 直接调用）：

- `inference_pipeline.postprocess_slat_output()` 可生成 `outputs["glb"]`（trimesh GLB）
- `layout_post_optimization_utils.export_transformed_mesh_glb()` 为库函数，**无 CLI**

### 3.4 SAM3D 输出能力（库级别，未封装 CLI）

| 产物 | 来源 | CLI 状态 |
|------|------|----------|
| `splat.ply` / Gaussian | `output["gs"].save_ply()` | ✅ demo.py |
| `glb` | `postprocessing_utils.to_glb()` | ❌ 注释掉的 `glb.export()` |
| `mesh` | `outputs["mesh"]` | ❌ 无导出脚本 |
| `pointcloud` | `SceneVisualizer.object_pointcloud()` | ❌ 无导出脚本 |
| `depth` / `normal` / `camera.json` | MoGe / pointmap pipeline 内部 | ❌ 无标准导出 |
| MJCF / USD / URDF | — | ❌ **完全缺失** |

### 3.5 当前 handoff 链路（手工）

```
SAM3 --out--> outputs/.../masks/mask_XXX.png
                    cutouts/cutout_XXX.png
                           │
                           ▼ (需选择 mask_index + 路径对齐)
SAM3D demo.py --> splat.ply
```

**缺口**：无参数化 CLI 接受 `--image --mask --out-dir`；`demo.py` 路径写死在源码中。

---

## 4. eai-idev2.1 平台审计

### 4.1 技术栈

| 层 | 技术 | 位置 |
|----|------|------|
| 前端 | Next.js 15 + React 19 + TypeScript | `src/` |
| 后端 | FastAPI + SQLAlchemy + Pydantic | `backend/app/` |
| 数据库 | PostgreSQL | `.env` `DATABASE_URL` |
| 对象存储 | MinIO | 数据资产模块 |
| 队列（可选） | Redis + RQ | `USE_QUEUE=true` 时启用 |

### 4.2 前端路由与 workspace 模块

| 路径 | 页面 | 与资产流水线关系 |
|------|------|------------------|
| `/workspace/resources` | 资源总览 Hub | 可新增「3D 资产生成」卡片 |
| `/workspace/resources/assets` | 操作对象库 | ★ **首选入口**（`ResourceLibraryPage`，含 mock「导入」） |
| `/workspace/simulation/console` | 仿真 Run Console | 可复用 live 预览布局 |
| `/workspace/evaluation/report` | 评测报告 | 可参考结果页结构 |
| `/upload` | 数据上传 | 可参考 MinIO 直传流程 |

资源 Hub 配置：`src/lib/workspace/resourceHubSections.ts`  
操作对象页：`src/app/(platform)/workspace/resources/assets/page.tsx`

### 4.3 可复用前端组件

| 组件 | 路径 | 用途 |
|------|------|------|
| `ResourceLibraryPage` | `src/components/workspace/ResourceLibraryPage.tsx` | 资源列表 + 导入/新建按钮 |
| `ModelAssetFileUploadZone` | `src/components/workspace/resources/ModelAssetFileUploadZone.tsx` | 文件拖拽上传 |
| `SimulationRunConsoleLayout` | `src/components/workspace/simulation/SimulationRunConsoleLayout.tsx` | 任务状态 + 侧栏 + 预览区 |
| `SimulationViewport` | `src/components/workspace/simulation/SimulationViewport.tsx` | 仿真视口 |
| `RegistryResourceDetailDrawer` | `src/components/workspace/resources/RegistryResourceDetailDrawer.tsx` | 资源详情抽屉 |
| `EpisodeViewerLayout` / WebSocket 播放 | `src/features/asset-viewer/` | MCAP/HDF5 推流（3D mesh 预览需新组件） |

### 4.4 后端 API 框架

| 模块 | 路径 |
|------|------|
| 路由注册 | `backend/app/api/router.py` |
| Workspace 任务 | `backend/app/api/routes_workspace_jobs.py` → `/api/workspace/jobs` |
| 线缆穿杆（参考模式） | `backend/app/api/routes_workspace_cable_threading.py` |
| 资源注册 | `backend/app/api/routes_workspace_resources.py` |
| 数据资产上传 | `backend/app/api/routes_data_assets.py` → `/api/data-assets/upload-init` |
| 模型资产 | `backend/app/api/workspace/model_assets.py` |

### 4.5 任务 / Job 机制

**DB 模型**（`backend/app/models/workspace_job.py`）：

- `WorkspaceJob`：`job_id`, `job_type`, `task_type`, `status`, `runtime_path`, `metadata_json`
- `WorkspaceArtifact`：`artifact_type`, `file_path`, `url_path`

**运行时 job 模式**（以 CableThreading 为范本）：

```
backend/app/services/cable_threading_service.py
  WORKING_DIR = integrations/CableThreadingMVP
  OUTPUT_ROOT = runtime_outputs/cable_threading
  subprocess.Popen(..., cwd=WORKING_DIR)
  live/status.json + live/latest.jpg  → 前端轮询
```

**OpenPI 外部路径模式**（`.env`）：

```
OPENPI_ROOT=/home/ubuntu/project/openpi
OPENPI_PYTHON=/home/ubuntu/project/openpi/platform_python.sh
```

### 4.6 runtime_outputs 组织方式

```
runtime_outputs/
├── cable_threading/jobs/{ct_gen_|ct_eval_|ct_vid_}*/
├── dual_arm_cable/jobs/
├── isaac_lab/jobs/{isaac_gen_|isaac_eval_|...}*/
├── evaluations/jobs/{eval_*}/
├── training/jobs/{train_*}/
├── datasets/{imports,built}/
└── evaluation_reports/
```

统一评测层约定（见 `docs/evaluation-adapter-layer.md`）：

```
runtime_outputs/evaluations/{evalJobId}/
  status.json, logs/, results/, videos/, artifacts/, metadata/
```

### 4.7 异步 / 进度 / 通信机制

| 机制 | 使用情况 | 位置 |
|------|----------|------|
| `subprocess.Popen` | ✅ 主力（CableThreading、Training、Isaac Lab） | `*_service.py` |
| HTTP 轮询 `status.json` | ✅ 前端 Run Console | `runConsoleAdapters.ts` |
| `live/latest.jpg` 帧预览 | ✅ CableThreading generate/eval | `cable_threading_service.py` |
| WebSocket | ⚠️ 主要用于 MCAP/HDF5 播放、Agent 隧道 | `routes_label.py`, `job_ws` |
| Celery | ❌ 未使用 | — |
| FastAPI BackgroundTasks | ⚠️ 少量 legacy | `backend/tools/diagnostics/reproduce_job_flow.py` |
| Redis RQ 队列 | ⚠️ 可选 `USE_QUEUE=true` | `dispatcher.py` |

**推荐**：资产流水线 Phase 1 采用 **subprocess + status.json 轮询**（与 CableThreading 一致），Phase 2 可选 WebSocket 推送日志。

### 4.8 已有上传 / 下载 / 预览接口

| 能力 | API | 说明 |
|------|-----|------|
| MinIO 直传 | `POST /api/data-assets/upload-init` | 图片可先走此通道 |
| 导出下载 | `GET /api/data-assets/export/download?jobId=` | 批量导出 |
| Job 文件 | `GET /api/workspace/cable-threading/jobs/{id}/frame` | JPEG 帧 |
| Job 日志 | `GET /api/workspace/cable-threading/jobs/{id}/log` | 文本日志 |
| Job 视频 | `GET /api/workspace/cable-threading/jobs/{id}/video` | mp4 |
| 资源文件 | `resource_registry` YAML `files.*` 相对路径 | 仿真资产引用 |

### 4.9 平台内 SAM3 相关代码

**审计结论：eai-idev2.1 内无任何 `sam3` / `sam-3d-objects` / `asset3Drecons` 引用。**

---

## 5. 推荐前端页面位置

### 方案 A（推荐）：操作对象库下新增流水线页

```
/workspace/resources/assets          → 现有列表（发布后的 object 资源）
/workspace/resources/assets/pipeline → ★ 新建：SAM3→SAM3D 生成向导
/workspace/resources/assets/pipeline/[jobId] → 任务详情/预览/导出
```

**理由**：与「操作对象」资源类型直接关联；可替换现有 mock「导入」按钮跳转至 pipeline 页。

### 方案 B：独立 workspace 模块

```
/workspace/asset-pipeline            → 与 simulation/training 同级
```

**理由**：流程较长（上传→分割→重建→导出→发布），独立模块更清晰；需在 sidebar 增加入口。

### 方案 C：仿真控制台扩展

在 `/workspace/simulation/console` 增加「资产导入」Tab — 与仿真运行混排，**不推荐**（职责混淆）。

---

## 6. 推荐后端新增文件位置

```
eai-idev2.1/
├── .env                                    # 新增 SAM3_ROOT / SAM3_PYTHON / SAM3D_* 
├── integrations/Sam3dAssetPipeline/        # ★ 平台编排层（不修改上游 repo）
│   ├── run.py                              # 统一 CLI：segment | reconstruct | export | all
│   ├── segment_cli.py                      # 包装 run_sam3_box_select.py
│   ├── reconstruct_cli.py                  # 包装 notebook/inference
│   └── export_cli.py                       # Phase 2: mesh/glb/mjcf/usd
├── backend/app/services/
│   ├── sam3d_asset_service.py              # ★ job 调度、subprocess、status 写入
│   └── sam3d_asset_paths.py                # job 目录规范
├── backend/app/schemas/
│   └── sam3d_asset.py                      # Pydantic 请求/响应
├── backend/app/api/
│   └── routes_workspace_sam3d_assets.py    # ★ REST 路由
└── runtime_outputs/asset_pipeline/jobs/    # ★ 运行时产物
```

**router.py 注册建议**：

```python
api_router.include_router(
    routes_workspace_sam3d_assets.router,
    prefix="/workspace/asset-pipeline",
    tags=["workspace-asset-pipeline"],
)
```

---

## 7. 推荐 API 设计

### 7.1 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/workspace/asset-pipeline/jobs` | 创建 job + 上传图片（multipart 或 presigned URL） |
| `POST` | `/api/workspace/asset-pipeline/jobs/{job_id}/segment` | 触发 SAM3（prompt + boxes + mask_index） |
| `POST` | `/api/workspace/asset-pipeline/jobs/{job_id}/reconstruct` | 触发 SAM3D（需先 segment 完成） |
| `POST` | `/api/workspace/asset-pipeline/jobs/{job_id}/export` | 导出指定格式 |
| `POST` | `/api/workspace/asset-pipeline/jobs/{job_id}/publish` | 发布到 resource_registry object |
| `GET` | `/api/workspace/asset-pipeline/jobs/{job_id}` | job 状态 + 阶段进度 |
| `GET` | `/api/workspace/asset-pipeline/jobs/{job_id}/files/{rel_path}` | 下载/预览产物（overlay、ply、glb） |
| `GET` | `/api/workspace/asset-pipeline/jobs/{job_id}/live/frame` | 最新预览帧（JPEG） |
| `GET` | `/api/workspace/asset-pipeline/jobs` | 列表（分页） |

> 用户原始命名 `/api/workspace/assets/*` 亦可；建议使用 `asset-pipeline` 前缀避免与 `data-assets`、`model-assets` 混淆。

### 7.2 请求/响应示例

**创建 job：**

```json
POST /api/workspace/asset-pipeline/jobs
{ "name": "tai4_fixture", "projectId": "optional" }
→ { "jobId": "asset_job_20260702_143000_a1b2", "status": "created" }
```

**分割：**

```json
POST .../segment
{
  "prompt": "中间部分的机械结构的复合块件",
  "positiveBoxes": [[430, 80, 1320, 900]],
  "negativeBoxes": [],
  "confidenceThreshold": 0.05,
  "selectedMaskIndex": 9
}
→ { "status": "segmenting", "numMasks": 12 }
```

**重建：**

```json
POST .../reconstruct
{ "maskIndex": 9, "seed": 42, "withMeshPostprocess": true }
→ { "status": "reconstructing" }
```

**导出：**

```json
POST .../export
{ "formats": ["ply", "glb", "obj", "mjcf", "usd", "urdf"] }
→ { "status": "exporting", "artifacts": [...] }
```

---

## 8. 推荐数据结构

### 8.1 AssetJob（job.json + DB metadata_json）

```typescript
interface AssetJob {
  jobId: string;                    // asset_job_{timestamp}_{hex}
  status: 'created' | 'segmenting' | 'segmented' | 'reconstructing' |
          'reconstructed' | 'exporting' | 'completed' | 'failed';
  phase: string;
  progress: number;                 // 0..1
  message?: string;
  createdAt: string;
  updatedAt: string;
  input: AssetInputImage;
  segmentation?: SegmentationResult;
  reconstruction?: ReconstructionResult;
  exports: ExportArtifact[];
  error?: string;
}
```

### 8.2 AssetInputImage

```typescript
interface AssetInputImage {
  originalPath: string;             // input/image.png
  width: number;
  height: number;
  uploadedAt: string;
}
```

### 8.3 SegmentationResult

```typescript
interface SegmentationResult {
  prompt?: string;
  positiveBoxes: number[][];
  negativeBoxes: number[][];
  numMasks: number;
  selectedMaskIndex?: number;
  overlayPath: string;
  combinedMaskPath: string;
  detectionsPath: string;           // sam3/detections.json
  masks: Array<{ index: number; score: number; maskPath: string; cutoutPath: string }>;
}
```

### 8.4 ReconstructionResult

```typescript
interface ReconstructionResult {
  maskIndex: number;
  seed: number;
  gsPlyPath?: string;
  pointcloudPlyPath?: string;
  meshObjPath?: string;
  glbPath?: string;
  previewGifPath?: string;
  previewMp4Path?: string;
}
```

### 8.5 ExportArtifact

```typescript
interface ExportArtifact {
  format: 'ply' | 'obj' | 'glb' | 'mjcf' | 'usd' | 'urdf';
  path: string;
  sizeBytes: number;
  simBackend?: 'mujoco' | 'isaacsim' | 'generic';
  status: 'pending' | 'ready' | 'failed';
}
```

### 8.6 DB 映射

| 字段 | 值 |
|------|-----|
| `WorkspaceJob.job_type` | `asset_pipeline` |
| `WorkspaceJob.task_type` | `sam3d_asset_import` |
| `WorkspaceJob.runtime_path` | `runtime_outputs/asset_pipeline/jobs/{job_id}` |

---

## 9. 推荐 runtime_outputs 目录结构

```
runtime_outputs/asset_pipeline/jobs/asset_job_{YYYYMMDD}_{HHMMSS}_{hex}/
├── job.json                         # AssetJob 主 manifest
├── input/
│   └── image.png
├── sam3/
│   ├── masks/
│   │   └── mask_{idx}.png
│   ├── cutouts/
│   │   └── cutout_{idx}.png
│   ├── overlay.png
│   ├── combined_mask.png
│   └── detections.json
├── sam3d/
│   ├── gs.ply                       # Gaussian splat
│   ├── pointcloud.ply               # Phase 2
│   ├── mesh.obj                     # Phase 2
│   ├── preview.gif
│   └── preview.mp4
├── exports/
│   ├── model.ply
│   ├── model.obj
│   ├── model.glb
│   ├── model.xml                    # MJCF
│   ├── model.usd
│   └── model.urdf
├── live/
│   ├── status.json                  # 轮询用（对齐 cable_threading）
│   └── latest.jpg                   # 分割 overlay 或重建预览
├── logs/
│   ├── segment.log
│   ├── reconstruct.log
│   └── export.log
└── metadata/
    └── source_request.json
```

**发布后**（resource_registry 引用）：

```
runtime_outputs/asset_pipeline/published/{asset_slug}/
  manifest.yaml
  preview.png
  mujoco/object.xml + meshes/
  isaacsim/object.usd
```

---

## 10. 前端页面设计建议

```
┌─────────────────────────────────────────────────────────────────┐
│  3D 资产生成流水线                          [jobId] [状态徽章]   │
├──────────────────────────────┬──────────────────────────────────┤
│ ① 上传图片区                  │  分割结果预览区                    │
│   [拖拽 / 选择文件]           │   overlay.png + mask 缩略图列表   │
│                              │   点击选择 mask_index              │
│ ② 文本提示输入区              ├──────────────────────────────────┤
│   [prompt textarea]          │  3D 预览区                         │
│                              │   Phase1: overlay / cutout         │
│ ③ 框选区（Canvas）            │   Phase2: splat GIF / three.js GLB │
│   正向框 / 负向框             │   Phase3: 仿真视口占位              │
│   [添加框] [清除]             │                                    │
│                              ├──────────────────────────────────┤
│ ④ 操作按钮                    │  任务日志 / 进度区                 │
│   [运行分割]                  │   phase + progress bar           │
│   [运行重建]                  │   segment.log / reconstruct.log  │
│   [导出 ▼ format multi]      │                                    │
│   [发布到对象库]              │                                    │
└──────────────────────────────┴──────────────────────────────────┘
```

| 区域 | 组件建议 | 复用来源 |
|------|----------|----------|
| 上传 | `ModelAssetFileUploadZone` 或 MinIO presigned | `ImportPretrainedModelModal` |
| 框选 | 新建 `MaskBoxCanvas`（Canvas2D） | — |
| 分割预览 | `<img>` + mask 列表 | CableThreading live frame 模式 |
| 3D 预览 | `@react-three/fiber` 或 iframe Gradio | `load_splat.py --interactive` 可作 fallback |
| 进度 | `SimulationRunConsoleLayout` | 现有 Run Console |
| 导出 | Checkbox group + 下载链接 | `WorkspaceArtifact` 列表 |

---

## 11. 环境变量建议（Phase 1 配置，尚未添加）

```bash
ASSET3DRECONS_ROOT=/home/ubuntu/project/asset3Drecons
SAM3_ROOT=/home/ubuntu/project/asset3Drecons/sam3
SAM3_PYTHON=/home/ubuntu/miniconda3/envs/sam3/bin/python
SAM3D_OBJECTS_ROOT=/home/ubuntu/project/asset3Drecons/sam-3d-objects
SAM3D_OBJECTS_PYTHON=/home/ubuntu/miniconda3/envs/sam3d-objects/bin/python
SAM3D_PIPELINE_ENABLED=true
SAM3D_OUTPUT_ROOT=runtime_outputs/asset_pipeline/jobs
```

---

## 12. 下一步代码修改清单（本轮不执行）

### Phase 1 — 最小可跑通（分割 → splat.ply → 预览）

| # | 任务 | 文件/位置 |
|---|------|-----------|
| 1 | 新增 `.env` 路径配置 | `.env`, `backend/app/core/config.py` |
| 2 | 新增 job 路径工具 | `backend/app/services/sam3d_asset_paths.py` |
| 3 | 新增 platform CLI 包装（不修改上游） | `integrations/Sam3dAssetPipeline/run.py` |
| 4 | 包装 SAM3：非交互 `--pos-box` 模式 | `integrations/.../segment_cli.py` |
| 5 | 包装 SAM3D：参数化 `--image --mask --out` | `integrations/.../reconstruct_cli.py` |
| 6 | 解决 mask 命名：`cutout_{idx}.png` → SAM3D 可读 | 包装层复制为 `{idx}.png` 或传绝对 mask 路径 |
| 7 | 新增 backend service（subprocess + status.json） | `backend/app/services/sam3d_asset_service.py` |
| 8 | 新增 API 路由 | `backend/app/api/routes_workspace_sam3d_assets.py` |
| 9 | 注册路由 | `backend/app/api/router.py` |
| 10 | 新增 Pydantic schemas | `backend/app/schemas/sam3d_asset.py` |
| 11 | 前端 API client | `src/lib/api/sam3dAssetPipelineClient.ts` |
| 12 | 前端 pipeline 页面 | `src/app/(platform)/workspace/resources/assets/pipeline/page.tsx` |
| 13 | 替换 mock 导入按钮 | `ResourceLibraryPage.tsx` → 链接到 pipeline |
| 14 | WorkspaceJob 索引同步 | `workspace_job_service.py` 识别 `asset_job_*` |

### Phase 2 — mesh / glb / 点云

| # | 任务 |
|---|------|
| 15 | 扩展 `reconstruct_cli.py` 开启 `with_mesh_postprocess=True` |
| 16 | 导出 `glb`、`mesh.obj`、`pointcloud.ply` |
| 17 | 前端 three.js GLB 预览 |

### Phase 3 — 仿真格式导出

| # | 任务 |
|---|------|
| 18 | 新建 `export_cli.py`：mesh → MJCF（trimesh + 模板） |
| 19 | mesh → USD（pxr.Usd 或 Isaac 工具链） |
| 20 | mesh → URDF（trimesh + 惯性估计） |
| 21 | 发布到 `configs/resources/objects/` |
| 22 | `upsert_resource_definition(source=imported)` |

### Phase 4 — 体验增强

| # | 任务 |
|---|------|
| 23 | WebSocket 日志流（可选） |
| 24 | MinIO 归档大文件 |
| 25 | 与评测 scene/task 关联（required_assets.objects） |

---

## 13. 风险与阻塞项

| 风险 | 说明 | 缓解 |
|------|------|------|
| 双 conda 环境 | SAM3 与 SAM3D 不能同进程 import | 分阶段 subprocess，指定各自 `SAM3_PYTHON` / `SAM3D_OBJECTS_PYTHON` |
| matplotlib 交互 | 无法用于 Web | API 只支持 `--pos-box`/`--neg-box` 参数 |
| mask 命名不一致 | SAM3 `cutout_001.png` vs SAM3D `{idx}.png` | 包装层统一 |
| demo.py 硬编码 | 不便平台调用 | 新建 platform CLI，不直接改 demo.py（或仅加 argparse） |
| MJCF/USD/URDF 缺失 | 需自研 export_cli | Phase 3 单独交付 |
| splat.ply 体积大 (~57MB) | 网络传输慢 | 预览用 GIF/缩略图；PLY 走下载 |
| GPU 独占 | 并发 job 争抢 L20 | job 队列 + 单 GPU 锁 |

---

## 14. 参考：现有可对照实现

| 能力 | 参考文件 |
|------|----------|
| subprocess 异步 job | `backend/app/services/cable_threading_service.py` |
| generate-async API | `backend/app/api/routes_workspace_cable_threading.py` |
| live 预览轮询 | `src/lib/workspace/runConsoleAdapters.ts` |
| 外部 ML 路径配置 | `.env` `OPENPI_ROOT` / `ISAACLAB_ROOT` |
| 资源发布 | `backend/app/services/resource_definition_service.py` → `upsert_resource_definition` |
| 对象 registry YAML | `configs/resources/objects/object_cable_v1.yaml` |

---

*本文档由只读审计生成，不包含任何源码变更。*
