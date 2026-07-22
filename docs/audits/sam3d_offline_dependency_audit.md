# SAM3D 离线依赖审计

审计对象：`/home/ubuntu/project/asset3Drecons/sam-3d-objects`

## 1. DINO 加载

文件：`sam3d_objects/model/backbone/dit/embedder/dino.py`

```python
self.backbone = torch.hub.load(
    repo_or_dir=repo_or_dir,  # 默认 facebookresearch/dinov2
    model=dino_model,           # 默认 dinov2_vitb14
    source=source,              # 默认 github
)
```

配置来源：

- `checkpoints/hf/ss_generator.yaml` — 2 处 `Dino`，`dino_model: dinov2_vitl14_reg`（无 source/repo）
- `checkpoints/hf/slat_generator.yaml` — 同上

默认 `source=github` 会尝试联网拉取。

**离线方案**：job-local yaml patch 为 `source: local` + `repo_or_dir: <本地 hub 目录>`

服务器已有缓存：

`/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main/hubconf.py`

## 2. MoGe 深度模型

`checkpoints/hf/pipeline.yaml`：

```yaml
depth_model:
  model:
    _target_: moge.model.v1.MoGeModel.from_pretrained
    pretrained_model_name_or_path: Ruicheng/moge-vitl
```

`from_pretrained` 逻辑：

- 本地路径存在 → 直接 `torch.load`
- 否则 → `hf_hub_download`（联网）

服务器已有 HF cache：

`/home/ubuntu/.cache/huggingface/hub/models--Ruicheng--moge-vitl/snapshots/.../model.pt`

**离线方案**：

- job-local pipeline.yaml 将 `pretrained_model_name_or_path` 改为 `model.pt` 绝对路径
- 环境变量 `HF_HUB_OFFLINE=1` 等禁止联网

## 3. pipeline.yaml 引用链

```
pipeline.yaml
  → ss_generator.yaml / slat_generator.yaml (Dino)
  → ss_*.ckpt / slat_*.ckpt (本地 checkpoints/hf/)
  → depth_model MoGe
```

`Inference(config_file)` 设置 `workspace_dir = dirname(config_file)`，相对路径相对于 job-local config 目录。

## 4. 平台实现

| 模块 | 作用 |
|------|------|
| `offline_dependency_check.py` | 启动前检查 DINO/MoGe/checkpoint |
| `sam3d_config_patch.py` | 生成 job-local yaml |
| `reconstruct_cli.py` | offline-mode + fail fast（约 30s 内） |

## 5. 配置项（backend config）

```
SAM3D_OFFLINE_MODE=true
SAM3D_DINOV2_REPO=/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main
SAM3D_DINOV2_MODEL=dinov2_vitl14_reg
SAM3D_MOGE_MODEL_PATH=/home/ubuntu/.cache/huggingface/hub/models--Ruicheng--moge-vitl
SAM3D_TORCH_HOME=/home/ubuntu/.cache/torch
SAM3D_HF_HOME=/home/ubuntu/.cache/huggingface
SAM3D_RECONSTRUCT_TIMEOUT_SECONDS=1800
```

## 6. 卡住 job 根因（asset_job_20260702_110728_b0e7）

- 重建子进程在 `Loading DINO model ... source: github` 阶段尝试联网
- 环境 Network unreachable，长时间重试
- 轮询脚本约 25 分钟后因后端 Connection refused 退出（可能 OOM 或进程重启）

## 7. 结论

- **不改上游 yaml**；使用 job-local patch
- 依赖齐全时可离线重建；缺失时 **30 秒内 failed**，不再卡 20+ 分钟
