# 采集端 Agent 变更记录（Changelog）

本文件记录采集端 Agent 的版本变更要点，便于平台侧安装/升级时确认能力差异。

## 0.1.29

- 删除：`SCRIPT_DELETE_DATA` / `DELETE /api/agent/script/data` 支持 `allow_job_workspace`，可删除与平台作业编号一致的 **四位数字作业目录**（含空目录），便于平台删除作业时同步清理采集端磁盘。

## 0.1.28

- 预览：`EAI_AGENT_PREVIEW_FLUID=1` 时默认 **30 FPS**、JPEG 质量 **55**、长边最大 **640px**（可分别用 `EAI_AGENT_PREVIEW_MAX_FPS` / `EAI_AGENT_PREVIEW_JPEG_QUALITY` / `EAI_AGENT_PREVIEW_MAX_EDGE` 覆盖），用降低清晰度换更流畅的 MJPEG 预览。

## 0.1.27

- 优化：相机预览解码/JPEG 改在工作线程执行（DDS 回调仅拷贝并入队），减轻 `rclpy` spin 占用；预览帧率默认上限 **12 FPS**（可通过环境变量调节），缓解与 bag 录制等资源争用。

## 0.1.26

- 修复：停止采集（`COLLECT_STOP` / `POST /api/agent/collect/stop`）现在会终止**整个采集脚本进程组**，避免脚本内部拉起子进程后仅父进程退出导致“看似停止但仍在采集”的问题。
- 规范：默认 `agent_id` 推荐使用网卡 MAC（小写冒号），并与平台侧识别逻辑保持一致。

## 0.1.25

- 新增：同步到 MinIO 支持通过平台 WebSocket 隧道下发 `DATA_SYNC`（采集端执行上传并返回 `minio_path`）。

