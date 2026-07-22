# 数据资产导入 V1 手工 / curl 验证

前置：后端可访问、已登录拿到 `ACCESS_TOKEN`、存在 `PROJECT_ID`、MinIO 已配置且项目 bucket 可写。

## 1) 单文件 .mcap（curl）

```bash
export API=http://127.0.0.1:8000
export TOKEN='...'
export PROJECT_ID='...'

curl -sS -X POST "$API/api/data-assets/import" \
  -H "Authorization: Bearer $TOKEN" \
  -F "project=$PROJECT_ID" \
  -F "project_name=$PROJECT_ID" \
  -F "files=@/path/to/sample.mcap;type=application/octet-stream"
```

预期 JSON：`ok: true`，`data.imported` 长度 1，项含 `minio_path` 以 `minio://` 开头。

数据库：`data_assets` 新行 `sync_status=synced`，`meta` JSON 内 `storage.minio_path` 存在。

MinIO：bucket（项目名转拼音规则）下存在 `projects/{project_id}/import/{code}/...` 对象。

导出：对该 `id` 走现有导出任务，不应出现「资产未绑定 MinIO 路径」。

## 2) 多文件（curl）

多个 `-F "files=@a.mcap" -F "files=@b.mcap"`（文件名无路径分隔符）。

预期：`imported` 条数等于文件数，每条独立 `minio_path`。

## 3) 文件夹 / 相对路径（浏览器为主）

curl 模拟 multipart 文件名带路径（与 `webkitRelativePath` 一致）：

```bash
curl -sS -X POST "$API/api/data-assets/import" \
  -H "Authorization: Bearer $TOKEN" \
  -F "project=$PROJECT_ID" \
  -F "files=@./meta/info.json;filename=myset/meta/info.json" \
  -F "files=@./data/chunk.something;filename=myset/data/x.bin"
```

需满足后端 LeRobot 目录校验；通过后为**一条** `format=lerobot` 资产，MinIO 为前缀上传。

## 4) 失败（MinIO 不可用）

临时错误 `MINIO_ENDPOINT=127.0.0.1:1` 重启后端后重复单文件导入。

预期：`data.failed` 有原因；不应出现 `synced` 且缺 `minio_path` 的新行（服务端应回滚本地暂存与已写 MinIO/DB，以实际日志为准）。

## 前端步骤摘要

1. 数据页 → 导入数据 → 选项目 → 选择文件（多 mcap/hdf5/zip）→ 确认。  
2. 勿混用「选择文件」与「选择文件夹」；混选应提示错误。  
3. 成功后列表刷新；选中新资产导出到白名单目录。
