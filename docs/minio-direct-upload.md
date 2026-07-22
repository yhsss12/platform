# MinIO 单文件直传（Phase 1）部署与联调

## 前端源（Origin）

浏览器向 **预签名 URL** 发 `PUT` 时，请求源为前端页面所在 Origin（例如 `http://localhost:3000` 或生产域名）。  
后端 `.env` / `app.core.config` 中的 `FRONTEND_ORIGIN` 用于 Cookie、CORS 等与前端对齐；**MinIO bucket CORS 中的 AllowedOrigins 必须与用户实际访问前端的 Origin 一致**（协议 + 主机 + 端口）。

## MinIO 必须配置的 CORS

直传失败且浏览器控制台出现 CORS / preflight 相关错误时，在 **对应项目 bucket** 上配置 CORS（控制台 *Access Rules → CORS*，或 `mc anonymous set-json` / S3 API `PutBucketCors`）。

### 推荐最小规则（单文件 PUT 预签名）

- **AllowedOrigins**：列出所有会打开数据平台的前端地址，例如  
  `http://localhost:3000`、`https://app.example.com`（不要用 `*` 若带凭证或遇浏览器限制，按环境逐项列出更稳）。
- **AllowedMethods**：至少包含 **`PUT`**、**`HEAD`**（部分客户端或 SDK 会发 HEAD；调试时也可能用到）。若同一 bucket 还有下载/预览，可再加 **`GET`**。
- **AllowedHeaders**：至少包含  
  - `Content-Type`（当 `upload-init` 返回的 `headers` 要求带 `Content-Type` 时，浏览器会带该头，需与预签名一致）  
  - `Content-Length`  
  为减少遗漏，可使用 **`*`**（MinIO 支持时）或显式列出上述头。
- **ExposeHeaders**（可选）：`ETag`（便于排查上传结果）。
- **MaxAgeSeconds**：例如 `3600`（预检缓存）。

### `mc` 示例（按环境改 Origin）

将下列 JSON 存为 `cors.json` 后执行（`my-bucket` 改为实际 bucket）：

```json
[
  {
    "AllowedOrigins": ["http://localhost:3000"],
    "AllowedMethods": ["PUT", "HEAD", "GET"],
    "AllowedHeaders": ["Content-Type", "Content-Length", "Authorization"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

```bash
mc anonymous set-json cors.json myminio/my-bucket
```

生产环境请将 `AllowedOrigins` 改为真实前端 HTTPS 域名；若预签名 URL 指向与页面不同的 MinIO 外网域名，**仍由该 bucket 的 CORS 放行页面 Origin**。

## 数据库迁移

在 `backend` 目录执行（需已配置 `DATABASE_URL`）：

```bash
cd backend && alembic upgrade head
```

会执行 `007_upload_sessions`：创建 `upload_sessions`（若表已存在则跳过，兼容历史 `create_all`）。

## 联调验证清单

### 前端操作

1. 登录 → 数据页 → **导入** → 选择项目 → **仅选 1 个** `.mcap` → 确认；列表应出现新资产，`file_path` 为 `minio://...`。
2. 同上选 **1 个** `.hdf5`（或 `.h5`）。
3. **多选多个文件** 或 **选择文件夹**：在 `NEXT_PUBLIC_DIRECT_UPLOAD_MULTI` 未关闭时走 MinIO 多段直传；关闭时仍走原 multipart 导入。
4. 对直传资产执行 **导出**（现有导出任务流），应成功。
5. **删除** 该资产后，MinIO 中对应对象应被删除（或控制台中对象消失）。

### 接口级（需 Bearer Token）

1. `POST /api/data-assets/upload-init`，`upload_mode=single_file`，合法 `project_id`、`filename`、`size_bytes` → 得 `upload_url`、`upload_session_id`。
2. `PUT upload_url`，Body 为原始文件字节，头与返回的 `headers` 一致。
3. `POST /api/data-assets/upload-complete`，`upload_session_id`、与文件一致的 `size_bytes` → `ok: true` 且 `data.asset`。
4. **重试**：会话为 `failed` 或仍为 `presigned`（未过期）、对象已上传且大小一致时，再次 `upload-complete` 应成功或幂等返回同一资产；`completed` 后重复调用应仍返回同一资产。

### 预期数据库

- `upload_sessions`：成功完成后对应行 `status=completed`。
- `data_assets`：新行 `file_path` 形如 `minio://{bucket}/projects/{project_id}/import/v2/{session_id}/{filename}`，`sync_status=synced`，`meta` 中含 `storage.minio_path`（与 `file_path` 一致）。

### 预期 MinIO

- Bucket 为项目名规则归一化后的名称（与现有导入一致）。
- Object key：`projects/{project_id}/import/v2/{upload_session_id}/{filename}`。

## Phase 2（多文件 / 文件夹）

- `POST /api/data-assets/upload-init`：`upload_mode` 为 `multi_file`（≥2 文件）或 `directory`（`items[].relative_path` 为 webkit 相对路径）；响应 `upload_items[]` 逐项 PUT。
- `POST /api/data-assets/upload-complete`：`multi_file` 返回 `data.assets` + `data.failed_items`（部分成功允许）；`directory` 需 `manifest`（`root_dir_name`、`paths`、`total_files`、`total_size_bytes`），入库 **单条** 目录资产，`file_path` 为 `minio://.../dir/.../` 前缀。
- 迁移：`008_upload_sessions_p2` 为 `upload_sessions` 增加 `upload_mode`、`items_json` 等列。
- 前端：`NEXT_PUBLIC_DIRECT_UPLOAD_MULTI=0` 关闭多文件/目录直传（仅走标准 `/import`）。**仅当 `upload-init` 失败、或预签名项缺少 `upload_url`、或上述开关关闭时**才回退标准 `/import`；一旦浏览器已对 MinIO 发起任意文件的 PUT，前端**不再**自动回退，避免重复资产与孤儿对象与旧导入重复写入。
- 目录直传完成后：仅当服务端 `_inspect_lerobot_dir` 判定为 LeRobot 布局时 `format=lerobot` 并解析 meta；否则 `format=directory`，`parse_status=未解析`，`error_msg` 标明未识别为 LeRobot；`meta.storage.dataset_shape` 为 `lerobot` 或 `generic_directory`。

## 孤儿对象与会话状态（最小收口）

直传对象统一落在项目 bucket 下前缀（与单文件一致，会话 id 区分批次）：

`projects/{project_id}/import/v2/{upload_session_id}/...`

- **`upload_sessions.status`**：`presigned`（已发预签名）、`completed`（已登记资产）、`failed`（完成失败或校验失败等）。若 PUT 已开始但 `upload-complete` 失败，会话应落在 **`failed`**，前端已禁止再自动走 `/import`，避免同一批文件二次入库。
- **人工清理 MinIO**：在确认不需要重试 `upload-complete` 后，可按前缀删除该会话下所有对象，例如使用 MinIO 控制台「按前缀删除」，或 `mc`：`mc rm --recursive --force myminio/{bucket}/projects/{project_id}/import/v2/{upload_session_id}/`（将占位符换成实际 bucket 名与 id）。**删除前请确认无有效资产仍引用该前缀**（`completed` 的资产 `file_path`/`minio_path` 会指向其对象或目录前缀）。
- **重试**：若对象已齐、大小与会话一致，可在修正网络或权限后仅重试 `upload-complete`（无需重新 PUT），具体幂等行为见路由实现。

## 已知限制

- 预签名 URL 主机须浏览器可达；内网 MinIO 需配合隧道或对外域名。
- `presigned` 会话过期后须重新 `upload-init`；`failed` 可在对象仍在时重试 `upload-complete`（单文件 / 多文件 / 目录规则见路由实现）。
- 多文件直传单批文件数上限 300；非 LeRobot 布局的目录资产为 `format=directory`，导出走「整目录拷贝」到导出包中的 `directory/` 子目录。
