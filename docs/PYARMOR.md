# PyArmor 交付构建说明

## 保护范围

| 路径 | 说明 |
|------|------|
| `backend/app/services/` | 除跳过列表外全部混淆 |
| `backend/app/api/` | REST 路由层 |
| `backend/app/crud/` | 数据访问层 |
| `scripts/relman/` | 除 `flexible_mcap_to_hdf5.py`（由 Cython 保护） |
| `backend/worker.py` | RQ 任务入口 |
| `/app/label_task_description.py` | 自动标注脚本（项目根，打入镜像后混淆） |

**超大文件（默认）：** `mcap_converter.py`、`flexible_mcap_to_hdf5.py` 由 **Cython → .so** 保护（见 `docs/CYTHON_CORE.md`）。

**保持明文（薄层/框架）：** `backend/app/main.py`、`backend/app/models/`、`backend/app/db/`、`backend/app/schemas/`、`alembic/`。

## 构建

```bash
export NEXT_PUBLIC_API_URL=http://<客户IP>:8000
export PYARMOR_ENABLED=true

docker compose -f docker-compose.yml up -d --build
# 或
./scripts/release/build_release_images.sh
```

关闭混淆（调试）：

```bash
docker compose build --build-arg PYARMOR_ENABLED=false
```

## 环境变量（build-arg / Dockerfile ARG）

| 变量 | 默认 | 说明 |
|------|------|------|
| `PYARMOR_ENABLED` | `true` | `false` 时不混淆 |
| `PYARMOR_SKIP_FILES` | `mcap_converter.py,flexible_mcap_to_hdf5.py,routes_data_assets.py` | 保留明文的大文件 |
| `PYARMOR_MAX_LINES` | `2000` | 超过此行数的 `.py` 保留明文（Trial 限制） |
| `PYARMOR_OBF_CODE` | `1` | trial 许可可改为 `0` |

**Trial 说明：** `backend/app/api/` 改为**逐文件**混淆；`routes_data_assets.py`（约 5k 行）及超过 `PYARMOR_MAX_LINES` 的路由保留明文，核心业务仍在已混淆的 `services/` 与 Cython `.so` 中。

## 认证相关 .env 别名

根目录 `.env` 常用 `SECRET_KEY` / `ALGORITHM` / `ACCESS_TOKEN_EXPIRE_MINUTES`，后端 `Settings` 已映射为 `JWT_SECRET` / `JWT_ALG` / `ACCESS_TOKEN_EXPIRE_MIN`，Docker `env_file` 注入后登录与刷新令牌可正常工作。

## 许可证说明

- **超大脚本**默认用 **Cython** 编译（`CYTHON_ENABLED=true`），不再依赖 PyArmor 处理大文件。
- 若仅使用 PyArmor Trial、且关闭 Cython，超大文件会列入 `PYARMOR_SKIP_FILES` 并以 **明文** 留在镜像中（不推荐交付）。

## 验证

```bash
./scripts/release/verify_release_image.sh eai-ide:local
docker exec eai-app head -3 /app/backend/app/services/annotation_service.py
# 应看到 from pyarmor_runtime_000000 import __pyarmor__
```

## 已知限制

- `mcap_converter.custom_processor` 动态 `importlib` 加载在混淆环境下不可用；交付环境请勿依赖该扩展点。
- 混淆需在 **与运行相同的平台** 上生成（Linux x86_64 → Ubuntu 22.04 容器内构建）。
