# 核心大文件 Cython 保护（方案 B）

PyArmor Trial 无法混淆超大脚本时，对以下两个模块在 **Docker builder** 内编译为 `.so`，**交付镜像中不包含 `.py` 明文**：

| 模块 | 路径 |
|------|------|
| MCAP → HDF5 核心 | `backend/app/services/mcap_converter.py` → `mcap_converter*.so` |
| Relman 转换 | `scripts/relman/flexible_mcap_to_hdf5.py` → `flexible_mcap_to_hdf5*.so` |

其余 `app/services` 仍由 **PyArmor** 保护（见 `docs/PYARMOR.md`）。

## 构建

默认已开启（`Dockerfile.eai-ide`）：

```bash
export CYTHON_ENABLED=true   # 默认
docker compose build
```

本地调试镜像（保留 .py 便于排错）：

```bash
docker compose build --build-arg CYTHON_ENABLED=false --build-arg PYARMOR_ENABLED=false
```

## 脚本

- `scripts/release/cython_compile_core.sh` — 在 builder 内执行，编译后删除 `.py` / `.c`
- 顺序：安装依赖 → **Cython** → **PyArmor** → runtime COPY

## 验证

```bash
./scripts/release/verify_release_image.sh eai-ide:local

docker exec eai-app ls /app/backend/app/services/mcap_converter*.so
docker exec eai-app test ! -f /app/backend/app/services/mcap_converter.py
```

## 开发说明

- **Git 仓库内仍是 .py 源码**，仅交付镜像去掉明文。
- 修改 `mcap_converter` / `flexible_mcap_to_hdf5` 后须 **重新 docker build** 才会更新 .so。
- 编译需 `build-essential`（builder 阶段已安装）。

## 与 PyArmor 正式许可的关系

若已购买 PyArmor Pro 并清空 `PYARMOR_SKIP_FILES`，可二选一：

- 继续 Cython（保护强度通常更高）
- 或 `CYTHON_ENABLED=false`，改由 PyArmor 混淆两个大文件

不建议对同一模块同时 Cython + PyArmor。
