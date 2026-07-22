# EAI IDE 客户现场部署包

本目录用于**离线交付**：客户机只需 `docker load` + 填写 `.env` + `compose up`，无需源码与构建环境。

## 目录结构

```
on-site-deploy/
├── README.md                          # 本说明
├── app.env.example                    # 环境变量模板 → 复制为 .env
├── docker-compose.yml                 # app + worker + redis（仅 image，不 build）
├── docker-compose.postgres-minio.yml  # postgres + minio
├── eai_ide_backup.sql                 # 需从仓库根目录复制（首次初始化 PG）
├── images/                            # docker save 产物（由 export-images.sh 生成）
│   ├── eai-ide-local.tar
│   ├── eai-postgres-local.tar
│   ├── eai-minio-local.tar
│   └── redis-7-alpine.tar
└── scripts/
    ├── export-images.sh               # 在开发机执行：打包镜像
    ├── load-images.sh                 # 现场：载入镜像
    ├── start-all.sh                   # 现场：启动全套
    └── verify.sh                      # 现场：健康检查
```

## 一、在公司（出发前）准备

### 推荐：一键打包「文件夹 + 镜像 tar」

在项目根目录执行（**先设置客户 IP**）：

```bash
export NEXT_PUBLIC_API_URL=http://<客户服务器IP>:8000
chmod +x scripts/release/pack_offline_deploy.sh
./scripts/release/pack_offline_deploy.sh
```

产物：

- 目录：`dist/eai-ide-offline/`（含 `images/*.tar`、compose、脚本、`DEPLOY.md`）
- 压缩包：`dist/eai-ide-offline.tar.gz`（便于 U 盘拷贝）

将整个目录或 tar.gz 拷到目标机即可，**目标机不需要源码**。

---

### 手动分步（与一键包等价）

### 1. 构建交付镜像（在项目根目录 `eai-idev2.0-main/`）

**重要：** 构建前确保根目录**没有**会被 COPY 的 `.env`（已由 `.dockerignore` 排除，但请勿使用 `docker build -f` 时传入含密钥的 build-arg）。

将 `NEXT_PUBLIC_API_URL` 设为**客户浏览器能访问的 API 地址**（会写入前端静态包）：

```bash
cd /path/to/eai-idev2.0-main

export NEXT_PUBLIC_API_URL=http://<客户服务器IP>:8000

docker compose -f docker-compose.postgres-minio.yml build
docker compose build
```

### 2. 验证镜像

```bash
chmod +x scripts/release/verify_release_image.sh
./scripts/release/verify_release_image.sh eai-ide:local
```

交付镜像默认：**Cython .so** 保护 `mcap_converter` 与 `flexible_mcap_to_hdf5`；**PyArmor** 保护其余 `services` 与 `worker.py`。见 `docs/CYTHON_CORE.md`、`docs/PYARMOR.md`。

### 3. 打现场包

```bash
chmod +x release/on-site-deploy/scripts/*.sh
./release/on-site-deploy/scripts/export-images.sh
```

### 4. 复制数据库种子（若需要）

```bash
cp eai_ide_backup.sql release/on-site-deploy/
```

将整个 `release/on-site-deploy/` 目录拷贝到 U 盘或内网共享。

---

## 二、客户现场部署

### 1. 载入镜像

```bash
cd /opt/eai-ide   # 建议路径
./scripts/load-images.sh
```

### 2. 配置环境

```bash
cp app.env.example .env
vim .env
```

必改项：`CHANGE_ME_SERVER_IP`（三处 URL/MinIO 公网）、`POSTGRES_PASSWORD`、`DATABASE_URL`、`JWT_SECRET`、`MINIO_*`、`AGENT_TUNNEL_TOKEN`、`NEXT_PUBLIC_API_URL`、`PUBLIC_BASE_URL`。

> `app.env.example` 已与项目根 `.env` + `backend/.env.example` 对齐。开发 `.env` 中的 `SECRET_KEY` 在现场请改为 **`JWT_SECRET`**（后端 `Settings` 实际读取的变量名）。

> 若构建镜像时 `NEXT_PUBLIC_API_URL` 与客户现场 IP 不一致，需在公司用正确 IP **重新 build** `eai-ide:local` 后再 export。

### 3. 启动

```bash
./scripts/start-all.sh
```

### 4. 验证

```bash
./scripts/verify.sh
```

浏览器访问：`http://<客户IP>:3001`  
API：`http://<客户IP>:8000/health`

---

## 三、镜像命名（与现网一致）

| 镜像 | 容器 |
|------|------|
| `eai-ide:local` | `eai-app`、`eai-worker` |
| `eai-postgres:local` | `eai-postgres` |
| `eai-minio:local` | `eai-minio` |
| `redis:7-alpine`（或 `.env` 中 `REDIS_IMAGE`） | `eai-redis` |

---

## 四、停止与卸载

```bash
docker compose down
docker compose -f docker-compose.postgres-minio.yml down
# 删数据卷（慎用）：docker volume rm ...
```
