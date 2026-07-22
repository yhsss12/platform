# Docker：API 与 Worker 拆分部署

同一镜像 **`Dockerfile.eai-ide`**，入口脚本 `docker/start.sh` 由环境变量 **`START_MODE`** 切换：

| `START_MODE` | 行为 |
|--------------|------|
| `all`（默认） | 后台 uvicorn + 前台 `npm run dev`，与历史单容器一致 |
| `api` | 仅 FastAPI，监听 `8000` |
| `worker` | 仅 RQ Worker，消费 `RQ_WORKER_QUEUES` |

本地推荐镜像标签：**`eai-ide:local`**（`docker compose` 会在构建 **`app`** 服务时打上该标签，`worker` 共用此镜像）。

---

## 推荐：`docker compose`（一键：前端+后端同容器 + Worker + Redis）

仓库根目录 **`docker-compose.yml`** 已配置：

| 服务 | 容器名 | 说明 |
|------|--------|------|
| `redis` | `eai-redis` | **`network_mode: host`**，监听宿主机 **6379**（勿与宿主机已有 Redis 冲突） |
| `app` | `eai-app` | **`network_mode: host`** + **`START_MODE=all`**：uvicorn **8000**、Next **3001** 直接绑在宿主机；compose 注入 **`REDIS_HOST=127.0.0.1`**、`INTERNAL_API_URL` |
| `worker` | `eai-worker` | **`network_mode: host`** + **`START_MODE=worker`**；`REDIS_HOST=127.0.0.1` |

**说明（host 网络）**

- **仅 Linux** 下 `network_mode: host` 行为符合预期；Docker Desktop（Mac/Windows）不适用或行为不同。
- **无 `-p` 映射**：访问 **`http://<宿主机IP>:8000`**（API）、**`http://<宿主机IP>:3001`**（前端）。
- **PostgreSQL 在本机** 时，`.env` 里 **`DATABASE_URL` 主机建议 `127.0.0.1`**，Postgres 看到的客户端即本机连接，**一般无需再为 Docker bridge 网段改 `pg_hba`**。
- **`REDIS_HOST`**：当前 compose 已对 **`app`/`worker` 写死 `127.0.0.1`**（与 host 网络下本机 Redis 一致）；`.env` 里若仍写 `redis` 会被覆盖。

**资源分配（可按宿主机改 `docker-compose.yml` 中 `deploy.resources`）**

| 服务 | limits（上限） | reservations（软预留） | 意图 |
|------|------------------|---------------------------|------|
| `redis` | 1 CPU / 512M | 0.1 CPU / 128M | 队列 broker 轻量 |
| `app` | 3 CPU / 5G | 0.5 CPU / 1G | Next + FastAPI 同进程组 |
| `worker` | 6 CPU / 10G | 0.75 CPU / 2G | 转换 / 标注 / 批量等重任务 |

内存小于 16G 的机器请酌情 **降低 `worker` 的 `memory` 上限**，或暂时删除各服务下的 `deploy:` 段（旧版 Compose 若不支持 `deploy` 也可删掉整段）。

**`.env`（与 compose 的关系）**

- **`docker-compose.yml` 覆盖**：`START_MODE`；**`INTERNAL_API_URL=http://127.0.0.1:8000`**（Next rewrite）；**`REDIS_HOST`/`REDIS_PORT`** 在 **`app`/`worker` 上写死为 `127.0.0.1`/`6379`**（host 网络）。**`DATABASE_URL`、MinIO、`USE_QUEUE`、`RQ_WORKER_QUEUES`、`NEXT_PUBLIC_API_URL`、JWT 等仍来自 `.env`**。
- **PostgreSQL 在本机**：`DATABASE_URL` 主机用 **`127.0.0.1`** 最省事；库在其它机器时仍用可达 IP。
- **浏览器访问**：**`http://<宿主机IP>:3001`**（前端）、**`http://<宿主机IP>:8000/docs`**（API）。`.env` 中 **`NEXT_PUBLIC_API_URL`** 填浏览器能访问的后端根地址（如 **`http://192.168.1.5:8000`**）。

**启动**

```bash
cd /path/to/eai-idev2.0-main
docker compose up -d --build
```

**常用命令**

```bash
docker compose ps
docker compose logs -f app worker
docker compose down
```

**校验**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs
redis-cli ping
```

### 故障排查：`ECONNREFUSED 127.0.0.1:8000` + `no pg_hba.conf entry for host "172.xx.x.x"`

**`ECONNREFUSED`**：多为 uvicorn 未起来（见 **`docker logs eai-app`** 里 lifespan/数据库错误）。

**`no pg_hba... host "172.xx.x.x"`**：出现在 **bridge 网络** 的容器连「宿主机/它机上的 Postgres」时，客户端源 IP 为 **Docker 网桥地址**。当前仓库 **`docker-compose.yml` 已改为 `network_mode: host`**（Linux），一般应把 **`DATABASE_URL` 主机改为 `127.0.0.1`**（本机库），即可避免按 bridge IP 做认证的问题。

若 Postgres **不在本机**，仍用远程 IP 时，须在 **`pg_hba.conf`** 放行 **真实客户端源地址**（host 网络下多为 **宿主机网卡 IP**，而非 `172.20.x.x`）。

以下为 **仍使用 bridge 编排** 时的参考：在 **`pg_hba.conf`** 放行 Docker 网段（如 **`172.16.0.0/12`**）、确认 **`listen_addresses`**、区分 **`host`/`hostssl`**、必要时 **`host.docker.internal`**；修改 **`docker/start.sh`** 后须 **`docker compose build app --no-cache`**。

**扩 Worker 实例（可选）**

默认 `docker-compose.yml` 使用固定 `container_name: eai-worker`，与 `scale` 冲突。需要多 Worker 时请去掉 `worker` 的 `container_name` 后再执行：

```bash
docker compose up -d --scale worker=2
```

**仅在本机另起前端（一般不必，compose 的 `app` 已含 Next）**

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

---

## 构建镜像（本机）

手动 `docker run` 或单独排查时可构建：

```bash
docker build -f Dockerfile.eai-ide -t eai-ide:local .
```

若构建卡在拉取基础镜像，可先测 Docker Hub 连通性：

```bash
docker pull ubuntu:22.04
```

构建完成后镜像体积通常为 **数 GB 量级**（含 Node、Python 依赖、PyTorch CPU 等），可用 `docker images eai-ide:local` 查看。

---

## 推荐环境变量（拆分部署）

- `USE_QUEUE=true`
- `AUTO_START_WORKERS=false`（仅 API 容器）——避免 API 进程内再 fork worker，与独立 worker 容器重复消费。
- `REDIS_HOST`、`REDIS_PORT`：**`docker compose` 且使用文件内 `redis` 服务时** 在 `.env` 中填 **`redis`** / **`6379`**；手写 `docker run` 且 Redis 容器名为 `eai-redis` 时填 **`eai-redis`**。

Worker 容器不需要暴露 HTTP 端口；浏览器与前端访问 **宿主机映射到 API 容器的端口**（例如 `8000` 或 `3002` → 容器内 `8000`）。

---

## 双容器完整启动流程（`docker run` + `eai-net`）

以下假设：项目根目录已有可用 **`.env`**（`DATABASE_URL`、MinIO、JWT 等），且数据库/MinIO 从容器内可访问（常见做法：填 **宿主机局域网 IP** 或 **`host.docker.internal`**，Linux 需在 `docker run` 上加 **`--add-host=host.docker.internal:host-gateway`**）。

### 0. 构建镜像（若尚未构建）

```bash
cd /path/to/eai-idev2.0-main
docker build -f Dockerfile.eai-ide -t eai-ide:local .
```

### 1. 准备 `.env` 中与队列相关的项

```bash
USE_QUEUE=true
AUTO_START_WORKERS=false
REDIS_HOST=eai-redis
REDIS_PORT=6379
```

（数据库等其它变量保持与你线上一致。）

### 2. 创建 Docker 网络（仅需一次；若已存在会报错可忽略）

```bash
docker network create eai-net
```

### 3. 清理同名旧容器（可选）

```bash
docker rm -f eai-api eai-worker eai-redis 2>/dev/null || true
```

### 4. 启动 Redis（与 API / Worker 在同一网络）

```bash
docker run -d --name eai-redis --network eai-net --restart unless-stopped \
  -p 6379:6379 \
  redis:7-alpine redis-server --appendonly yes
```

### 5. 启动 API 容器

宿主机 **8000** 未被占用时：

```bash
docker run -d --name eai-api --network eai-net --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -e START_MODE=api \
  -e USE_QUEUE=true \
  -e AUTO_START_WORKERS=false \
  -e REDIS_HOST=eai-redis \
  eai-ide:local
```

宿主机 **8000 已被占用**（例如改映射到 **3002**）：

```bash
docker run -d --name eai-api --network eai-net --restart unless-stopped \
  -p 3002:8000 \
  --env-file .env \
  -e START_MODE=api \
  -e USE_QUEUE=true \
  -e AUTO_START_WORKERS=false \
  -e REDIS_HOST=eai-redis \
  eai-ide:local
```

访问 API：`http://<宿主机IP>:8000` 或 `http://<宿主机IP>:3002`。

若 Postgres 在宿主机且希望用 `host.docker.internal`，在上述 `docker run` 中增加：

`--add-host=host.docker.internal:host-gateway`

并把 `.env` 里 `DATABASE_URL` 主机改为 `host.docker.internal`（或直接使用可达 IP）。

### 6. 启动 Worker 容器（第二容器）

```bash
docker run -d --name eai-worker --network eai-net --restart unless-stopped \
  --env-file .env \
  -e START_MODE=worker \
  -e USE_QUEUE=true \
  -e REDIS_HOST=eai-redis \
  -e REDIS_PORT=6379 \
  -e RQ_WORKER_QUEUES=gpu_queue,cpu_queue,io_queue,collect_queue \
  eai-ide:local
```

Worker **不要映射 HTTP 端口**。若 Worker 也需访问宿主机 Postgres，同样加上 `--add-host=host.docker.internal:host-gateway`。

API 与 Worker 若需读写 **同一主机目录**，两边应挂载 **相同 volume**（`-v /data/assets:/data/assets` 等），路径需与你的 `.env`/MinIO 一致。

### 7. 校验

```bash
docker ps
redis-cli -h 127.0.0.1 -p 6379 ping
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs
# 若 API 映射在 3002：
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3002/docs
docker logs eai-worker --tail 50
```

### 8. 前端（宿主机开发时）

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:3002
# 或 8000，须与第 5 步映射一致
npm run dev
```

---

## 方式一：`docker compose`（摘要）

完整说明（资源、`EAI_API_PORT`、扩缩容注意）见 **文首「推荐：`docker compose`」**；此处仅作索引。

---

## 方式二：两条命令速启（同「完整流程」缩写）

与上文 **双容器完整启动流程** 一致；仅端口固定为 **8000**。若端口冲突请改用 **`-p 3002:8000`** 并同步 **`NEXT_PUBLIC_API_URL`**。

```bash
docker network create eai-net 2>/dev/null || true
docker rm -f eai-redis eai-api eai-worker 2>/dev/null || true
docker run -d --name eai-redis --network eai-net --restart unless-stopped \
  -p 6379:6379 redis:7-alpine redis-server --appendonly yes
docker run -d --name eai-api --network eai-net --restart unless-stopped -p 8000:8000 \
  --env-file .env -e START_MODE=api -e USE_QUEUE=true -e AUTO_START_WORKERS=false \
  -e REDIS_HOST=eai-redis eai-ide:local
docker run -d --name eai-worker --network eai-net --restart unless-stopped \
  --env-file .env -e START_MODE=worker -e USE_QUEUE=true -e REDIS_HOST=eai-redis \
  -e RQ_WORKER_QUEUES=gpu_queue,cpu_queue,io_queue,collect_queue eai-ide:local
```

---

## 单容器（不改历史行为）

不传 `START_MODE` 或显式 `START_MODE=all`，保持原 **`docker/start.sh`**：**后端 + 前端同容器**。示例：

```bash
docker run --rm -p 8000:8000 -p 3000:3000 --env-file .env eai-ide:local
```

若仍希望队列由本容器自动拉 worker，使用 `USE_QUEUE=true` 且 **`AUTO_START_WORKERS=true`**（默认）。

---

## 数据目录与 MinIO

转换、同步若依赖本地路径，请确保 Worker 容器能访问与 **app** 一致的存储（共享 volume、或统一走 MinIO `warehouse_path`）。仅拆容器不自动挂载磁盘时，需在 compose 中为 **`app`/`worker`** 增加相同 `volumes`。

---

## 说明

- 导出任务当前仍在 API 进程内 `asyncio` 执行；拆分 API/Worker **不会**把导出搬到 worker，如需需另行改造。
- `collect_queue` 若暂无入队任务，worker 监听该队列无害。
