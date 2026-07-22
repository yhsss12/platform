# EAI IDE 离线部署（文件夹 + 镜像）

本目录为完整现场包：**无需源码、无需 build**，仅需 Docker（Linux + host 网络）。

构建时前端 API 地址已写入镜像：`http://172.18.0.114:8000`

## 目标机要求

- Linux，已安装 Docker 与 Docker Compose v2
- 端口未被占用：5432、9000、9001、6379、8000、3001
- 勿在宿主机同时运行 `npm run dev`（会占用 3001）

## 部署步骤

```bash
# 1. 拷贝整个目录到目标机，例如：
sudo mkdir -p /opt/eai-ide
sudo cp -a ./eai-ide-offline/* /opt/eai-ide/
cd /opt/eai-ide

# 2. 载入镜像
./scripts/load-images.sh

# 3. 配置环境（必改密码与 IP）
cp app.env.example .env
# 编辑 .env：将 CHANGE_ME_SERVER_IP 改为 172.18.0.114:8000，并修改数据库/JWT/MinIO 密码

# 4. 一键启动（自动 load 镜像 + compose up）
./install-and-start.sh

# 5. 验证
./scripts/verify.sh
curl http://127.0.0.1:8000/health
```

浏览器访问：`http://172.18.0.114:8000:3001`

## 停止

```bash
docker compose down
docker compose -f docker-compose.postgres-minio.yml down
```

## 镜像列表

| 文件 | 镜像名 |
|------|--------|
| images/eai-ide-local.tar | eai-ide:local |
| images/eai-postgres-local.tar | eai-postgres:local |
| images/eai-minio-local.tar | eai-minio:local |
| images/redis-7-alpine.tar | redis:7-alpine |

详细说明见 README.md。
