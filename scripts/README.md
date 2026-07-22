# 项目服务管理脚本

## 目录分类

- `maintenance/`：维护、诊断和清理辅助脚本。
- `diagnostics/`：Isaac 等运行环境的只读诊断与探针脚本。
- `release/`：镜像构建、代码保护、离线打包和交付校验脚本。
- `verification/`：API、浏览器、工作空间和任务集成的验收/回归脚本。
- `legacy/`：保留供排查参考、但不再作为当前入口的历史脚本。
- `lib/`：服务管理脚本共用函数。

## 脚本说明

### 0. `restart-all.sh` - 重启整个项目（推荐）
停止前后端与队列 worker，重启 Docker 基础设施（PostgreSQL + MinIO），再后台拉起后端与前端。

```bash
cd /home/ubuntu/project/eai-idev2.1 && bash scripts/restart-all.sh
```

仅重启前后端、不碰 Docker：

```bash
SKIP_DOCKER=1 bash scripts/restart-all.sh
```

### 1. `restart.sh` - 重启服务
停止现有服务并重新启动（不含 Docker，适合只改代码后快速重启）。

### 2. `start.sh` - 启动服务
启动服务（如果服务已在运行则跳过）。

### 3. `stop.sh` - 停止服务
停止正在运行的服务。

## 使用方法

### 重启所有服务
```bash
./scripts/restart.sh
# 或
./scripts/restart.sh all
```

### 重启前端
```bash
./scripts/restart.sh frontend
```

### 重启后端
```bash
./scripts/restart.sh backend
```

### 启动服务
```bash
./scripts/start.sh          # 启动所有服务
./scripts/start.sh frontend # 只启动前端
./scripts/start.sh backend  # 只启动后端
```

### 停止服务
```bash
./scripts/stop.sh           # 停止所有服务
./scripts/stop.sh frontend  # 只停止前端
./scripts/stop.sh backend   # 只停止后端
```

## 配置

### 端口配置
- 前端端口：`3001`（可在脚本中修改 `FRONTEND_PORT`，与 `package.json` 中 `next dev -p 3001` 一致）
- 后端端口：`8000`（可在脚本中修改 `BACKEND_PORT`，或通过环境变量设置）

### 日志文件
- 日志目录：`logs/`
- 前端日志：`logs/frontend.log`
- 后端日志：`logs/backend.log`

### PID 文件
- PID 目录：`.pids/`
- 前端 PID：`.pids/frontend.pid`
- 后端 PID：`.pids/backend.pid`

## CableThreadingMVP 训练环境

重建 `cable-threading-mvp` conda 环境后，请安装 robomimic BC 训练依赖（`PYTHONNOUSERSITE=1` 场景）：

```bash
bash integrations/CableThreadingMVP/scripts/install_training_deps.sh
```

## 注意事项

1. 脚本会自动检测端口占用，如果端口已被占用会尝试停止相关进程
2. 如果进程无法正常停止，脚本会强制终止（kill -9）
3. 日志文件会自动创建在 `logs/` 目录
4. 后端服务目前未配置，如需启动后端请修改 `start_backend()` 函数

## 查看日志

```bash
# 查看前端日志
tail -f logs/frontend.log

# 查看所有日志
tail -f logs/*.log
```

## 故障排查

### 页面 HTTP 500/502（Next.js `.next` 缓存损坏）

**现象：** 访问 `http://<host>:3001/...`（如 `/workspace/data`）返回 HTTP 500 或 502。

**典型日志**（`logs/frontend.log`）：

```
Error: Cannot find module './xxxx.js'
TypeError: Cannot read properties of undefined (reading '/_app')
ENOENT: .../.next/server/vendor-chunks/@swc.js
GET /workspace/data 500 in ...
```

**原因：** Next.js dev 的 `.next` 构建缓存损坏，或 HMR chunk 与运行时不同步。属于**前端构建缓存问题**，不是后端 API 故障，也与 `runs/` 数据清理无关。

**处理：**

```bash
./scripts/restart.sh frontend
```

该脚本会删除 `.next`、`node_modules/.cache`、`.turbo` 等 dev 缓存，以固定命令启动前端：

```bash
INTERNAL_API_URL=http://127.0.0.1:8000 yarn dev -H 0.0.0.0 -p 3001
```

（实现见 `scripts/lib/frontend_dev.sh`，`start.sh` / `restart-all.sh` 共用，**不会**启动到默认 3000 端口。）

启动后自动 curl 验证本机与 LAN IP 的 `/login`；失败则打印最近 120 行 `frontend.log` 并以非 0 退出。

`next.config.js` 已配置 `allowedDevOrigins`（含 LAN IP），dev 模式关闭 webpack 持久化 cache。

**验证：**

```bash
curl -i http://127.0.0.1:3001/workspace/data
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/api/workspace/datasets   # 需登录 token，401 表示后端正常
```

若前端 200、后端 `/health` 200，则问题已恢复。若 datasets 接口 401，属未带认证，非服务异常。

**注意：** 不要因此恢复已清理的 `runs/phygen_*` 等运行产物；此类 500/502 与 PhyGen/PINN 目录清理无关。

### 端口被占用但无法停止
```bash
# 手动查找并停止进程
lsof -ti:3000 | xargs kill -9
# 或
netstat -tlnp | grep :3000
```

### 服务启动失败
1. 检查日志文件：`logs/frontend.log`
2. 检查依赖是否安装：`pnpm install`
3. 检查端口是否被占用：`lsof -i:3000`
