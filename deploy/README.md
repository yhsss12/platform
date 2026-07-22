# 部署配置

本目录保存 systemd unit 和其他部署侧配置。

- `rq-worker-*.service`：当前队列 worker unit 模板。
- `legacy/`：保留用于历史参考、但含旧路径或不再作为当前部署入口的配置。

项目本地开发服务仍通过 `scripts/start.sh`、`scripts/restart.sh` 和 `scripts/stop.sh` 管理。
