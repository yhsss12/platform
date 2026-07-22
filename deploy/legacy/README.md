# 历史部署配置

本目录中的配置不应直接安装到当前机器。

`eai-app.service` 引用了旧仓库 `/home/ubuntu/project/eai-ide`，仅保留用于追溯早期部署方式。

`generate_service.sh` 使用历史 `backend/.venv` 和开发服务器配置，不是当前服务管理入口。
