# 发布脚本

本目录统一保存交付链路工具：

- `build_release_images.sh`：构建本地交付镜像并执行校验。
- `cython_compile_core.sh`：镜像构建阶段编译核心 Python 模块。
- `pack_offline_deploy.sh`：生成现场离线部署包。
- `pyarmor_obfuscate.sh`：镜像构建阶段的 Python 代码保护。
- `verify_release_image.sh`：检查交付镜像内容。

这些脚本均从仓库根目录解析构建上下文，移动后命令入口统一为
`scripts/release/<script-name>`。
