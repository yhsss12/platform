# 随包分发的 venv 依赖（.deb，可选）

采集机若**未**安装 `python3.x-venv`（`ensurepip` 不可用），一键安装会尝试 `apt-get install`；在**无网/弱网**环境下，可先把所需 `.deb` 放进本目录再**重新打 Agent 包**，安装脚本会优先用这些 deb 通过 `apt-get install` 装齐依赖。

## 1. 与采集机一致的 Ubuntu 上准备

在与目标机**相同**的大版本/架构/默认 `python3` 版本 的机器上执行（示例为 Python 3.10，即包名 `python3.10-venv`）：

```bash
cd /path/to/eai-idev2.0-main/agent/debs
sudo ./fetch-ubuntu-venv-debs.sh
# 或指定与 python3 一致的版本: sudo ./fetch-ubuntu-venv-debs.sh 3.10
```

脚本会把下载的 `*.deb` 放在**当前目录**。确认该目录有 `.deb` 后，在 `backend/agent_packages` 下重新执行 `build_agent_bundle.sh` 打 tar。

## 2. 手动用 apt 只下载、不安装

在同类 Ubuntu 上（需联网 apt）：

```bash
mkdir -p /path/to/eai-idev2.0-main/agent/debs
cd /tmp
sudo apt-get update
sudo apt-get -y -o "Dir::Cache::archives=$(pwd)" install --download-only python3.10-venv
sudo cp -v *.deb /path/to/eai-idev2.0-main/agent/debs/
# 将 python3.10-venv 换成: python3 -c "import sys;v=sys.version_info;print(f'python{v.major}.{v.minor}-venv')"
```

## 3. 说明

- `*.deb` 对**发行版/版本/架构**敏感，请在**与采集机同系列**的镜像上准备。
- 本目录若为空，安装脚本仍会尝试在线 `apt install pythonX.Y-venv`（见 `installer_template.sh`）。
