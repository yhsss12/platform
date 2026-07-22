"""训练节点 SSH 客户端（密码或私钥，均来自环境变量/配置，不入库）。"""

from __future__ import annotations

import logging
import os
import socket
from typing import TYPE_CHECKING

from app.core.env_loader import ensure_dotenv_loaded

if TYPE_CHECKING:
    from app.services.training_node_service import TrainingNodeConfig

logger = logging.getLogger(__name__)


class TrainingNodeSSHError(RuntimeError):
    pass


def _resolve_ssh_auth(config: TrainingNodeConfig) -> tuple[str, dict]:
    """返回 (auth_mode, connect_kwargs_extra)。"""
    ensure_dotenv_loaded()
    password = (config.ssh_password or "").strip()
    key_path = (config.ssh_key_path or "").strip()
    expanded_key = os.path.expanduser(key_path) if key_path else ""

    if password:
        return "password", {
            "password": password,
            "allow_agent": False,
            "look_for_keys": False,
        }

    if expanded_key and os.path.isfile(expanded_key):
        return "key_file", {
            "key_filename": expanded_key,
            "allow_agent": False,
            "look_for_keys": False,
        }

    return "missing", {}


def ssh_credentials_missing_message() -> str:
    return (
        "未配置 SSH 认证：请在 backend/.env 或项目根 .env 中设置 "
        "TRAIN_NODE_L20_PASSWORD（推荐）或 TRAIN_NODE_L20_SSH_KEY"
    )


class TrainingNodeSSHClient:
    def __init__(self, config: TrainingNodeConfig) -> None:
        self._config = config
        self._client = None
        self._auth_mode = "missing"

    def _connect(self):
        if self._client is not None:
            return self._client

        try:
            import paramiko
        except ImportError as exc:
            raise TrainingNodeSSHError(
                "缺少 paramiko 依赖，请安装 backend/requirements.txt 中的 paramiko"
            ) from exc

        cfg = self._config
        if not cfg.host or not cfg.ssh_user:
            raise TrainingNodeSSHError("SSH host/user 未配置")

        auth_mode, auth_kwargs = _resolve_ssh_auth(cfg)
        self._auth_mode = auth_mode
        if auth_mode == "missing":
            raise TrainingNodeSSHError(ssh_credentials_missing_message())

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = {
            "hostname": cfg.host,
            "port": cfg.ssh_port,
            "username": cfg.ssh_user,
            "timeout": 20,
            "banner_timeout": 20,
            "auth_timeout": 20,
            **auth_kwargs,
        }

        try:
            client.connect(**connect_kwargs)
        except paramiko.AuthenticationException as exc:
            client.close()
            if auth_mode == "password":
                raise TrainingNodeSSHError(
                    "SSH 认证失败：TRAIN_NODE_L20_PASSWORD 不正确或账号被锁定"
                ) from exc
            if auth_mode in {"key_file"}:
                raise TrainingNodeSSHError(
                    "SSH 认证失败：请检查 TRAIN_NODE_L20_SSH_KEY 私钥是否已授权到远端 zyf 用户"
                ) from exc
            raise TrainingNodeSSHError(f"SSH 认证失败: {exc}") from exc
        except (paramiko.SSHException, socket.error, OSError) as exc:
            client.close()
            raise TrainingNodeSSHError(str(exc)) from exc

        self._client = client
        return client

    def run(self, command: str, *, timeout: float = 60) -> tuple[int, str, str]:
        client = self._connect()
        try:
            _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            return rc, out, err
        except TrainingNodeSSHError:
            raise
        except Exception as exc:
            raise TrainingNodeSSHError(str(exc)) from exc

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> TrainingNodeSSHClient:
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
