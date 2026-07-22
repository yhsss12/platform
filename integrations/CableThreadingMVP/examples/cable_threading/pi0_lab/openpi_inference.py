"""openpi inference bridge for pi0 cable-threading evaluation."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PI0_SHIM_MARKER = b"PI0_PLATFORM_SHIM_CHECKPOINT"


def _resolve_openpi_python() -> list[str]:
    raw = os.environ.get("OPENPI_PYTHON", "").strip()
    if not raw:
        raise RuntimeError("OPENPI_PYTHON 未配置，无法进行 pi0 推理")
    parts = shlex.split(raw)
    if not parts:
        raise RuntimeError("OPENPI_PYTHON 未配置，无法进行 pi0 推理")
    return parts


def _find_infer_script() -> Path:
    root_raw = os.environ.get("OPENPI_ROOT", "").strip()
    if not root_raw:
        raise RuntimeError("OPENPI_ROOT 未配置，无法进行 pi0 推理")
    root = Path(root_raw).expanduser().resolve()
    candidates = [
        root / "scripts" / "infer.py",
        root / "scripts" / "serve_policy.py",
    ]
    override = os.environ.get("OPENPI_INFER_SCRIPT", "").strip()
    if override:
        candidates.insert(0, Path(override).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"openpi 推理脚本未找到（OPENPI_ROOT={root}）")


def _load_checkpoint_meta(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"pi0 checkpoint not found: {checkpoint_path}")
    raw = checkpoint_path.read_bytes()
    if raw == PI0_SHIM_MARKER or raw.startswith(b"PI0_PLATFORM_SHIM"):
        raise RuntimeError("pi0 checkpoint 来自平台 shim，请使用真实 openpi 训练产物进行评测")

    text = checkpoint_path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload

    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    raise ValueError(f"无法解析 pi0 checkpoint: {checkpoint_path}")


def infer_pi0_action_chunk(
    *,
    checkpoint_path: Path,
    obs: dict[str, Any],
    camera_keys: list[str],
    low_dim_keys: list[str],
    action_dim: int,
    device: str = "cuda",
    action_horizon: int = 8,
) -> list[np.ndarray]:
    """Run one openpi inference step and return an action chunk."""
    meta = _load_checkpoint_meta(checkpoint_path)
    camera_keys = list(meta.get("camera_keys") or camera_keys)
    low_dim_keys = list(meta.get("low_dim_keys") or low_dim_keys)
    action_dim = int(meta.get("action_dim") or action_dim)
    action_horizon = int(meta.get("action_horizon") or action_horizon)

    infer_script = _find_infer_script()
    python_cmd = _resolve_openpi_python()

    with tempfile.TemporaryDirectory(prefix="pi0_infer_") as tmpdir:
        obs_path = Path(tmpdir) / "obs.json"
        obs_path.write_text(json.dumps(obs, default=_json_default), encoding="utf-8")
        cmd = [
            *python_cmd,
            str(infer_script),
            "--checkpoint",
            str(meta.get("checkpointPath") or checkpoint_path),
            "--obs-json",
            str(obs_path),
            "--action-dim",
            str(action_dim),
            "--horizon",
            str(action_horizon),
        ]
        env = os.environ.copy()
        root = os.environ.get("OPENPI_ROOT", "").strip()
        if root:
            env["OPENPI_ROOT"] = str(Path(root).expanduser().resolve())
        completed = subprocess.run(
            cmd,
            cwd=root or None,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"pi0 推理失败: {detail or 'unknown error'}")

        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        actions = payload.get("actions") or []
        if not actions:
            raise RuntimeError("pi0 推理返回空 action chunk")
        return [np.asarray(row, dtype=np.float32) for row in actions]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
