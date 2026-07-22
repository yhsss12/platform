"""Isaac Lab replay_demos.py 命令参数封装（基于官方脚本 argparse）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# replay_demos.py 原生参数（不含 AppLauncher 扩展项）
REPLAY_DEMO_SCRIPT = "scripts/tools/replay_demos.py"
HDF5_TO_MP4_SCRIPT = "scripts/tools/hdf5_to_mp4.py"

# AppLauncher.add_app_launcher_args 注入、且 replay_demos 支持的常用 flags
APPLAUNCHER_REPLAY_FLAGS = frozenset({"headless", "enable_cameras"})


@dataclass(frozen=True)
class ReplayDemoCliParams:
    task_id: str
    dataset_file: Path
    headless: bool = True
    enable_cameras: bool = True
    video: bool = True
    num_envs: int = 1
    select_episodes: list[int] = field(default_factory=list)
    validate_states: bool = False
    validate_success_rate: bool = False
    enable_pinocchio: bool = False


def build_replay_demos_cli_args(params: ReplayDemoCliParams) -> list[str]:
    """构建传给 IsaacLabCliRunner 的 replay_demos 参数（不含 -p 与脚本路径）。"""
    if not params.task_id.strip():
        raise ValueError("task_id is required")
    if not params.dataset_file.is_file():
        raise FileNotFoundError(f"dataset_file not found: {params.dataset_file}")

    args: list[str] = [
        "--task",
        params.task_id.strip(),
        "--dataset_file",
        str(params.dataset_file),
        "--num_envs",
        str(max(1, int(params.num_envs))),
    ]

    if params.select_episodes:
        args.append("--select_episodes")
        args.extend(str(int(ep)) for ep in params.select_episodes)

    if params.validate_states:
        args.append("--validate_states")
    if params.validate_success_rate:
        args.append("--validate_success_rate")
    if params.enable_pinocchio:
        args.append("--enable_pinocchio")

    if params.headless:
        args.append("--headless")
    if params.enable_cameras:
        args.append("--enable_cameras")

    return args


def build_hdf5_to_mp4_cli_args(
    *,
    input_file: Path,
    output_dir: Path,
    input_keys: Optional[list[str]] = None,
) -> list[str]:
    """replay_demos 不直接支持 --video；若 HDF5 含相机帧，可用 hdf5_to_mp4 后处理。"""
    args = [
        "--input_file",
        str(input_file),
        "--output_dir",
        str(output_dir),
    ]
    if input_keys:
        args.append("--input_keys")
        args.extend(input_keys)
    return args


def replay_demos_supports_video_flag() -> bool:
    """官方 replay_demos.py 当前无 --video；视频通过后处理或 status.videoAvailable=false 表达。"""
    return False
