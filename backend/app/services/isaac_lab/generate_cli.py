"""Isaac Lab Mimic / record_demos 数据生成 CLI 参数封装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ANNOTATE_DEMOS_SCRIPT = "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"
GENERATE_DATASET_SCRIPT = "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"
MIMIC_GENERATE_WITH_LIVE_SCRIPT = "scripts/platform/stack_cube_mimic_generate_with_live.py"
REPLAY_WITH_LIVE_SCRIPT = "scripts/platform/stack_cube_replay_with_live.py"
RECORD_DEMOS_SCRIPT = "scripts/tools/record_demos.py"

DEFAULT_MIMIC_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
DEFAULT_RECORD_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEFAULT_SCRIPTED_EXPERT_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEFAULT_EXPERT_POLICY_TASK_ID = DEFAULT_SCRIPTED_EXPERT_TASK_ID
SCRIPTED_EXPERT_SCRIPT_BASENAME = "stack_cube_scripted_expert.py"
EXPERT_POLICY_SCRIPT_BASENAME = "stack_cube_expert_policy.py"

GENERATION_MODES = frozenset(
    {"mimic_auto", "teleop_record", "replay_seed", "scripted_expert", "expert_policy"}
)


def normalize_generation_mode(mode: str | None) -> str:
    """Map legacy scripted_expert to canonical expert_policy for new jobs."""
    value = (mode or "expert_policy").strip()
    if value == "scripted_expert":
        return "expert_policy"
    return value


def is_expert_policy_mode(mode: str | None) -> bool:
    value = (mode or "").strip()
    return value in {"expert_policy", "scripted_expert"}


@dataclass(frozen=True)
class MimicGenerateCliParams:
    mimic_task_id: str
    seed_dataset_file: Path
    annotated_dataset_file: Path
    output_dataset_file: Path
    num_demos: int
    num_envs: int
    headless: bool
    enable_cameras: bool
    device: str = "cpu"


@dataclass(frozen=True)
class ScriptedExpertCliParams:
    task_id: str
    dataset_file: Path
    num_demos: int
    seed: int
    max_attempts: int
    headless: bool
    enable_cameras: bool
    device: str = "cpu"
    record_camera_obs: bool = True
    image_resolution: int = 128
    include_wrist_camera: bool = False
    live_frame_dir: Path | None = None
    live_status_out: Path | None = None
    live_frame_every: int = 5


@dataclass(frozen=True)
class TeleopRecordCliParams:
    task_id: str
    dataset_file: Path
    num_demos: int
    headless: bool
    enable_cameras: bool
    teleop_device: str = "keyboard"
    device: str = "cpu"


def build_annotate_demos_cli_args(
    *,
    mimic_task_id: str,
    input_file: Path,
    output_file: Path,
    headless: bool,
    enable_cameras: bool,
    device: str = "cpu",
) -> list[str]:
    args = [
        "--task",
        mimic_task_id.strip(),
        "--input_file",
        str(input_file),
        "--output_file",
        str(output_file),
        "--auto",
        "--device",
        device,
    ]
    if headless:
        args.append("--headless")
    if enable_cameras:
        args.append("--enable_cameras")
    return args


def build_generate_dataset_cli_args(params: MimicGenerateCliParams) -> list[str]:
    args = [
        "--task",
        params.mimic_task_id.strip(),
        "--input_file",
        str(params.annotated_dataset_file),
        "--output_file",
        str(params.output_dataset_file),
        "--generation_num_trials",
        str(max(1, int(params.num_demos))),
        "--num_envs",
        str(max(1, int(params.num_envs))),
        "--device",
        params.device,
    ]
    if params.headless:
        args.append("--headless")
    if params.enable_cameras:
        args.append("--enable_cameras")
    return args


def build_mimic_generate_with_live_cli_args(
    params: MimicGenerateCliParams,
    *,
    live_frame_dir: Path,
    live_status_out: Path | None = None,
    preview_video_out: Path | None = None,
    live_frame_every: int = 5,
    visual_env_index: int = 0,
) -> list[str]:
    args = build_generate_dataset_cli_args(params)
    args.extend(["--live_frame_dir", str(live_frame_dir)])
    args.extend(["--live_frame_every", str(max(1, live_frame_every))])
    args.extend(["--visual_env_index", str(max(0, visual_env_index))])
    if live_status_out is not None:
        args.extend(["--live_status_out", str(live_status_out)])
    if preview_video_out is not None:
        args.extend(["--preview_video_out", str(preview_video_out)])
    return args


def build_replay_with_live_cli_args(
    *,
    task_id: str,
    dataset_file: Path,
    live_frame_dir: Path,
    headless: bool,
    enable_cameras: bool,
    device: str = "cpu",
    live_status_out: Path | None = None,
    preview_video_out: Path | None = None,
    live_frame_every: int = 3,
    select_episodes: list[int] | None = None,
) -> list[str]:
    args = [
        "--task",
        task_id.strip(),
        "--dataset_file",
        str(dataset_file),
        "--live_frame_dir",
        str(live_frame_dir),
        "--live_frame_every",
        str(max(1, live_frame_every)),
        "--device",
        device,
    ]
    eps = select_episodes if select_episodes else [0]
    args.append("--select_episodes")
    args.extend(str(int(ep)) for ep in eps)
    if live_status_out is not None:
        args.extend(["--live_status_out", str(live_status_out)])
    if preview_video_out is not None:
        args.extend(["--preview_video_out", str(preview_video_out)])
    if headless:
        args.append("--headless")
    if enable_cameras:
        args.append("--enable_cameras")
    return args


def build_expert_policy_cli_args(params: ScriptedExpertCliParams) -> list[str]:
    return build_scripted_expert_cli_args(params)


def build_scripted_expert_cli_args(params: ScriptedExpertCliParams) -> list[str]:
    args = [
        "--task",
        params.task_id.strip(),
        "--dataset_file",
        str(params.dataset_file),
        "--num_demos",
        str(max(1, int(params.num_demos))),
        "--seed",
        str(int(params.seed)),
        "--max_attempts",
        str(max(0, int(params.max_attempts))),
        "--device",
        params.device,
    ]
    if params.headless:
        args.append("--headless")
    if params.enable_cameras:
        args.append("--enable_cameras")
    if not params.record_camera_obs:
        args.append("--no-record_camera_obs")
    if params.image_resolution:
        args.extend(["--image_resolution", str(max(32, int(params.image_resolution)))])
    if params.include_wrist_camera:
        args.append("--include_wrist_camera")
    if params.live_frame_dir is not None:
        args.extend(["--live_frame_dir", str(params.live_frame_dir)])
    if params.live_status_out is not None:
        args.extend(["--live_status_out", str(params.live_status_out)])
    if params.live_frame_every:
        args.extend(["--live_frame_every", str(max(1, int(params.live_frame_every)))])
    return args


def build_record_demos_cli_args(params: TeleopRecordCliParams) -> list[str]:
    args = [
        "--task",
        params.task_id.strip(),
        "--dataset_file",
        str(params.dataset_file),
        "--num_demos",
        str(max(1, int(params.num_demos))),
        "--teleop_device",
        params.teleop_device,
        "--device",
        params.device,
    ]
    if params.headless:
        args.append("--headless")
    if params.enable_cameras:
        args.append("--enable_cameras")
    return args


def resolve_num_envs(num_demos: int, requested: Optional[int] = None) -> int:
    """平台演示默认单环境；高级设置可指定并行环境数以加速生成。"""
    if requested is not None and requested > 0:
        return min(max(1, requested), max(1, num_demos))
    return 1
