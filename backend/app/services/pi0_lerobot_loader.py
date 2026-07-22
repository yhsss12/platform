"""Load and validate platform-native LeRobot v3 datasets for pi0 / openpi training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from app.core.platform_paths import platform_paths

logger = logging.getLogger(__name__)

PLATFORM_IMAGE_KEY_MAP = {
    "agentview_image": "observation.images.agentview",
    "robot0_eye_in_hand_image": "observation.images.eye_in_hand",
}

OPENPI_IMAGE_KEY_MAP = {
    "agentview_image": "image.base_0_rgb",
    "robot0_eye_in_hand_image": "image.left_wrist_0_rgb",
}

REQUIRED_METADATA_FIELDS = (
    "format",
    "task_instruction",
    "robot",
    "controller_type",
    "state_dim",
    "action_dim",
    "action_mode",
    "action_representation",
    "image_keys",
    "pi0Ready",
)


@dataclass(frozen=True)
class Pi0LeRobotDatasetSpec:
    root: Path
    metadata: dict[str, Any]
    info: dict[str, Any]
    task_instruction: str
    state_dim: int
    action_dim: int
    robot: str
    controller_type: str
    action_mode: str
    action_representation: str
    image_keys: list[str]
    pi0_ready: bool
    pi0_ready_reason: str
    frame_count: int
    episode_count: int
    norm_stats: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def is_platform_lerobot_v3_dataset(root: Path | str) -> bool:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        return False
    has_meta = (path / "meta" / "info.json").is_file()
    has_parquet = any((path / "data").rglob("*.parquet")) if (path / "data").is_dir() else False
    has_sidecar = (path / "metadata.json").is_file()
    return has_meta and has_parquet and has_sidecar


def resolve_lerobot_path_from_manifest(manifest: dict[str, Any]) -> Path | None:
    """Resolve native LeRobot dataset root from a platform dataset manifest."""
    runtime_roots = (platform_paths.runs_root,)
    candidates: list[str] = []

    lerobot_block = manifest.get("lerobot")
    if isinstance(lerobot_block, dict):
        for key in ("path", "metadataPath"):
            value = lerobot_block.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    elif isinstance(lerobot_block, str) and lerobot_block.strip():
        candidates.append(lerobot_block.strip())

    lerobot_meta = manifest.get("lerobotMetadata")
    if isinstance(lerobot_meta, dict):
        for key in ("path", "metadataPath"):
            value = lerobot_meta.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

    artifacts = manifest.get("artifacts") or {}
    for key in ("lerobotPath", "lerobot"):
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    source_job_id = str(manifest.get("sourceJobId") or "").strip()
    for rel in candidates:
        path = Path(rel).expanduser()
        if not path.is_absolute() and source_job_id.startswith("ct_gen_"):
            path = next(
                (
                    (root / "cable_threading" / "jobs" / source_job_id / rel).resolve()
                    for root in runtime_roots
                    if (root / "cable_threading" / "jobs" / source_job_id / rel).exists()
                ),
                (runtime_roots[0] / "cable_threading" / "jobs" / source_job_id / rel).resolve(),
            )
        elif not path.is_absolute():
            path = (project_root / path).resolve()
        else:
            path = path.resolve()
        if path.name == "metadata.json":
            path = path.parent
        if is_platform_lerobot_v3_dataset(path):
            return path.resolve()
    return None


def load_norm_stats(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    for candidate in (root / "stats.json", root / "meta" / "stats.json"):
        data = _read_json(candidate)
        if data:
            return data
    return {}


def _load_parquet_table(root: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow 不可用，无法读取 LeRobot parquet") from exc

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"LeRobot data parquet 缺失: {root / 'data'}")
    return pq.read_table(parquet_files[0])


def inspect_lerobot_dataset(root: Path | str) -> Pi0LeRobotDatasetSpec:
    path = Path(root).expanduser().resolve()
    if not is_platform_lerobot_v3_dataset(path):
        raise FileNotFoundError(f"不是平台 LeRobot v3 数据集: {path}")

    metadata = _read_json(path / "metadata.json")
    info = _read_json(path / "meta" / "info.json")
    table = _load_parquet_table(path)

    state_dim = int(metadata.get("state_dim") or info.get("features", {}).get("observation.state", {}).get("shape", [0])[0])
    action_dim = int(metadata.get("action_dim") or info.get("features", {}).get("action", {}).get("shape", [0])[0])
    image_keys = list(metadata.get("image_keys") or [])
    task_instruction = str(
        metadata.get("task_instruction")
        or metadata.get("taskInstruction")
        or ""
    ).strip()

    return Pi0LeRobotDatasetSpec(
        root=path,
        metadata=metadata,
        info=info,
        task_instruction=task_instruction,
        state_dim=state_dim,
        action_dim=action_dim,
        robot=str(metadata.get("robot") or "Panda"),
        controller_type=str(metadata.get("controller_type") or ""),
        action_mode=str(metadata.get("action_mode") or ""),
        action_representation=str(metadata.get("action_representation") or ""),
        image_keys=image_keys,
        pi0_ready=bool(metadata.get("pi0Ready")),
        pi0_ready_reason=str(metadata.get("pi0ReadyReason") or ""),
        frame_count=int(metadata.get("frame_count") or info.get("total_frames") or table.num_rows),
        episode_count=int(metadata.get("episode_count") or info.get("total_episodes") or 1),
        norm_stats=load_norm_stats(path),
    )


def validate_lerobot_for_pi0(root: Path | str) -> tuple[bool, str]:
    try:
        spec = inspect_lerobot_dataset(root)
    except (FileNotFoundError, RuntimeError) as exc:
        return False, str(exc)

    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in spec.metadata]
    if missing:
        return False, f"metadata.json 缺少字段: {', '.join(missing)}"

    if not spec.pi0_ready:
        return False, spec.pi0_ready_reason or "pi0Ready=false"

    if spec.state_dim != 9:
        return False, f"state_dim={spec.state_dim}, expected 9"
    if spec.action_dim != 8:
        return False, f"action_dim={spec.action_dim}, expected 8"
    if spec.robot != "Panda":
        return False, f"robot={spec.robot}, expected Panda"
    if spec.controller_type != "JOINT_POSITION":
        return False, f"controller_type={spec.controller_type}, expected JOINT_POSITION"
    if spec.action_mode != "joint_delta_derived":
        return False, f"action_mode={spec.action_mode}, expected joint_delta_derived"
    if spec.action_representation != "normalized_joint_delta":
        return False, (
            f"action_representation={spec.action_representation}, expected normalized_joint_delta"
        )
    if not spec.task_instruction:
        return False, "task_instruction 缺失"

    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key not in spec.image_keys:
            return False, f"image_keys 缺少 {key}"

    table = _load_parquet_table(spec.root)
    if "observation.state" not in table.column_names:
        return False, "parquet 缺少 observation.state"
    if "action" not in table.column_names:
        return False, "parquet 缺少 action"

    state_arr = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action_arr = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state_arr.ndim != 2 or state_arr.shape[1] != 9:
        return False, f"observation.state shape={getattr(state_arr, 'shape', None)}"
    if action_arr.ndim != 2 or action_arr.shape[1] != 8:
        return False, f"action shape={getattr(action_arr, 'shape', None)}"

    for platform_key, lerobot_key in PLATFORM_IMAGE_KEY_MAP.items():
        video_dir = spec.root / "videos" / lerobot_key
        if not any(video_dir.rglob("*.mp4")):
            return False, f"缺少视频: {video_dir}"

    if not spec.norm_stats:
        return False, "norm stats 缺失（stats.json / meta/stats.json）"

    return True, ""


def build_openpi_field_mapping(spec: Pi0LeRobotDatasetSpec) -> dict[str, Any]:
    return {
        "platform_image_keys": dict(PLATFORM_IMAGE_KEY_MAP),
        "openpi_image_keys": dict(OPENPI_IMAGE_KEY_MAP),
        "state_key": "observation.state",
        "action_key": "action",
        "prompt_key": "task_instruction",
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "third_wrist_padding": True,
    }


def iter_lerobot_training_batches(
    root: Path | str,
    *,
    batch_size: int = 2,
    max_batches: int | None = None,
    load_images: bool = True,
) -> Iterator[dict[str, Any]]:
    spec = inspect_lerobot_dataset(root)
    table = _load_parquet_table(spec.root)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    total = states.shape[0]

    video_readers: dict[str, Any] = {}
    video_frames: dict[str, list[np.ndarray]] = {}
    if load_images:
        try:
            import imageio.v2 as iio
        except ImportError:
            iio = None
        if iio is not None:
            for platform_key, lerobot_key in PLATFORM_IMAGE_KEY_MAP.items():
                mp4_files = sorted((spec.root / "videos" / lerobot_key).rglob("*.mp4"))
                if not mp4_files:
                    continue
                frames = [np.asarray(frame, dtype=np.uint8) for frame in iio.mimread(str(mp4_files[0]))]
                video_frames[platform_key] = frames
        else:
            try:
                import cv2
            except ImportError:
                cv2 = None
            if cv2 is not None:
                for platform_key, lerobot_key in PLATFORM_IMAGE_KEY_MAP.items():
                    mp4_files = sorted((spec.root / "videos" / lerobot_key).rglob("*.mp4"))
                    if not mp4_files:
                        continue
                    capture = cv2.VideoCapture(str(mp4_files[0]))
                    frames: list[np.ndarray] = []
                    while capture.isOpened():
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    capture.release()
                    if frames:
                        video_frames[platform_key] = frames

    produced = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch: dict[str, Any] = {
            "observation.state": states[start:end],
            "action": actions[start:end],
            "task_instruction": spec.task_instruction,
            "frame_index": frame_indices[start:end],
            "batch_size": end - start,
        }
        for platform_key in PLATFORM_IMAGE_KEY_MAP:
            frames = video_frames.get(platform_key)
            if frames is not None:
                indices = np.clip(frame_indices[start:end], 0, len(frames) - 1)
                batch[platform_key] = np.stack([frames[int(idx)] for idx in indices], axis=0)
            else:
                batch[platform_key] = None
        batch["image.left_wrist_0_rgb_pad"] = np.zeros((end - start, 256, 256, 3), dtype=np.uint8)
        yield batch
        produced += 1
        if max_batches is not None and produced >= max_batches:
            break


def build_smoke_schema_record(spec: Pi0LeRobotDatasetSpec, *, dataset_path: Path) -> dict[str, Any]:
    return {
        "modelType": "pi0",
        "datasetFormat": "lerobot",
        "datasetPath": str(dataset_path),
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "robot": spec.robot,
        "controller_type": spec.controller_type,
        "action_mode": spec.action_mode,
        "action_representation": spec.action_representation,
        "task_instruction": spec.task_instruction,
        "image_keys": list(spec.image_keys),
        "pi0ReadyData": spec.pi0_ready,
        "field_mapping": build_openpi_field_mapping(spec),
        "norm_stats_source": "stats.json",
    }
