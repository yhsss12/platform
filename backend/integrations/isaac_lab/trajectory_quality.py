"""Offline HDF5 trajectory quality analysis for Isaac Lab stack-cube datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Arm dims 0:3 translation + 3:6 rotation (IK-Rel); gripper is dim 6.
ARM_ACTION_DIMS = slice(0, 6)
GRIPPER_DIM = 6

def _gripper_discrete_state(value: float) -> str:
    if value > 0.25:
        return "open"
    if value < -0.25:
        return "close"
    return "hold"


def _count_gripper_switches(gripper: np.ndarray) -> int:
    if gripper is None or len(gripper) < 2:
        return 0
    switches = 0
    prev = _gripper_discrete_state(float(gripper[0]))
    for value in gripper[1:]:
        state = _gripper_discrete_state(float(value))
        if state != "hold" and prev != "hold" and state != prev:
            switches += 1
        if state != "hold":
            prev = state
    return switches

DEFAULT_ARM_DELTA_WARN = 0.35
DEFAULT_ARM_DELTA_FAIL = 0.55
DEFAULT_ROT_SATURATION_WARN = 0.85
DEFAULT_EPISODE_LENGTH_WARN = 500
DEFAULT_GRIPPER_SWITCH_WARN = 8

MOTION_WARNING_MARKERS = (
    "arm_action_delta",
    "rot_cmd_saturated",
    "wrist_flip",
    "joint_jump",
)


def _is_motion_anomaly_warning(tag: str) -> bool:
    lower = tag.lower()
    return any(marker in lower for marker in MOTION_WARNING_MARKERS)


def classify_quality_display(
    *,
    quality_status: str,
    quality_warnings: list[str],
    suspicious_wrist_flip_count: int = 0,
    suspicious_joint_jump_count: int = 0,
    generation_mode: str = "",
) -> dict[str, str]:
    """Map raw QA status to user-facing display tier/label/hint/severity."""
    if quality_status == "failed":
        return {
            "qualityDisplayTier": "failed",
            "qualityDisplayLabel": "未通过",
            "qualityDisplayHint": "轨迹质量检查未通过，不建议用于训练。",
            "qualityDisplaySeverity": "failed",
            "qualityDisplayDescription": "轨迹质量检查未通过，不建议用于训练。",
            "qualityDisplayRecommendation": "请重新生成数据或调整策略参数后再次导出。",
        }
    if quality_status == "passed":
        return {
            "qualityDisplayTier": "passed",
            "qualityDisplayLabel": "通过",
            "qualityDisplayHint": "",
            "qualityDisplaySeverity": "passed",
            "qualityDisplayDescription": "轨迹动作质量正常。",
            "qualityDisplayRecommendation": "",
        }

    has_motion = suspicious_wrist_flip_count > 0 or suspicious_joint_jump_count > 0
    if not has_motion:
        has_motion = any(_is_motion_anomaly_warning(w) for w in quality_warnings)

    if not has_motion:
        scripted_hint = (
            "该数据集动作质量较稳定，但执行步数偏长，后续可继续优化策略效率。"
            if generation_mode in {"scripted_expert", "expert_policy"}
            else "轨迹动作质量正常，但执行步数较长，后续可优化策略效率。"
        )
        return {
            "qualityDisplayTier": "usable_long_episode",
            "qualityDisplayLabel": "可用（episode 较长）",
            "qualityDisplayHint": "轨迹动作质量正常，但执行步数较长，后续可优化策略效率。",
            "qualityDisplaySeverity": "mild",
            "qualityDisplayDescription": "轨迹动作质量正常，但执行步数较长。",
            "qualityDisplayRecommendation": scripted_hint,
        }

    return {
        "qualityDisplayTier": "motion_warning",
        "qualityDisplayLabel": "存在警告",
        "qualityDisplayHint": (
            "检测到部分轨迹存在动作突变、旋转饱和或机械臂姿态异常，"
            "建议使用脚本专家策略重新生成或更换 seed demonstrations。"
        ),
        "qualityDisplaySeverity": "motion",
        "qualityDisplayDescription": "检测到部分轨迹存在动作突变、旋转饱和或机械臂姿态异常。",
        "qualityDisplayRecommendation": (
            "建议使用脚本专家策略重新生成或更换 seed demonstrations。"
        ),
    }


@dataclass
class DemoQuality:
    demo_key: str
    episode_length: int
    mean_arm_delta: float
    max_arm_delta: float
    mean_action_norm: float
    max_action_norm: float
    max_rot_cmd: float
    gripper_switch_count: int
    suspicious_wrist_flip_count: int
    suspicious_joint_jump_count: int
    low_quality: bool
    warnings: list[str] = field(default_factory=list)


def _load_actions(handle, demo_key: str) -> Optional[np.ndarray]:
    path = f"data/{demo_key}/actions"
    if path not in handle:
        return None
    arr = np.asarray(handle[path])
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return arr.astype(np.float32)


def _load_joint_pos(handle, demo_key: str) -> Optional[np.ndarray]:
    candidates = (
        f"data/{demo_key}/obs/joint_pos",
        f"data/{demo_key}/obs/agentview_joint_pos",
    )
    for path in candidates:
        if path in handle:
            arr = np.asarray(handle[path])
            if arr.ndim == 2 and arr.shape[0] >= 2:
                return arr.astype(np.float32)
    return None


def analyze_demo_actions(
    actions: np.ndarray,
    *,
    demo_key: str,
    joint_pos: Optional[np.ndarray] = None,
    arm_delta_warn: float = DEFAULT_ARM_DELTA_WARN,
    arm_delta_fail: float = DEFAULT_ARM_DELTA_FAIL,
) -> DemoQuality:
    arm = actions[:, ARM_ACTION_DIMS]
    arm_deltas = np.linalg.norm(np.diff(arm, axis=0), axis=1) if len(arm) > 1 else np.array([0.0])
    action_norms = np.linalg.norm(actions, axis=1)
    rot_cmds = np.abs(arm[:, 3:6]) if arm.shape[1] >= 6 else np.zeros((len(arm), 3))

    gripper = actions[:, GRIPPER_DIM] if actions.shape[1] > GRIPPER_DIM else None
    grip_switches = _count_gripper_switches(gripper)

    wrist_flip_count = 0
    if len(rot_cmds) > 1:
        rot_delta = np.linalg.norm(np.diff(rot_cmds, axis=0), axis=1)
        wrist_flip_count = int(np.sum(rot_delta > 0.45))

    joint_jump_count = 0
    if joint_pos is not None and len(joint_pos) > 1:
        j_delta = np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)
        joint_jump_count = int(np.sum(j_delta > 0.25))

    warnings: list[str] = []
    max_arm_delta = float(np.max(arm_deltas)) if len(arm_deltas) else 0.0
    mean_arm_delta = float(np.mean(arm_deltas)) if len(arm_deltas) else 0.0
    max_rot = float(np.max(rot_cmds)) if rot_cmds.size else 0.0

    if max_arm_delta >= arm_delta_fail:
        warnings.append(f"arm_action_delta_spike:{max_arm_delta:.3f}")
    elif max_arm_delta >= arm_delta_warn:
        warnings.append(f"arm_action_delta_high:{max_arm_delta:.3f}")

    if max_rot >= DEFAULT_ROT_SATURATION_WARN:
        warnings.append(f"rot_cmd_saturated:{max_rot:.3f}")

    if wrist_flip_count > 0:
        warnings.append(f"wrist_flip_steps:{wrist_flip_count}")

    if joint_jump_count > 0:
        warnings.append(f"joint_jump_steps:{joint_jump_count}")

    if len(actions) >= DEFAULT_EPISODE_LENGTH_WARN:
        warnings.append(f"episode_length_high:{len(actions)}")

    if grip_switches > DEFAULT_GRIPPER_SWITCH_WARN:
        warnings.append(f"gripper_switch_count:{grip_switches}")

    low_quality = max_arm_delta >= arm_delta_fail or (
        max_arm_delta >= arm_delta_warn and (wrist_flip_count > 0 or joint_jump_count > 0)
    )

    return DemoQuality(
        demo_key=demo_key,
        episode_length=int(len(actions)),
        mean_arm_delta=mean_arm_delta,
        max_arm_delta=max_arm_delta,
        mean_action_norm=float(np.mean(action_norms)),
        max_action_norm=float(np.max(action_norms)),
        max_rot_cmd=max_rot,
        gripper_switch_count=grip_switches,
        suspicious_wrist_flip_count=wrist_flip_count,
        suspicious_joint_jump_count=joint_jump_count,
        low_quality=low_quality,
        warnings=warnings,
    )


def analyze_hdf5_dataset(
    dataset_path: Path | str,
    *,
    generation_mode: Optional[str] = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        return {
            "datasetPath": str(dataset_path),
            "demoCount": 0,
            "validDemoCount": 0,
            "lowQualityDemoCount": 0,
            "qualityStatus": "failed",
            "qualityWarnings": ["dataset_file_missing"],
            "generationMode": generation_mode,
        }

    try:
        import h5py
    except ImportError:
        return {
            "datasetPath": str(dataset_path),
            "qualityStatus": "failed",
            "qualityWarnings": ["h5py_not_available"],
            "generationMode": generation_mode,
        }

    demos: list[DemoQuality] = []
    with h5py.File(dataset_path, "r") as handle:
        if "data" not in handle:
            return {
                "datasetPath": str(dataset_path),
                "demoCount": 0,
                "qualityStatus": "failed",
                "qualityWarnings": ["missing_data_group"],
                "generationMode": generation_mode,
            }
        for demo_key in sorted(handle["data"].keys()):
            actions = _load_actions(handle, demo_key)
            if actions is None:
                continue
            joint_pos = _load_joint_pos(handle, demo_key)
            demos.append(analyze_demo_actions(actions, demo_key=demo_key, joint_pos=joint_pos))

    if not demos:
        return {
            "datasetPath": str(dataset_path),
            "demoCount": 0,
            "qualityStatus": "failed",
            "qualityWarnings": ["no_demos_found"],
            "generationMode": generation_mode,
        }

    low_count = sum(1 for d in demos if d.low_quality)
    warn_count = sum(1 for d in demos if d.warnings)
    all_warnings: list[str] = []
    for d in demos:
        for w in d.warnings:
            tagged = f"{d.demo_key}:{w}"
            if tagged not in all_warnings:
                all_warnings.append(tagged)

    if low_count > 0:
        quality_status = "warning" if low_count < len(demos) else "failed"
    elif warn_count > 0:
        quality_status = "warning"
    else:
        quality_status = "passed"

    display = classify_quality_display(
        quality_status=quality_status,
        quality_warnings=all_warnings,
        suspicious_wrist_flip_count=int(np.sum([d.suspicious_wrist_flip_count for d in demos])),
        suspicious_joint_jump_count=int(np.sum([d.suspicious_joint_jump_count for d in demos])),
        generation_mode=generation_mode,
    )

    return {
        "datasetPath": str(dataset_path),
        "generationMode": generation_mode,
        "demoCount": len(demos),
        "validDemoCount": len(demos) - low_count,
        "lowQualityDemoCount": low_count,
        "meanActionDelta": float(np.mean([d.mean_arm_delta for d in demos])),
        "maxActionDelta": float(np.max([d.max_arm_delta for d in demos])),
        "meanActionNorm": float(np.mean([d.mean_action_norm for d in demos])),
        "maxActionNorm": float(np.max([d.max_action_norm for d in demos])),
        "meanEpisodeLength": float(np.mean([d.episode_length for d in demos])),
        "maxEpisodeLength": int(np.max([d.episode_length for d in demos])),
        "gripperSwitchCountMean": float(np.mean([d.gripper_switch_count for d in demos])),
        "suspiciousWristFlipCount": int(np.sum([d.suspicious_wrist_flip_count for d in demos])),
        "suspiciousJointJumpCount": int(np.sum([d.suspicious_joint_jump_count for d in demos])),
        "qualityStatus": quality_status,
        "qualityWarnings": all_warnings,
        **display,
        "demos": [
            {
                "demoKey": d.demo_key,
                "episodeLength": d.episode_length,
                "meanArmDelta": d.mean_arm_delta,
                "maxArmDelta": d.max_arm_delta,
                "maxRotCmd": d.max_rot_cmd,
                "gripperSwitchCount": d.gripper_switch_count,
                "lowQuality": d.low_quality,
                "warnings": d.warnings,
            }
            for d in demos
        ],
    }


def write_trajectory_quality_report(
    dataset_path: Path | str,
    report_path: Path | str,
    *,
    generation_mode: Optional[str] = None,
    behavior_report: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    report = analyze_hdf5_dataset(dataset_path, generation_mode=generation_mode)
    if behavior_report:
        try:
            from integrations.isaac_lab.scripts.stack_cube_expert_policy_behavior import (
                summarize_behavior_status,
            )

            behavior_status, behavior_warnings = summarize_behavior_status(behavior_report)
        except Exception:
            behavior_status = behavior_report.get("behaviorStatus", "unknown")
            behavior_warnings = behavior_report.get("behaviorWarnings", [])
        report["behaviorStatus"] = behavior_status
        report["behaviorWarnings"] = list(behavior_warnings)
        if behavior_status == "failed":
            report["qualityStatus"] = "failed"
            report["qualityDisplayTier"] = "failed"
            report["qualityDisplayLabel"] = "未通过（行为验证失败）"
            report["qualityDisplayDescription"] = "轨迹动作可能平滑，但抓取/堆叠行为验证未通过。"
            report["qualityDisplayRecommendation"] = "请修复专家策略后重新生成，不要用于训练。"
        elif behavior_status == "warning" and report.get("qualityStatus") == "passed":
            report["qualityStatus"] = "warning"
    out = Path(report_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
