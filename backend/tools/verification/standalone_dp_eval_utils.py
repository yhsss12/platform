"""Standalone DP eval helpers used by the manual verification runner."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

import numpy as np

STANDALONE_DP_OBS_SCHEMA: dict[str, Any] = {
    "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
    "low_dim_keys": ["robot0_joint_pos", "robot0_gripper_qpos"],
    "low_dim_dim": 9,
    "action_dim": 7,
}

STANDALONE_LOW_DIM_KEY_DIMS: dict[str, int] = {
    "robot0_joint_pos": 7,
    "robot0_gripper_qpos": 2,
}

REQUIRED_HDF5_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_joint_pos",
    "robot0_gripper_qpos",
)

DATASET_MISSING_JOINT_POS_MESSAGE = (
    "当前数据集没有 robot0_joint_pos，无法按关节角 + 夹爪 + 视频方案验证，"
    "需要重新生成包含 joint_pos 的数据集。"
)

# Keys where env may return more dims than training HDF5; take leading dims with warning.
LOW_DIM_KEYS_ALLOW_PREFIX_CROP = frozenset({"robot0_gripper_qpos"})


class StandaloneObsAdaptError(Exception):
    """Observation could not be adapted to checkpoint schema."""


def _norm_keys(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def inspect_hdf5_dataset_schema(hdf5_path: Path) -> dict[str, Any]:
    """Read-only HDF5 obs key/shape inspection for standalone schema validation."""
    import h5py

    path = hdf5_path.expanduser().resolve()
    info: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "errors": [],
        "num_demos": 0,
        "demo_obs": [],
        "has_robot0_joint_pos": False,
        "obs_keys_union": [],
        "obs_shapes_sample": {},
    }
    if not path.is_file():
        info["errors"].append(f"dataset not found: {path}")
        return info

    try:
        with h5py.File(path, "r") as handle:
            demos = sorted(k for k in handle["data"].keys() if k.startswith("demo_"))
            info["num_demos"] = len(demos)
            if not demos:
                info["errors"].append("no demos found in HDF5")
                return info

            union_keys: set[str] = set()
            demo_rows: list[dict[str, Any]] = []
            for demo in demos[:5]:
                obs = handle["data"][demo]["obs"]
                shapes = {str(key): tuple(obs[key].shape) for key in obs.keys()}
                demo_rows.append({"demo": demo, "obs_keys": sorted(shapes.keys()), "obs_shapes": shapes})
                union_keys.update(shapes.keys())
            info["demo_obs"] = demo_rows
            info["obs_keys_union"] = sorted(union_keys)
            if demo_rows:
                info["obs_shapes_sample"] = demo_rows[0]["obs_shapes"]

            has_joint = all("robot0_joint_pos" in handle["data"][demo]["obs"] for demo in demos)
            info["has_robot0_joint_pos"] = has_joint
            missing = [key for key in REQUIRED_HDF5_OBS_KEYS if key not in union_keys]
            if not has_joint:
                info["errors"].append(DATASET_MISSING_JOINT_POS_MESSAGE)
            for key in missing:
                if key == "robot0_joint_pos" and not has_joint:
                    continue
                info["errors"].append(f"missing required obs key in dataset: {key}")
            info["ok"] = has_joint and not missing
    except Exception as exc:
        info["errors"].append(f"failed to inspect HDF5: {exc}")
    return info


def validate_dataset_joint_pos_schema(dataset_paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "paths": [str(path) for path in dataset_paths],
        "datasets": [],
        "schema": STANDALONE_DP_OBS_SCHEMA,
    }
    if not dataset_paths:
        result["ok"] = False
        result["errors"].append("At least one dataset path is required")
        return result

    for path in dataset_paths:
        if not path.is_file():
            result["ok"] = False
            result["errors"].append(f"dataset not found: {path}")
            continue
        inspected = inspect_hdf5_dataset_schema(path)
        result["datasets"].append(inspected)
        if not inspected.get("ok"):
            result["ok"] = False
            result["errors"].extend(inspected.get("errors") or [])
    if not result["ok"] and not any(DATASET_MISSING_JOINT_POS_MESSAGE in err for err in result["errors"]):
        missing_joint = [item for item in result["datasets"] if not item.get("has_robot0_joint_pos")]
        if missing_joint:
            result["errors"].insert(0, DATASET_MISSING_JOINT_POS_MESSAGE)
    return result


def write_standalone_train_config_yaml(run_dir: Path, *, base_config_path: Path | None = None) -> Path:
    """Write standalone-only DP yaml with joint_pos + gripper + video schema."""
    import yaml

    base_path = base_config_path or Path(__file__).resolve().parents[3] / (
        "integrations/CableThreadingMVP/examples/cable_threading/dp_configs/cable_threading.yaml"
    )
    data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid base DP config: {base_path}")
    data["image_keys"] = list(STANDALONE_DP_OBS_SCHEMA["image_keys"])
    data["low_dim_keys"] = list(STANDALONE_DP_OBS_SCHEMA["low_dim_keys"])
    data["low_dim_dim"] = int(STANDALONE_DP_OBS_SCHEMA["low_dim_dim"])
    data["action_dim"] = int(STANDALONE_DP_OBS_SCHEMA["action_dim"])
    out = run_dir / "standalone_dp_joint_pos.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def resolve_expected_low_dim_key_dims(train_config: dict[str, Any]) -> dict[str, int]:
    low_dim_keys = _norm_keys(train_config.get("low_dim_keys"))
    if not low_dim_keys:
        raise StandaloneObsAdaptError("checkpoint train_config missing low_dim_keys")
    dims: dict[str, int] = {}
    for key in low_dim_keys:
        if key in STANDALONE_LOW_DIM_KEY_DIMS:
            dims[key] = STANDALONE_LOW_DIM_KEY_DIMS[key]
        else:
            raise StandaloneObsAdaptError(
                f"unsupported low_dim key in checkpoint: {key!r}; "
                f"standalone test only supports {list(STANDALONE_LOW_DIM_KEY_DIMS.keys())}"
            )
    expected_total = int(train_config.get("low_dim_dim") or sum(dims.values()))
    if sum(dims.values()) != expected_total:
        raise StandaloneObsAdaptError(
            f"low_dim_dim mismatch in train_config: keys sum to {sum(dims.values())}, "
            f"expected {expected_total}"
        )
    return dims


def list_low_dim_candidate_keys(raw_obs: dict[str, Any], image_keys: list[str]) -> dict[str, tuple[int, ...]]:
    image_set = set(image_keys)
    candidates: dict[str, tuple[int, ...]] = {}
    for key, value in raw_obs.items():
        if key in image_set:
            continue
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            candidates[str(key)] = tuple(int(dim) for dim in arr.shape)
    return candidates


def build_standalone_dp_observation(
    raw_env_obs: dict[str, Any],
    checkpoint_train_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Select checkpoint schema fields from env raw obs for standalone DP eval."""
    return adapt_raw_obs_for_checkpoint(raw_env_obs, checkpoint_train_config)


def adapt_raw_obs_for_checkpoint(
    raw_obs: dict[str, Any],
    train_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Build policy obs dict using only checkpoint-declared image/low_dim keys."""
    image_keys = _norm_keys(train_config.get("image_keys"))
    low_dim_keys = _norm_keys(train_config.get("low_dim_keys"))
    key_dims = resolve_expected_low_dim_key_dims(train_config)

    warnings: list[str] = []
    adapted: dict[str, Any] = {}
    per_key_shapes: dict[str, list[int]] = {}

    for key in image_keys:
        if key not in raw_obs:
            raise StandaloneObsAdaptError(f"missing image key: {key}")
        img = np.asarray(raw_obs[key])
        if img.ndim != 3:
            raise StandaloneObsAdaptError(
                f"image key {key} shape mismatch: expected 3D HWC, got {tuple(img.shape)}"
            )
        adapted[key] = img

    low_dim_parts: list[np.ndarray] = []
    for key in low_dim_keys:
        if key not in raw_obs:
            raise StandaloneObsAdaptError(f"missing low_dim key: {key}")
        expected_dim = key_dims[key]
        arr = np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)
        actual_dim = int(arr.shape[0])
        if actual_dim == expected_dim:
            part = arr
        elif actual_dim > expected_dim and key in LOW_DIM_KEYS_ALLOW_PREFIX_CROP:
            part = arr[:expected_dim]
            warnings.append(
                f"{key} actual dim {actual_dim}, expected {expected_dim}, using first {expected_dim} dims "
                "to match training schema"
            )
        else:
            raise StandaloneObsAdaptError(
                f"low_dim key {key} shape mismatch: expected ({expected_dim},), got ({actual_dim},)"
            )
        adapted[key] = part.astype(np.float32, copy=False)
        per_key_shapes[key] = [int(part.shape[0])]
        low_dim_parts.append(part)

    low_dim_concat = np.concatenate(low_dim_parts, axis=0) if low_dim_parts else np.zeros(0, dtype=np.float32)
    expected_low_dim_dim = int(train_config.get("low_dim_dim") or sum(key_dims.values()))
    diagnostics = {
        "image_keys": image_keys,
        "low_dim_keys": low_dim_keys,
        "per_key_shapes": per_key_shapes,
        "low_dim_concat_shape": [int(low_dim_concat.shape[0])],
        "expected_low_dim_dim": expected_low_dim_dim,
    }
    if int(low_dim_concat.shape[0]) != expected_low_dim_dim:
        raise StandaloneObsAdaptError(
            f"low_dim_concat shape mismatch: expected ({expected_low_dim_dim},), "
            f"got ({int(low_dim_concat.shape[0])},)"
        )
    return adapted, warnings, diagnostics


def log_standalone_eval_diagnostics(
    *,
    log_file: TextIO,
    checkpoint_path: Path,
    train_config: dict[str, Any],
    raw_obs: dict[str, Any] | None,
    selected_diagnostics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> None:
    image_keys = _norm_keys(train_config.get("image_keys"))
    low_dim_keys = _norm_keys(train_config.get("low_dim_keys"))
    lines = [
        f"checkpoint path: {checkpoint_path}",
        f"checkpoint image_keys: {image_keys}",
        f"checkpoint low_dim_keys: {low_dim_keys}",
        f"checkpoint low_dim_dim: {train_config.get('low_dim_dim')}",
        f"checkpoint action_dim: {train_config.get('action_dim')}",
        f"checkpoint n_obs_steps: {train_config.get('n_obs_steps')}",
        f"checkpoint n_action_steps: {train_config.get('n_action_steps')}",
    ]
    if raw_obs is not None:
        lines.append(f"env raw obs keys: {sorted(str(k) for k in raw_obs.keys())}")
        for key in ("robot0_joint_pos", "robot0_gripper_qpos"):
            if key in raw_obs:
                lines.append(f"env raw obs {key} shape: {tuple(np.asarray(raw_obs[key]).shape)}")
            else:
                lines.append(f"env raw obs {key} shape: <missing>")
        candidates = list_low_dim_candidate_keys(raw_obs, image_keys)
        lines.append(f"env raw low_dim candidate keys and shapes: {candidates}")
    if selected_diagnostics:
        lines.append(f"selected low_dim_keys: {selected_diagnostics.get('low_dim_keys')}")
        lines.append(f"selected per-key shapes: {selected_diagnostics.get('per_key_shapes')}")
        lines.append(f"final low_dim_concat shape: {selected_diagnostics.get('low_dim_concat_shape')}")
    if warnings:
        for item in warnings:
            lines.append(f"warning: {item}")
    for line in lines:
        log_file.write(line + "\n")
    log_file.flush()


def resolve_eval_env_kwargs(train_config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "robot": "Panda",
        "cable_model": "composite_cable",
        "grasp_mode": "attachment",
        "difficulty": "easy",
    }
    dataset_path = train_config.get("dataset")
    if not dataset_path:
        datasets = train_config.get("datasets")
        if isinstance(datasets, list) and datasets:
            dataset_path = datasets[0]
    if not dataset_path:
        return defaults
    path = Path(str(dataset_path)).expanduser()
    if not path.is_file():
        return defaults
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            raw = handle["data"].attrs.get("env_args", "{}")
        env_args = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
        if not isinstance(env_args, dict):
            return defaults
        return {
            "robot": str(env_args.get("robot") or defaults["robot"]),
            "cable_model": str(env_args.get("cable_model") or defaults["cable_model"]),
            "grasp_mode": str(env_args.get("grasp_mode") or defaults["grasp_mode"]),
            "difficulty": str(env_args.get("difficulty") or defaults["difficulty"]),
        }
    except Exception:
        return defaults


def run_standalone_dp_eval(
    *,
    checkpoint_path: Path,
    episodes: int,
    device: str,
    max_steps: int,
    output_dir: Path,
    cable_mvp_root: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Run in-process DP eval with checkpoint-aligned observation adapter."""
    if str(cable_mvp_root) not in sys.path:
        sys.path.insert(0, str(cable_mvp_root))

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_log_path = output_dir.parent / "eval.log"
    results_dir = output_dir
    eval_csv = results_dir / "eval.csv"

    result: dict[str, Any] = {
        "exit_code": 1,
        "log_path": str(eval_log_path),
        "backend": "standalone_inprocess",
        "episodes": [],
        "success_rate": 0.0,
        "ever_success_rate": 0.0,
        "aggregate": {},
        "errors": [],
        "warnings": [],
        "obs_validation_failed": False,
        "failure_step": None,
        "image_keys": STANDALONE_DP_OBS_SCHEMA["image_keys"],
        "low_dim_keys": STANDALONE_DP_OBS_SCHEMA["low_dim_keys"],
    }

    with eval_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("standalone DP eval (joint_pos + gripper + video observation adapter)\n\n")
        try:
            import torch
            from examples.cable_threading.dp_lab.policy_runtime import DiffusionPolicyAdapter
            from examples.cable_threading.utils import (
                aggregate_rows,
                clip_action,
                make_env,
                policy_eval_camera_kwargs,
                summarize_episode,
                write_results_csv,
            )
            from examples.cable_threading.attachment_controller import build_attachment_controller

            payload = torch.load(checkpoint_path, map_location="cpu")
            train_config = payload.get("train_config") if isinstance(payload, dict) else {}
            if not isinstance(train_config, dict):
                raise RuntimeError("checkpoint missing train_config")

            result["image_keys"] = _norm_keys(train_config.get("image_keys"))
            result["low_dim_keys"] = _norm_keys(train_config.get("low_dim_keys"))

            policy = DiffusionPolicyAdapter(checkpoint_path, device=device)
            env_kwargs = resolve_eval_env_kwargs(train_config)
            log_file.write(f"eval env kwargs: {env_kwargs}\n\n")
            env = make_env(
                horizon=max_steps,
                seed=seed,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                **env_kwargs,
                **policy_eval_camera_kwargs(),
            )

            probe_obs = env.reset()
            try:
                adapted_probe, probe_warnings, probe_diag = build_standalone_dp_observation(
                    probe_obs, train_config
                )
            except StandaloneObsAdaptError as exc:
                result["obs_validation_failed"] = True
                result["failure_step"] = "eval_obs_validation"
                result["errors"].append(str(exc))
                log_file.write(f"OBS_VALIDATION_FAILED: {exc}\n")
                log_standalone_eval_diagnostics(
                    log_file=log_file,
                    checkpoint_path=checkpoint_path,
                    train_config=train_config,
                    raw_obs=probe_obs,
                )
                env.close()
                return result

            log_standalone_eval_diagnostics(
                log_file=log_file,
                checkpoint_path=checkpoint_path,
                train_config=train_config,
                raw_obs=probe_obs,
                selected_diagnostics=probe_diag,
                warnings=probe_warnings,
            )
            result["warnings"].extend(probe_warnings)
            policy.reset()
            _ = adapted_probe

            rows: list[dict[str, Any]] = []
            for episode in range(int(episodes)):
                episode_seed = seed + episode
                obs = env.reset()
                attach_ctrl = None
                if getattr(env, "grasp_mode", "attachment") == "attachment":
                    attach_ctrl = build_attachment_controller(env, replay_mode="policy")
                    attach_ctrl.reset()
                policy.reset()
                total_reward = 0.0
                info_rows: list[dict[str, Any]] = []
                done = False
                log_file.write(f"\n[episode {episode}] start seed={episode_seed}\n")
                while not done and len(info_rows) < max_steps:
                    adapted_obs, step_warnings, _ = build_standalone_dp_observation(obs, train_config)
                    for warn in step_warnings:
                        if warn not in result["warnings"]:
                            result["warnings"].append(warn)
                    action = policy.act(adapted_obs)
                    action = clip_action(env, action)
                    if attach_ctrl is not None:
                        attach_ctrl.pre_step(action, info=info_rows[-1] if info_rows else None)
                    obs, reward, done, info = env.step(action)
                    total_reward += float(reward)
                    info_rows.append(dict(info))

                summary = summarize_episode(
                    info_rows,
                    env,
                    total_reward,
                    policy_name="diffusion_policy",
                    episode_index=episode,
                    seed=episode_seed,
                )
                if attach_ctrl is not None and hasattr(attach_ctrl, "attachment_stats"):
                    summary.update(attach_ctrl.attachment_stats())
                rows.append(summary)
                log_file.write(
                    f"[episode {episode}] final_success={summary.get('final_success')} steps={summary.get('steps')}\n"
                )

            env.close()
            stats = aggregate_rows(rows)
            write_results_csv(eval_csv, rows)
            aggregate_doc = {
                "task_name": "线缆穿杆",
                "requested_episodes": int(episodes),
                "total_episodes": len(rows),
                "completed_episodes": len(rows),
                "success_episodes": sum(1 for row in rows if row.get("final_success")),
                "final_success_rate": stats.get("final_success_rate", 0.0),
                "ever_success_rate": stats.get("ever_success_rate", 0.0),
                "success_rate": stats.get("final_success_rate", 0.0),
                "attachment_mode": "policy",
                **stats,
            }
            (results_dir / "aggregate_result.json").write_text(
                json.dumps(aggregate_doc, indent=2, default=str),
                encoding="utf-8",
            )
            (results_dir / "per_episode_results.json").write_text(
                json.dumps(rows, indent=2, default=str),
                encoding="utf-8",
            )
            results_payload = {
                "success_rate": stats.get("final_success_rate", 0.0),
                "ever_success_rate": stats.get("ever_success_rate", 0.0),
                "num_episodes": len(rows),
                "aggregate": stats,
                "episodes": rows,
            }
            (results_dir / "eval.results.json").write_text(
                json.dumps(results_payload, indent=2, default=str),
                encoding="utf-8",
            )

            result["exit_code"] = 0
            result["success_rate"] = float(stats.get("final_success_rate", 0.0) or 0.0)
            result["ever_success_rate"] = float(stats.get("ever_success_rate", 0.0) or 0.0)
            result["aggregate"] = aggregate_doc
            result["episodes"] = rows
            result["aggregate_path"] = str(results_dir / "aggregate_result.json")
            result["per_episode_path"] = str(results_dir / "per_episode_results.json")
            result["results_json_path"] = str(results_dir / "eval.results.json")
            log_file.write(
                f"\ncompleted episodes={len(rows)} success_rate={result['success_rate']:.4f}\n"
            )
        except Exception as exc:
            result["errors"].append(str(exc))
            result["failure_step"] = result.get("failure_step") or "eval"
            log_file.write(f"ERROR: {exc}\n")
            log_file.write(traceback.format_exc())
            log_file.flush()
    return result
