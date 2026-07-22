"""Observation schema helpers for robomimic policy eval (no robosuite import)."""

from __future__ import annotations

import numpy as np

POLICY_EVAL_IMAGE_CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
POLICY_EVAL_IMAGE_OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
POLICY_EVAL_CAMERA_SIZE = 256

# Display camera for eval workbench video — same stable view as data generation (agentview).
# eval_display_camera remains in the scene XML for optional overrides via --live-display-camera.
EVAL_DISPLAY_CAMERA = "agentview"
DEFAULT_EVAL_DISPLAY_CAMERA = EVAL_DISPLAY_CAMERA
EVAL_DISPLAY_FRAME_WIDTH = 1280
EVAL_DISPLAY_FRAME_HEIGHT = 720
EVAL_DISPLAY_FPS = 20


def policy_eval_camera_kwargs():
    """Camera setup for policy eval; images are resized to model image_size at inference."""
    return {
        "camera_names": list(POLICY_EVAL_IMAGE_CAMERA_NAMES),
        "camera_heights": POLICY_EVAL_CAMERA_SIZE,
        "camera_widths": POLICY_EVAL_CAMERA_SIZE,
    }


def _obs_shape_tuple(value):
    arr = np.asarray(value)
    return tuple(int(dim) for dim in arr.shape)


def validate_act_obs_schema(policy, obs):
    """Validate env obs against ACT checkpoint image/low_dim keys."""
    cfg = getattr(policy, "cfg", None)
    if cfg is None:
        return {
            "valid": True,
            "expectedObsKeys": [],
            "actualObsKeys": sorted(obs.keys()),
            "missingKeys": [],
            "shapeMismatchKeys": [],
            "errorMessage": "",
        }

    image_keys = [str(key) for key in list(cfg.image_keys)]
    if not image_keys:
        return {
            "valid": False,
            "expectedObsKeys": [],
            "actualObsKeys": sorted(str(key) for key in obs.keys()),
            "missingKeys": ["<camera obs>"],
            "shapeMismatchKeys": [],
            "errorMessage": "ACT 模型需要图像观测，但 checkpoint 未配置 camera_names / image_keys。",
        }

    expected_keys = image_keys + [str(key) for key in list(cfg.low_dim_keys)]
    actual_keys = sorted(str(key) for key in obs.keys())
    missing_keys = [key for key in expected_keys if key not in obs]
    if missing_keys and all(key not in obs for key in image_keys):
        error_message = (
            "ACT 模型需要图像观测，但当前评测环境未提供 camera obs："
            + ", ".join([k for k in missing_keys if k in image_keys])
            + "。"
        )
    elif missing_keys:
        error_message = "ACT 评测环境与模型观测不匹配：缺少 " + ", ".join(missing_keys) + "。"
    else:
        error_message = ""

    return {
        "valid": not missing_keys,
        "expectedObsKeys": expected_keys,
        "actualObsKeys": actual_keys,
        "missingKeys": missing_keys,
        "shapeMismatchKeys": [],
        "errorMessage": error_message,
    }


def validate_pi0_obs_schema(policy, obs):
    """Validate env obs against pi0 checkpoint camera / low-dim keys."""
    camera_keys = list(getattr(policy, "camera_keys", []) or [])
    low_dim_keys = list(getattr(policy, "low_dim_keys", []) or [])
    if not camera_keys:
        return {
            "valid": False,
            "expectedObsKeys": [],
            "actualObsKeys": sorted(str(key) for key in obs.keys()),
            "missingKeys": ["<camera obs>"],
            "shapeMismatchKeys": [],
            "errorMessage": "pi0 模型需要图像观测，但 checkpoint 未配置 camera_keys。",
        }
    missing_keys = [key for key in camera_keys if key not in obs]
    missing_keys.extend(key for key in low_dim_keys if key not in obs)
    if missing_keys:
        image_missing = [key for key in missing_keys if key in camera_keys]
        if image_missing:
            error_message = (
                "pi0 模型需要图像观测，但当前评测环境未提供 camera obs："
                + ", ".join(image_missing)
                + "。"
            )
        else:
            error_message = "pi0 评测环境与模型观测不匹配：缺少 " + ", ".join(missing_keys) + "。"
    else:
        error_message = ""
    expected_keys = camera_keys + low_dim_keys
    return {
        "valid": not missing_keys,
        "expectedObsKeys": [str(k) for k in expected_keys],
        "actualObsKeys": sorted(str(key) for key in obs.keys()),
        "missingKeys": missing_keys,
        "shapeMismatchKeys": [],
        "errorMessage": error_message,
    }


def _dp_expected_low_dim_dim(cfg) -> int:
    if hasattr(cfg, "resolved_low_dim_dim"):
        return int(cfg.resolved_low_dim_dim)
    low_dim_dim = getattr(cfg, "low_dim_dim", None)
    if low_dim_dim is not None and int(low_dim_dim) > 0:
        return int(low_dim_dim)
    return 9


def validate_diffusion_policy_obs_schema(policy, obs):
    """Validate env obs against DP checkpoint image/low_dim keys and shapes."""
    cfg = getattr(policy, "cfg", None)
    empty = {
        "valid": True,
        "expectedObsKeys": [],
        "actualObsKeys": sorted(obs.keys()),
        "missingKeys": [],
        "shapeMismatchKeys": [],
        "shapeWarnings": [],
        "errorMessage": "",
    }
    if cfg is None:
        return empty

    expected_keys = [str(key) for key in list(cfg.image_keys) + list(cfg.low_dim_keys)]
    actual_keys = sorted(str(key) for key in obs.keys())
    missing_keys = [key for key in expected_keys if key not in obs]

    shape_mismatch_keys: list[dict[str, object]] = []
    shape_warnings: list[dict[str, object]] = []
    expected_low_dim_dim = _dp_expected_low_dim_dim(cfg)
    actual_low_dim_dim = 0

    for key in list(cfg.low_dim_keys):
        if key not in obs:
            continue
        arr = np.asarray(obs[key])
        if arr.size == 0:
            shape_mismatch_keys.append(
                {"key": str(key), "expected": ">0 elements", "actual": arr.shape}
            )
            continue
        actual_low_dim_dim += int(arr.reshape(-1).shape[0])

    if not missing_keys and actual_low_dim_dim > 0 and actual_low_dim_dim != expected_low_dim_dim:
        shape_mismatch_keys.append(
            {
                "key": "<low_dim_concat>",
                "expected": (expected_low_dim_dim,),
                "actual": (actual_low_dim_dim,),
            }
        )

    image_size = int(getattr(cfg, "image_size", 128) or 128)
    for key in list(cfg.image_keys):
        if key not in obs:
            continue
        actual_shape = _obs_shape_tuple(obs[key])
        if len(actual_shape) != 3:
            shape_mismatch_keys.append(
                {
                    "key": str(key),
                    "expected": (image_size, image_size, 3),
                    "actual": actual_shape,
                }
            )
            continue
        if actual_shape[2] != 3:
            shape_mismatch_keys.append(
                {
                    "key": str(key),
                    "expected": (actual_shape[0], actual_shape[1], 3),
                    "actual": actual_shape,
                }
            )
            continue
        if actual_shape[0] != image_size or actual_shape[1] != image_size:
            shape_warnings.append(
                {
                    "key": str(key),
                    "expected": (image_size, image_size, 3),
                    "actual": actual_shape,
                    "note": "will_resize_at_inference",
                }
            )

    if missing_keys:
        error_message = (
            "Diffusion Policy 评测环境与模型观测不匹配：缺少 "
            + ", ".join(missing_keys)
            + "。"
        )
    elif shape_mismatch_keys:
        first = shape_mismatch_keys[0]
        error_message = (
            f"Diffusion Policy 评测环境与模型观测 shape 不匹配：{first['key']} "
            f"期望 {first['expected']}，实际 {first['actual']}。"
        )
    elif shape_warnings:
        first = shape_warnings[0]
        error_message = (
            f"Diffusion Policy 图像观测 {first['key']} 运行时 shape {first['actual']} "
            f"与模型 image_size {image_size} 不一致，推理时将 resize。"
        )
    else:
        error_message = ""

    return {
        "valid": not missing_keys and not shape_mismatch_keys,
        "expectedObsKeys": expected_keys,
        "actualObsKeys": actual_keys,
        "missingKeys": missing_keys,
        "shapeMismatchKeys": shape_mismatch_keys,
        "shapeWarnings": shape_warnings,
        "expectedConfig": {
            "action_dim": int(getattr(cfg, "action_dim", 0) or 0),
            "n_obs_steps": int(getattr(cfg, "n_obs_steps", 0) or 0),
            "n_action_steps": int(getattr(cfg, "n_action_steps", 0) or 0),
            "low_dim_dim": expected_low_dim_dim,
            "image_size": image_size,
        },
        "errorMessage": error_message,
    }


def validate_policy_obs_schema(policy, obs):
    """Validate env obs against robomimic policy keys/shapes before rollout."""
    expected_keys = [str(key) for key in getattr(policy, "obs_keys", [])]
    actual_keys = sorted(str(key) for key in obs.keys())
    missing_keys = [key for key in expected_keys if key not in obs]

    shape_meta = getattr(policy, "ckpt_dict", {}) or {}
    nested = shape_meta.get("shape_metadata") if isinstance(shape_meta.get("shape_metadata"), dict) else {}
    all_shapes = shape_meta.get("all_shapes") or nested.get("all_shapes") or {}

    shape_mismatch_keys = []
    shape_warnings = []
    for key in expected_keys:
        if key not in obs:
            continue
        expected_shape = all_shapes.get(key)
        if not expected_shape:
            continue
        actual_shape = _obs_shape_tuple(obs[key])
        normalized_expected = tuple(int(dim) for dim in expected_shape)
        if key.endswith("_image"):
            if len(actual_shape) == 3 and len(normalized_expected) == 3:
                expected_channels = (
                    normalized_expected[0]
                    if normalized_expected[0] in (1, 3, 4)
                    else normalized_expected[2]
                )
                expected_spatial = (
                    normalized_expected[1:3]
                    if normalized_expected[0] in (1, 3, 4)
                    else normalized_expected[0:2]
                )
                if actual_shape[2] != expected_channels:
                    shape_mismatch_keys.append(
                        {"key": key, "expected": normalized_expected, "actual": actual_shape}
                    )
                elif actual_shape[0:2] != expected_spatial:
                    shape_warnings.append(
                        {
                            "key": key,
                            "expected": normalized_expected,
                            "actual": actual_shape,
                            "note": "will_resize_at_inference",
                        }
                    )
                continue
        if actual_shape != normalized_expected:
            actual_size = int(np.asarray(obs[key]).size)
            expected_size = int(np.prod(normalized_expected))
            if actual_size == expected_size:
                shape_warnings.append(
                    {
                        "key": key,
                        "expected": normalized_expected,
                        "actual": actual_shape,
                        "note": "will_reshape_at_inference",
                    }
                )
            else:
                shape_mismatch_keys.append(
                    {"key": key, "expected": normalized_expected, "actual": actual_shape}
                )

    if missing_keys:
        error_message = (
            "策略评测环境与模型观测不匹配：缺少 "
            + ", ".join(missing_keys)
            + "。"
        )
    elif shape_mismatch_keys:
        first = shape_mismatch_keys[0]
        error_message = (
            f"策略评测环境与模型观测 shape 不匹配：{first['key']} "
            f"期望 {first['expected']}，实际 {first['actual']}。"
        )
    else:
        error_message = ""

    return {
        "valid": not missing_keys and not shape_mismatch_keys,
        "expectedObsKeys": expected_keys,
        "actualObsKeys": actual_keys,
        "missingKeys": missing_keys,
        "shapeMismatchKeys": shape_mismatch_keys,
        "shapeWarnings": shape_warnings,
        "errorMessage": error_message,
    }


def apply_obs_validation_failure(live_config, validation, write_live_status):
    """Persist obs schema validation failure without entering rollout."""
    missing = validation.get("missingKeys") or []
    shape_mismatch = validation.get("shapeMismatchKeys") or []
    error_message = validation.get("errorMessage") or "策略评测环境与模型观测不匹配。"
    live_config["status"] = "failed"
    live_config["failedStage"] = "obs_validation"
    live_config["failureReason"] = "obs_key_mismatch" if missing else "obs_shape_mismatch"
    live_config["errorMessage"] = error_message
    live_config["error"] = error_message
    live_config["expectedObsKeys"] = validation.get("expectedObsKeys")
    live_config["actualObsKeys"] = validation.get("actualObsKeys")
    live_config["missingKeys"] = missing
    live_config["shapeMismatchKeys"] = shape_mismatch
    live_config["logPaths"] = {
        "stdout": "logs/run.log",
        "stderr": "logs/run.log",
        "run": "logs/run.log",
    }
    write_live_status(live_config)
