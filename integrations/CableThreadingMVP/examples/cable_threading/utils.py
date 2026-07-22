import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

_ROBOT_CAMERA_NAMES = {"eye_in_hand", "robotview"}

logger = logging.getLogger(__name__)


def _create_step_metric_recorder(env, episode_index, live_config):
    if live_config is None or not live_config.get("record_step_metrics"):
        return None
    try:
        integrations_dir = Path(__file__).resolve().parents[3]
        common_root = integrations_dir / "common"
        if str(common_root) not in sys.path:
            sys.path.insert(0, str(common_root))
        from step_metrics import StepMetricRecorder
    except Exception as exc:
        logger.warning("step metrics import failed: %s", exc)
        return None

    output_root = Path(str(live_config.get("step_metrics_output_dir") or ""))
    if not output_root:
        return None

    control_freq = float(getattr(env, "control_freq", 20) or 20)
    dt = 1.0 / control_freq if control_freq > 0 else None
    episode_dir = output_root / f"episode_{int(episode_index) + 1:03d}"
    try:
        recorder = StepMetricRecorder(
            job_id=str(live_config.get("job_id") or "unknown"),
            episode_index=int(episode_index) + 1,
            output_dir=episode_dir,
            dt=dt,
            control_frequency_hz=control_freq,
            record_full_arrays=bool(live_config.get("record_step_metrics_full_arrays")),
            downsample=int(live_config.get("step_metrics_downsample") or 1),
            video_fps=live_config.get("video_fps") or live_config.get("eval_video_fps"),
        )
        recorder.start_episode()
        return recorder
    except Exception as exc:
        logger.warning("step metrics recorder init failed: %s", exc)
        return None


def _record_step_metric(recorder, step, action, reward, done, info=None, step_wall_sec=None):
    if recorder is None:
        return
    try:
        recorder.record_step(
            step=step,
            action=action,
            reward=reward,
            done=done,
            info=info,
            step_wall_sec=step_wall_sec,
        )
    except Exception as exc:
        logger.warning("step metric record failed: %s", exc)


def _finish_step_metric_recorder(recorder, summary, env):
    if recorder is None:
        return
    try:
        horizon = int(getattr(env, "horizon", 600) or 600)
        step_count = int(summary.get("steps") or 0)
        ep_summary = recorder.finish_episode(
            success=bool(summary.get("final_success")),
            timeout=bool(step_count >= horizon and not summary.get("final_success")),
        )
        summary["stepMetricsSummary"] = ep_summary
    except Exception as exc:
        logger.warning("step metrics finish failed: %s", exc)


DEFAULT_LIVE_FRAME_WIDTH = 1280
DEFAULT_LIVE_FRAME_HEIGHT = 720
DEFAULT_LIVE_DISPLAY_ASPECT_RATIO = "16:9"
DEFAULT_LIVE_JPEG_QUALITY = 90
DEFAULT_LIVE_WARMUP_STEPS = 10
DEFAULT_LIVE_SIM_FORWARD_WARMUP = 5
DEFAULT_LIVE_RENDER_WARMUP_COUNT = 5
DEFAULT_LIVE_REQUIRED_CONSECUTIVE_VALID = 3
LIVE_OBS_RESOLUTION_CANDIDATES = (1024, 768, 640)

_LIVE_FRAME_MIN_MEAN = 4.0
_LIVE_FRAME_MAX_MEAN = 251.0
_LIVE_DARK_PIXEL_THRESHOLD = 25.0
_LIVE_MIN_BRIGHT_PIXEL_RATIO = 0.08
# Dirty EGL/offscreen buffers: normal agentview frames have row/col diff ~1–3.
_LIVE_STRIPE_COL_DIFF_MIN = 25.0
_LIVE_STRIPE_ROW_DIFF_MIN = 20.0
_LIVE_STRIPE_STD_MIN = 45.0
_LIVE_DIRTY_COL_DIFF_MIN = 22.0
_LIVE_DIRTY_ROW_DIFF_MIN = 18.0
_LIVE_DIRTY_STD_MIN = 40.0
_LIVE_DIRTY_COLOR_STD_MIN = 35.0
_LIVE_DIRTY_MEAN_DIFF_MIN = 26.0
_LIVE_GARBLED_MEAN_DIFF_MIN = 28.0


def _resolve_robot_camera(name, robot_prefix="robot0"):
    if name in _ROBOT_CAMERA_NAMES:
        return f"{robot_prefix}_{name}"
    return name
from robosuite.utils.dlo.controller_adapter import RobosuiteControllerAdapter
from robosuite.utils.dlo.episode_schema import validate_transition_trajectories


DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "cable_end_pos",
    "pole_points",
    "endpoint_goal_pos",
    "attachment_state",
]

SUPPORTED_CABLE_MODELS = ["rmb", "flex", "composite_cable", "composite_soft"]
SUPPORTED_DIFFICULTIES = ["easy", "medium", "hard"]

from examples.cable_threading.obs_schema import (  # noqa: E402
    POLICY_EVAL_CAMERA_SIZE,
    POLICY_EVAL_IMAGE_CAMERA_NAMES,
    POLICY_EVAL_IMAGE_OBS_KEYS,
    apply_obs_validation_failure as _apply_obs_validation_failure,
    policy_eval_camera_kwargs,
    validate_policy_obs_schema,
)


def atomic_write_json(path, payload):
    """原子写入 JSON，避免读取到半文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def _live_render_dimensions(live_config):
    if not live_config:
        return None, None
    width = live_config.get("frame_width")
    height = live_config.get("frame_height")
    if width is None or height is None:
        return None, None
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _log_live_fallback(live_config, fallback_reason, detail=""):
    key = f"_fallback_{fallback_reason}_logged"
    if live_config is not None and live_config.get(key):
        return
    if live_config is not None:
        live_config[key] = True
    message = f"[live_frame] fallback_reason={fallback_reason}"
    if detail:
        message = f"{message} {detail}"
    logger.warning(message)


def live_camera_kwargs(live_config):
    """Live-only camera resolution — training env keeps CableThreading default 256×256."""
    if not live_config:
        return {}
    camera = live_config.get("camera", "agentview")
    height = int(live_config.get("frame_height", DEFAULT_LIVE_FRAME_HEIGHT))
    width = int(live_config.get("frame_width", DEFAULT_LIVE_FRAME_WIDTH))
    return {
        "camera_names": [camera],
        "camera_heights": height,
        "camera_widths": width,
    }


def build_expert_env_make_kwargs(*, live_enabled: bool, live_config, hdf5_out: bool) -> dict:
    """Build make_env kwargs for expert data generation without duplicate camera_names."""
    if not live_enabled and not hdf5_out:
        return {}
    live_kwargs = live_camera_kwargs(live_config) if live_enabled and live_config else {}
    if hdf5_out:
        # HDF5 export needs both policy cameras; live display uses a single camera name.
        live_kwargs.pop("camera_names", None)
        return {
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            **live_kwargs,
        }
    return dict(live_kwargs)


def apply_obs_validation_failure(live_config, validation):
    _apply_obs_validation_failure(live_config, validation, write_live_status)


def warmup_live_env(env, live_config=None):
    """Stabilize offscreen renderer after reset before saving live frames."""
    if live_config is None:
        return
    if hasattr(env, "sim") and hasattr(env.sim, "forward"):
        for _ in range(int(live_config.get("live_sim_forward_warmup", DEFAULT_LIVE_SIM_FORWARD_WARMUP))):
            env.sim.forward()
    if hasattr(env, "sim") and hasattr(env.sim, "step"):
        try:
            for _ in range(2):
                env.sim.step()
        except Exception:
            pass


def _prepare_obs_live_frame(obs_rgb, target_width, target_height, live_config=None):
    """Use native obs resolution when it matches target; resize only as temporary fallback."""
    h, w = obs_rgb.shape[:2]
    tw, th = int(target_width), int(target_height)
    if live_config is not None:
        live_config["live_obs_resolution"] = f"{w}x{h}"
    if w == tw and h == th:
        if live_config is not None:
            live_config["live_obs_resized"] = False
        return _normalize_live_frame_rgb(obs_rgb)
    if live_config is not None:
        live_config["live_obs_resized"] = True
        if w < tw or h < th:
            if not live_config.get("_live_obs_resolution_low_logged"):
                live_config["_live_obs_resolution_low_logged"] = True
                logger.warning(
                    "[live_frame] live_obs_resolution_low=true obs_shape=%sx%s target=%sx%s",
                    w,
                    h,
                    tw,
                    th,
                )
    return _normalize_live_frame_rgb(obs_rgb, tw, th)


def _normalize_live_frame_rgb(rgb, target_width=None, target_height=None):
    """Ensure uint8 H×W×3; optionally upscale to target 1:1 live resolution."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    rgb_u8 = arr[..., :3]
    if rgb_u8.dtype != np.uint8:
        if rgb_u8.max() <= 1.0:
            rgb_u8 = (rgb_u8 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb_u8 = rgb_u8.clip(0, 255).astype(np.uint8)
    if target_width and target_height:
        h, w = rgb_u8.shape[:2]
        tw, th = int(target_width), int(target_height)
        if w != tw or h != th:
            from PIL import Image

            rgb_u8 = np.asarray(
                Image.fromarray(rgb_u8).resize((tw, th), Image.Resampling.LANCZOS)
            )
    return rgb_u8


def _coerce_live_frame_u8(rgb):
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    channel = arr[..., :3]
    if channel.dtype != np.uint8:
        if channel.max() <= 1.0:
            channel = (channel * 255.0).clip(0, 255).astype(np.uint8)
        else:
            channel = channel.clip(0, 255).astype(np.uint8)
    return channel


def normalize_live_rgb_frame(rgb, *, source="mujoco"):
    """Convert MuJoCo / OpenGL RGB (bottom-left origin) to browser display order."""
    del source  # reserved for future source-specific rules; all current paths use mujoco GL
    arr = _coerce_live_frame_u8(rgb)
    if arr is None:
        return None
    return arr[..., :3][::-1].copy()


def stabilize_live_display_frame(display, live_config=None):
    """Align orientation with the previous persisted frame (EGL double-buffer guard)."""
    if live_config is None or display is None:
        return display
    prev = live_config.get("_last_display_frame")
    if isinstance(prev, np.ndarray) and prev.shape == display.shape:
        direct_err = float(np.mean(np.abs(display.astype(np.int16) - prev.astype(np.int16))))
        flip_err = float(np.mean(np.abs(display[::-1].astype(np.int16) - prev.astype(np.int16))))
        if flip_err + 2.0 < direct_err:
            display = display[::-1].copy()
    live_config["_last_display_frame"] = display.copy()
    return display


def is_valid_live_frame(rgb):
    """Strict validation — reject snow, black screen, noise bars before writing latest.jpg."""
    channel = _coerce_live_frame_u8(rgb)
    if channel is None:
        return False, "invalid_shape"

    gray = channel.mean(axis=2).astype(np.float32)
    if not np.isfinite(gray).all():
        return False, "non_finite"

    vmin = float(channel.min())
    vmax = float(channel.max())
    if vmin < 0 or vmax > 255:
        return False, "invalid_range"

    mean = float(gray.mean())
    if mean < _LIVE_FRAME_MIN_MEAN:
        return False, "all_black"
    if mean > _LIVE_FRAME_MAX_MEAN:
        return False, "all_white"

    std = float(gray.std())
    if std < 1.0:
        return False, "flat_color"

    dark_ratio = float((gray < _LIVE_DARK_PIXEL_THRESHOLD).mean())
    bright_ratio = float((gray > 40.0).mean())
    if bright_ratio < _LIVE_MIN_BRIGHT_PIXEL_RATIO:
        return False, "low_valid_pixel_ratio"

    h, w = gray.shape
    top_rows = max(1, h // 7)
    bot_rows = max(1, h // 7)
    top_band = gray[:top_rows, :]
    mid_band = gray[h // 4 : 3 * h // 4, :]
    bot_band = gray[-bot_rows:, :]

    top_std = float(top_band.std())
    mid_std = float(mid_band.std())
    bot_std = float(bot_band.std())
    mid_mean = float(mid_band.mean())
    top_mean = float(top_band.mean())
    bot_mean = float(bot_band.mean())

    color_std = float(channel.astype(np.float32).std(axis=(0, 1)).mean())
    row_diff = float(np.abs(gray[1:, :] - gray[:-1, :]).mean())
    col_diff = float(np.abs(gray[:, 1:] - gray[:, :-1]).mean())
    mean_diff = (row_diff + col_diff) / 2.0

    if col_diff > _LIVE_STRIPE_COL_DIFF_MIN and row_diff > _LIVE_STRIPE_ROW_DIFF_MIN and std > _LIVE_STRIPE_STD_MIN:
        return False, "vertical_stripe_noise"

    if (
        col_diff > _LIVE_DIRTY_COL_DIFF_MIN
        and row_diff > _LIVE_DIRTY_ROW_DIFF_MIN
        and std > _LIVE_DIRTY_STD_MIN
        and color_std > _LIVE_DIRTY_COLOR_STD_MIN
    ):
        return False, "dirty_offscreen_buffer"

    if col_diff > _LIVE_STRIPE_COL_DIFF_MIN and mean_diff > _LIVE_DIRTY_MEAN_DIFF_MIN:
        return False, "vertical_stripe_noise"

    if dark_ratio > 0.55 and mid_mean < 40.0 and (top_std > 42.0 or bot_std > 42.0):
        return False, "noise_bars_black_center"

    if dark_ratio > 0.75 and std > 35.0 and color_std > 30.0:
        return False, "mostly_dark_color_noise"

    if top_std > 48.0 and bot_std > 48.0 and mid_std < 22.0 and mid_mean < 38.0:
        return False, "edge_noise_dark_center"

    if std > 65.0 and mean_diff > _LIVE_GARBLED_MEAN_DIFF_MIN and bright_ratio < 0.22:
        return False, "garbled_noise"

    if std > 55.0 and bright_ratio < 0.15 and color_std > 45.0:
        return False, "random_color_noise"

    if mean < 55.0 and dark_ratio > 0.65 and (top_std > 40.0 or bot_std > 40.0):
        return False, "dark_scene_noise_bars"

    return True, None


_is_valid_live_frame = is_valid_live_frame


def _obs_agentview_rgb(obs, camera_name="agentview"):
    image_key = f"{camera_name}_image"
    if not isinstance(obs, dict) or image_key not in obs:
        return None
    img = np.asarray(obs[image_key])
    if img.ndim != 3 or img.shape[2] < 3:
        return None
    # Raw MuJoCo / OpenGL RGB — flip once in normalize_live_rgb_frame() before persist.
    return np.asarray(img[..., :3])


def _note_invalid_live_frame(live_config, reason, *, persist_debug=True):
    """Increment skip counters; optionally save debug JPEG under invalid_frames/."""
    if live_config is None:
        return
    reason_key = str(reason or "unknown")
    live_config["skipped_invalid_frame"] = int(live_config.get("skipped_invalid_frame", 0)) + 1
    reason_counts = live_config.setdefault("invalid_frame_reasons", {})
    reason_counts[reason_key] = int(reason_counts.get(reason_key, 0)) + 1
    if not live_config.get("has_valid_frame"):
        live_config["frame_status"] = "waiting_valid_frame"
    if persist_debug:
        logger.debug("[live_frame] rejected invalid_frame reason=%s skipped=%s", reason_key, live_config["skipped_invalid_frame"])


def _live_render_camera_name(live_config):
    """Resolve MuJoCo camera for sim.render display capture."""
    if live_config is None:
        return "agentview"
    if live_config.get("record_camera"):
        return str(live_config["record_camera"])
    if live_config.get("display_camera"):
        return str(live_config["display_camera"])
    return live_config.get("camera", "agentview")


def _mark_camera_fallback(live_config, *, actual_camera: str, fallback_source: str):
    if live_config is None:
        return
    requested = _live_render_camera_name(live_config)
    live_config["camera_fallback_used"] = True
    live_config["actual_record_camera"] = actual_camera
    live_config["camera_fallback_source"] = fallback_source
    warning = (
        f"录制相机从 {requested} fallback 到 {actual_camera} ({fallback_source})"
    )
    live_config["camera_warning"] = warning
    logger.warning("[live_frame] camera_fallback requested=%s actual=%s source=%s", requested, actual_camera, fallback_source)
    print(f"WARNING: {warning}")


def _use_display_sim_render(live_config):
    """Eval workbench video uses native sim.render, not policy obs images."""
    if live_config is None:
        return False
    if live_config.get("jobType") == "generate":
        return False
    if live_config.get("frame_source") == "obs_image":
        return False
    if live_config.get("frame_source") == "sim_render":
        return True
    if live_config.get("display_camera"):
        return True
    return live_config.get("jobType") == "evaluate" and live_config.get("save_frames")


def _live_sequence_frame_suffix(live_config):
    if _use_display_sim_render(live_config):
        return ".png"
    return ".jpg"


def warmup_live_render_buffer(env, live_config=None):
    """Flush offscreen RGB buffer after reset; loop until consecutive valid frames."""
    if live_config is None or not hasattr(env, "sim"):
        return
    camera = _live_render_camera_name(live_config)
    width, height = _live_output_dimensions(live_config)
    render_count = int(live_config.get("live_render_warmup_count", DEFAULT_LIVE_RENDER_WARMUP_COUNT))
    required = int(
        live_config.get("live_required_consecutive_valid", DEFAULT_LIVE_REQUIRED_CONSECUTIVE_VALID)
    )
    max_attempts = int(live_config.get("live_render_warmup_max", max(render_count * 6, 30)))
    invalid_count = 0
    consecutive = 0
    for _ in range(max_attempts):
        if hasattr(env, "sim") and hasattr(env.sim, "forward"):
            env.sim.forward()
        try:
            img = env.sim.render(camera_name=camera, width=width, height=height)
        except Exception as exc:
            logger.warning("[live_frame] render_warmup failed: %s", exc)
            consecutive = 0
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            invalid_count += 1
            consecutive = 0
            continue
        rgb = normalize_live_rgb_frame(arr[..., :3])
        if rgb is None:
            invalid_count += 1
            consecutive = 0
            continue
        valid, reason = is_valid_live_frame(rgb)
        if not valid:
            invalid_count += 1
            consecutive = 0
            _note_invalid_live_frame(live_config, reason or "render_warmup_invalid", persist_debug=False)
            continue
        consecutive += 1
        if consecutive >= required:
            live_config["has_valid_frame"] = True
            live_config["_consecutive_valid"] = required
            live_config["frame_status"] = "ready"
            break
    if invalid_count:
        live_config["renderWarmupInvalidCount"] = int(live_config.get("renderWarmupInvalidCount", 0)) + invalid_count
        logger.info(
            "[live_frame] render_warmup invalid_count=%s camera=%s size=%sx%s ready=%s",
            invalid_count,
            camera,
            width,
            height,
            bool(live_config.get("has_valid_frame")),
        )


def _save_invalid_live_frame(rgb, live_config, reason):
    if live_config is None:
        return
    frame_dir = live_config.get("frame_dir")
    if not frame_dir:
        return
    idx = int(live_config.get("_invalid_frame_count", 0)) + 1
    live_config["_invalid_frame_count"] = idx
    # Invalid frames are diagnostic samples, not a frame-by-frame archive.  A
    # persistent bad offscreen buffer previously produced hundreds of large
    # JPEGs per evaluation.
    max_samples = max(0, int(live_config.get("invalid_frame_debug_max", 3) or 0))
    if idx > max_samples:
        return
    invalid_dir = Path(frame_dir) / "invalid_frames"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(reason))[:48]
    target = invalid_dir / f"frame_{idx:06d}_{safe_reason}.jpg"
    quality = int(live_config.get("jpeg_quality", DEFAULT_LIVE_JPEG_QUALITY))
    try:
        _write_jpeg_atomic(rgb, target, jpeg_quality=quality)
    except Exception as exc:
        logger.warning("[live_frame] failed to save invalid frame: %s", exc)


def _live_output_dimensions(live_config):
    frame_width, frame_height = _live_render_dimensions(live_config)
    if frame_width is None or frame_height is None:
        frame_width = DEFAULT_LIVE_FRAME_WIDTH
        frame_height = DEFAULT_LIVE_FRAME_HEIGHT
    return int(frame_width), int(frame_height)


def try_acquire_sim_render_frame(env, camera_name, live_config=None):
    """Render display frame at native output resolution via sim.render."""
    if env is None or not hasattr(env, "sim"):
        return None, None, "no_sim"
    width, height = _live_output_dimensions(live_config)
    try:
        img = env.sim.render(camera_name=camera_name, width=width, height=height)
    except Exception as exc:
        _log_live_fallback(
            live_config,
            "sim_render_failed",
            detail=f"camera={camera_name} error={exc}",
        )
        return None, None, "sim_render_failed"

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None, None, "sim_render_invalid_shape"
    rgb = normalize_live_rgb_frame(arr[..., :3])
    if rgb is None:
        return None, None, "sim_render_invalid_shape"
    valid, reason = is_valid_live_frame(rgb)
    if not valid:
        _note_invalid_live_frame(live_config, reason, persist_debug=False)
        _save_invalid_live_frame(rgb, live_config, reason)
        _log_live_fallback(
            live_config,
            "invalid_sim_render",
            detail=f"invalid_frame_reason={reason} camera={camera_name} shape={rgb.shape[:2]}",
        )
        return None, None, reason

    if live_config is not None:
        live_config["live_obs_resized"] = False
        live_config["live_obs_resolution"] = f"{width}x{height}"
        live_config["live_display_resolution"] = f"{width}x{height}"
    return rgb, "sim_render", None


def try_acquire_obs_live_frame(obs, camera_name="agentview", live_config=None):
    """Extract obs agentview frame only — no sim.render, no last-frame reuse."""
    image_key = f"{camera_name}_image"
    obs_rgb = _obs_agentview_rgb(obs, camera_name=camera_name)
    if obs_rgb is None:
        return None, None, "missing_obs_image"

    display_rgb = normalize_live_rgb_frame(obs_rgb)
    if display_rgb is None:
        return None, None, "missing_obs_image"

    valid_raw, raw_reason = is_valid_live_frame(display_rgb)
    if not valid_raw:
        _note_invalid_live_frame(live_config, raw_reason, persist_debug=False)
        _save_invalid_live_frame(display_rgb, live_config, raw_reason)
        _log_live_fallback(
            live_config,
            "invalid_obs_image",
            detail=(
                f"invalid_frame_reason={raw_reason} image_key={image_key} "
                f"shape={display_rgb.shape[:2]} source=obs_image"
            ),
        )
        return None, None, raw_reason

    out_w, out_h = _live_output_dimensions(live_config)
    frame = _prepare_obs_live_frame(display_rgb, out_w, out_h, live_config=live_config)
    if frame is None:
        return None, None, "normalize_failed"

    if live_config is not None and not live_config.get("live_obs_resized", False):
        valid_out, out_reason = (True, None)
    else:
        valid_out, out_reason = is_valid_live_frame(frame)
    if not valid_out:
        _note_invalid_live_frame(live_config, out_reason, persist_debug=False)
        _save_invalid_live_frame(frame, live_config, out_reason)
        _log_live_fallback(
            live_config,
            "invalid_obs_resized",
            detail=f"invalid_frame_reason={out_reason} shape={frame.shape[:2]}",
        )
        return None, None, out_reason

    return frame, "obs_image", None


def capture_live_frame_rgb(env, obs, camera_name="agentview", live_config=None):
    """Backward-compatible wrapper — obs-only stable path."""
    frame, source, _reason = try_acquire_obs_live_frame(obs, camera_name=camera_name, live_config=live_config)
    if frame is not None and live_config is not None:
        live_config["live_frame_source"] = source
    return frame


def _write_jpeg_atomic(rgb_array, target_path, jpeg_quality=DEFAULT_LIVE_JPEG_QUALITY):
    """Write JPEG via temp file + atomic replace."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    arr = np.asarray(rgb_array).astype(np.uint8)
    quality = int(jpeg_quality)
    try:
        from PIL import Image

        Image.fromarray(arr).save(tmp_path, format="JPEG", quality=quality)
    except Exception:
        import imageio.v3 as iio

        iio.imwrite(tmp_path, arr, quality=quality)
    tmp_path.replace(target_path)


def save_live_jpeg(rgb_array, live_frame_dir, jpeg_quality=DEFAULT_LIVE_JPEG_QUALITY):
    """保存 latest.jpg（先写 tmp 再 rename）。"""
    live_dir = Path(live_frame_dir)
    _write_jpeg_atomic(rgb_array, live_dir / "latest.jpg", jpeg_quality=jpeg_quality)


def save_live_sequence_frame(
    rgb_array,
    frames_dir,
    sequence_index,
    jpeg_quality=DEFAULT_LIVE_JPEG_QUALITY,
    *,
    suffix=".jpg",
):
    """保存 frames/frame_NNNNNN.{jpg|png}（先写 tmp 再 rename）。"""
    frames_path = Path(frames_dir)
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    frame_name = f"frame_{int(sequence_index):06d}{ext}"
    target = frames_path / frame_name
    if ext.lower() == ".png":
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
        arr = np.asarray(rgb_array).astype(np.uint8)
        try:
            from PIL import Image

            Image.fromarray(arr).save(tmp_path, format="PNG")
        except Exception:
            import imageio.v3 as iio

            iio.imwrite(tmp_path, arr)
        tmp_path.replace(target_path)
    else:
        _write_jpeg_atomic(rgb_array, target, jpeg_quality=jpeg_quality)


def persist_live_rgb(rgb, live_config):
    """Write latest.jpg / frames only for validated frames; never overwrite with invalid."""
    display = _coerce_live_frame_u8(rgb)
    if display is None:
        return False
    display = display[..., :3].copy()
    # acquire_* paths already normalized orientation; stabilize against EGL buffer alternation.
    display = stabilize_live_display_frame(display, live_config)
    valid, invalid_reason = is_valid_live_frame(display)
    if not valid:
        _note_invalid_live_frame(live_config, invalid_reason)
        _save_invalid_live_frame(display, live_config, invalid_reason)
        _log_live_fallback(
            live_config,
            "skip_invalid_persist",
            detail=f"invalid_frame_reason={invalid_reason}",
        )
        return False

    quality = int(live_config.get("jpeg_quality", DEFAULT_LIVE_JPEG_QUALITY))
    save_live_jpeg(display, live_config["frame_dir"], jpeg_quality=quality)
    live_config["frame_count"] = int(live_config.get("frame_count", 0)) + 1
    live_config["has_valid_frame"] = True
    live_config["frame_status"] = "ready"
    if live_config.get("save_frames"):
        if live_config.get("multi_episode_video"):
            next_index = int(live_config.get("global_saved_frame_count", 0)) + 1
            live_config["global_saved_frame_count"] = next_index
        else:
            next_index = int(live_config.get("saved_frame_count", 0)) + 1
        if next_index == 1:
            camera = _live_render_camera_name(live_config)
            logger.info(
                "[live_frame] persist record_camera=%s source=%s episode=%s",
                camera,
                live_config.get("live_frame_source"),
                live_config.get("episode"),
            )
        save_live_sequence_frame(
            display,
            live_config.get("frames_dir") or str(Path(live_config["frame_dir"]) / "frames"),
            next_index,
            jpeg_quality=quality,
            suffix=_live_sequence_frame_suffix(live_config),
        )
        live_config["saved_frame_count"] = next_index
    return True


def _warmup_live_after_reset(env, obs, live_config):
    """Stabilize live capture after env.reset(); EGL needs extra sim steps."""
    if live_config is None:
        return obs
    warmup_live_env(env, live_config)
    warmup_live_render_buffer(env, live_config)
    if live_config.get("has_valid_frame"):
        return obs
    egl_mode = os.environ.get("MUJOCO_GL", "").lower() == "egl"
    max_rounds = 120 if egl_mode else 12
    for _ in range(max_rounds):
        if live_config.get("has_valid_frame"):
            break
        if hasattr(env, "sim") and hasattr(env.sim, "forward"):
            env.sim.forward()
        warmup_live_render_buffer(env, live_config)
        if live_config.get("has_valid_frame"):
            break
        try:
            action = np.zeros(env.action_dim, dtype=float)
            obs, _, _, _ = env.step(action)
        except Exception:
            break
        live_config["step"] = int(live_config.get("step", 0)) + 1
        maybe_capture_and_persist_live_frame(env, obs, live_config)
    return obs


def _episode_saved_frame_count(live_config) -> int:
    if live_config.get("multi_episode_video"):
        start = int(live_config.get("episode_frame_start", 0))
        current = int(live_config.get("global_saved_frame_count", 0))
        return max(0, current - start)
    return int(live_config.get("saved_frame_count", 0))


def _ensure_episode_live_frames(env, obs, live_config):
    """Best-effort tail capture when rolling capture produced no saved frames."""
    if live_config is None or not live_config.get("save_frames"):
        return
    if _episode_saved_frame_count(live_config) > 0:
        return
    frame_every = max(1, int(live_config.get("frame_every", 5)))
    for _ in range(16):
        if _episode_saved_frame_count(live_config) > 0:
            return
        if hasattr(env, "sim") and hasattr(env.sim, "forward"):
            env.sim.forward()
        live_config["step"] = int(live_config.get("step", 0)) + frame_every
        maybe_capture_and_persist_live_frame(env, obs, live_config)
        frame, _, _ = acquire_live_frame(env, obs, live_config)
        if frame is not None and persist_live_rgb(frame, live_config):
            return
        try:
            action = np.zeros(env.action_dim, dtype=float)
            obs, _, _, _ = env.step(action)
        except Exception:
            break


def acquire_live_frame(env, obs, live_config):
    """Capture display frame using the fixed record camera (no silent multi-camera switch)."""
    record_camera = _live_render_camera_name(live_config)
    allow_fallback = bool(live_config.get("allow_camera_fallback"))

    if _use_display_sim_render(live_config):
        frame, source, reason = try_acquire_sim_render_frame(
            env, record_camera, live_config=live_config
        )
        if frame is not None:
            if live_config is not None:
                live_config["live_frame_source"] = source or "sim_render"
                live_config["actual_record_camera"] = record_camera
            return frame, source or "sim_render", reason

        if allow_fallback:
            obs_camera = live_config.get("camera", "agentview") if live_config else "agentview"
            frame, source, reason = try_acquire_obs_live_frame(
                obs,
                camera_name=obs_camera,
                live_config=live_config,
            )
            if frame is not None:
                _mark_camera_fallback(
                    live_config,
                    actual_camera=obs_camera,
                    fallback_source=source or "obs_image",
                )
                return frame, source, reason
        return None, None, reason

    obs_camera = live_config.get("camera", "agentview") if live_config else "agentview"
    return try_acquire_obs_live_frame(
        obs,
        camera_name=obs_camera,
        live_config=live_config,
    )


# Shared aliases for generate + evaluate pipelines
acquire_display_frame = acquire_live_frame
write_live_frame_if_valid = persist_live_rgb
render_warmup = warmup_live_render_buffer


def maybe_capture_and_persist_live_frame(env, obs, live_config):
    """Stable live write rhythm: warmup → consecutive valid obs frames → persist."""
    if live_config is None:
        return

    step = int(live_config.get("step", 0))
    warmup_steps = int(live_config.get("live_warmup_steps", DEFAULT_LIVE_WARMUP_STEPS))
    if live_config.get("has_valid_frame"):
        warmup_steps = 0
    required_consecutive = int(
        live_config.get("live_required_consecutive_valid", DEFAULT_LIVE_REQUIRED_CONSECUTIVE_VALID)
    )
    if live_config.get("has_valid_frame"):
        required_consecutive = 1

    if step <= warmup_steps:
        live_config["frame_status"] = "warming_up"
        return

    frame_every = max(1, int(live_config.get("frame_every", 5)))
    if step % frame_every != 0:
        return

    frame, source, invalid_reason = acquire_live_frame(env, obs, live_config)

    if frame is None:
        live_config["_consecutive_valid"] = 0
        return

    consecutive = int(live_config.get("_consecutive_valid", 0)) + 1
    live_config["_consecutive_valid"] = consecutive
    live_config["live_frame_source"] = source

    if consecutive < required_consecutive:
        live_config["frame_status"] = "warming_up"
        return

    persist_live_rgb(frame, live_config)


def _live_video_status_keys(live_config):
    """Return internal + JSON field names for live video synthesis status."""
    if live_config.get("jobType") == "evaluate":
        return {
            "status": "eval_video_status",
            "video": "eval_video",
            "exists": "eval_video_exists",
            "size_bytes": "eval_video_size_bytes",
            "error": "eval_video_error",
            "json_status": "evalVideoStatus",
            "json_video": "evalVideo",
            "json_exists": "evalVideoExists",
            "json_size_bytes": "evalVideoSizeBytes",
        }
    return {
        "status": "generate_video_status",
        "video": "generate_video",
        "exists": "generate_video_exists",
        "size_bytes": "generate_video_size_bytes",
        "error": "generate_video_error",
        "json_status": "generateVideoStatus",
        "json_video": "generateVideo",
        "json_exists": "generateVideoExists",
        "json_size_bytes": "generateVideoSizeBytes",
    }


def build_live_status_payload(live_config):
    video_keys = _live_video_status_keys(live_config)
    payload = {
        "status": live_config.get("status", "running"),
        "jobType": live_config.get("jobType", "generate"),
        "taskType": "cable_threading",
        "episode": live_config.get("episode", 0),
        "episodes": live_config.get("episodes", 1),
        "step": live_config.get("step", 0),
        "horizon": live_config.get("horizon", 600),
        "frameCount": live_config.get("frame_count", 0),
        "savedFrameCount": live_config.get(
            "saved_frame_count", live_config.get("frame_count", 0)
        ),
        "latestFrame": "latest.jpg",
        "frameStatus": live_config.get("frame_status", "warming_up"),
        "skippedInvalidFrame": int(live_config.get("skipped_invalid_frame", 0)),
        "invalidFrameReasons": dict(live_config.get("invalid_frame_reasons") or {}),
        "renderWarmupInvalidCount": int(live_config.get("renderWarmupInvalidCount", 0)),
        "hasValidFrame": bool(live_config.get("has_valid_frame", False)),
        "liveFrameSource": live_config.get("live_frame_source"),
        "liveObsResolution": live_config.get("live_obs_resolution"),
        "liveDisplayResolution": live_config.get("live_display_resolution"),
        "liveObsResized": bool(live_config.get("live_obs_resized", False)),
        "displayCamera": live_config.get("display_camera"),
        "recordCamera": live_config.get("record_camera") or live_config.get("display_camera"),
        "actualRecordCamera": live_config.get("actual_record_camera"),
        "cameraFallbackUsed": bool(live_config.get("camera_fallback_used", False)),
        "cameraWarning": live_config.get("camera_warning"),
        "videoResolution": live_config.get("videoResolution"),
        "videoFps": live_config.get("videoFps"),
        "videoPath": live_config.get("videoPath"),
        "browserVideoPath": live_config.get("browserVideoPath"),
        "videoStatus": live_config.get("videoStatus"),
        "phase": live_config.get("phase", ""),
        "successSoFar": live_config.get("success_so_far"),
        "finalSuccessRate": live_config.get("final_success_rate"),
        "successfulEpisodes": live_config.get("successful_episodes", 0),
        "error": live_config.get("error"),
        video_keys["json_status"]: live_config.get(video_keys["status"]),
        video_keys["json_video"]: live_config.get(video_keys["video"]),
        video_keys["json_exists"]: bool(live_config.get(video_keys["exists"], False)),
        video_keys["json_size_bytes"]: live_config.get(video_keys["size_bytes"], 0),
    }
    if live_config.get("jobType") != "evaluate":
        payload.update(
            {
                "generateVideoStatus": live_config.get("generate_video_status"),
                "generateVideo": live_config.get("generate_video"),
                "generateVideoExists": bool(
                    live_config.get("generate_video_exists", False)
                ),
                "generateVideoSizeBytes": live_config.get(
                    "generate_video_size_bytes", 0
                ),
            }
        )
    else:
        payload.update(
            {
                "evalBrowserVideo": live_config.get("eval_browser_video"),
                "evalBrowserVideoExists": bool(
                    live_config.get("eval_browser_video_exists", False)
                ),
                "evalBrowserVideoSizeBytes": live_config.get(
                    "eval_browser_video_size_bytes", 0
                ),
            }
        )
    for key in (
        "episodeSuccess",
        "savedDataset",
        "savedHdf5",
        "savedManifest",
        "savedCsv",
        "savedFailures",
    ):
        if key in live_config and live_config[key] is not None:
            payload[key] = live_config[key]
    return payload


def write_live_status(live_config):
    status_path = live_config.get("status_path")
    if not status_path:
        return
    atomic_write_json(Path(status_path), build_live_status_payload(live_config))


EXPERT_TO_TIMELINE_PHASE = {
    "approach_above_end": ("approach_above_end", "接近线缆末端"),
    "attach": ("attach", "建立线缆连接"),
    "pull_through": ("pull_through", "牵引线缆穿过杆间间隙"),
    "release": ("release", "释放线缆"),
    "settle_wait": ("settle_wait", "等待稳定"),
}


def _live_timeline_frame_index(live_config):
    if live_config.get("save_frames"):
        return int(live_config.get("saved_frame_count", 0))
    return int(live_config.get("frame_count", 0))


def record_live_timeline_event(live_config, phase_key, label):
    """Record first entry into a timeline phase during generate rollout."""
    if not live_config.get("timeline_path"):
        return
    episode = int(live_config.get("episode", 0))
    seen = live_config.setdefault("timeline_seen", set())
    seen_key = (episode, phase_key)
    if seen_key in seen:
        return
    seen.add(seen_key)
    frame_index = _live_timeline_frame_index(live_config)
    fps = float(live_config.get("video_fps", 20) or 20)
    event = {
        "episode": episode,
        "frameIndex": frame_index,
        "step": int(live_config.get("step", 0)),
        "timeSec": round(frame_index / fps, 3) if fps else 0.0,
        "phase": phase_key,
        "label": label,
    }
    live_config.setdefault("timeline_events", []).append(event)


def maybe_record_live_timeline_expert_phase(live_config, expert_phase_name):
    mapped = EXPERT_TO_TIMELINE_PHASE.get(expert_phase_name)
    if not mapped:
        return
    record_live_timeline_event(live_config, mapped[0], mapped[1])


def write_live_timeline(live_config):
    timeline_path = live_config.get("timeline_path")
    if not timeline_path:
        return
    events = live_config.get("timeline_events", [])
    if live_config.get("jobType") == "evaluate":
        atomic_write_json(Path(timeline_path), events)
        return
    payload = {
        "events": events,
        "videoFps": int(live_config.get("video_fps", 20)),
    }
    atomic_write_json(Path(timeline_path), payload)


def _browser_video_path(source_path):
    path = Path(source_path)
    return path.with_name(f"{path.stem}.browser{path.suffix}")


def _ensure_browser_mp4(source_path):
    """Create browser-friendly MP4 (H.264/yuv420p, faststart) beside source."""
    source = Path(source_path)
    if not source.is_file():
        return None
    browser_path = _browser_video_path(source)
    if browser_path.is_file() and browser_path.stat().st_size > 0:
        if browser_path.stat().st_mtime >= source.stat().st_mtime:
            return browser_path

    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    tmp_path = browser_path.with_suffix(".tmp.mp4")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            "fast",
            "-crf",
            "20",
            str(tmp_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
            if result.returncode == 0 and tmp_path.is_file() and tmp_path.stat().st_size > 0:
                tmp_path.replace(browser_path)
                return browser_path
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("[live_video] browser transcode failed: %s", exc)
        finally:
            if tmp_path.is_file():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    try:
        shutil.copy2(source, browser_path)
        return browser_path
    except OSError as exc:
        logger.warning("[live_video] browser copy failed: %s", exc)
        return None


def clear_live_saved_frames(live_config):
    """Clear encoded frame sequence between episodes."""
    frames_dir = live_config.get("frames_dir")
    if not frames_dir:
        frames_dir = str(Path(live_config.get("frame_dir", ".")) / "frames")
    frames_path = Path(frames_dir)
    if frames_path.is_dir():
        for pattern in ("frame_*.png", "frame_*.jpg"):
            for frame_path in frames_path.glob(pattern):
                try:
                    frame_path.unlink()
                except OSError:
                    pass
    live_config["saved_frame_count"] = 0
    live_config["frame_count"] = 0


def synthesize_live_video(live_config):
    """Encode saved frames/ sequence into MP4; updates live_config status fields."""
    video_out = live_config.get("video_out")
    if not video_out:
        return

    frames_dir = live_config.get("frames_dir")
    if not frames_dir:
        frames_dir = str(Path(live_config.get("frame_dir", ".")) / "frames")

    fps = int(live_config.get("video_fps", 20))
    video_keys = _live_video_status_keys(live_config)
    live_config[video_keys["status"]] = "encoding"
    write_live_status(live_config)

    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_path.glob("frame_*.jpg"))
    if not frame_files:
        latest = Path(live_config.get("frame_dir", frames_path.parent)) / "latest.jpg"
        if latest.is_file():
            import shutil

            frames_path.mkdir(parents=True, exist_ok=True)
            fallback = frames_path / "frame_000001.jpg"
            shutil.copy2(latest, fallback)
            frame_files = [fallback]
    if not frame_files:
        live_config[video_keys["status"]] = "no_frames"
        live_config[video_keys["video"]] = None
        live_config[video_keys["exists"]] = False
        live_config[video_keys["size_bytes"]] = 0
        write_live_status(live_config)
        return

    video_path = Path(video_out)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio

        first_frame = iio.imread(frame_files[0])
        live_config["videoResolution"] = f"{first_frame.shape[1]}x{first_frame.shape[0]}"

        writer = iio.get_writer(
            str(video_path),
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
        try:
            for frame_path in frame_files:
                writer.append_data(iio.imread(frame_path))
        finally:
            writer.close()

        size_bytes = video_path.stat().st_size
        live_config[video_keys["status"]] = "completed"
        live_config[video_keys["video"]] = str(video_path)
        live_config[video_keys["exists"]] = True
        live_config[video_keys["size_bytes"]] = size_bytes
        live_config["videoPath"] = str(video_path)
        live_config["videoStatus"] = "available"
        live_config["videoFps"] = fps

        browser_path = _ensure_browser_mp4(video_path)
        if browser_path is not None:
            live_config["eval_browser_video"] = str(browser_path)
            live_config["eval_browser_video_exists"] = True
            live_config["eval_browser_video_size_bytes"] = browser_path.stat().st_size
            live_config["browserVideoPath"] = str(browser_path)
        else:
            live_config["eval_browser_video_exists"] = False
            live_config["browserVideoPath"] = str(video_path)
    except Exception as exc:
        live_config[video_keys["status"]] = "failed"
        live_config[video_keys["video"]] = None
        live_config[video_keys["exists"]] = False
        live_config[video_keys["size_bytes"]] = 0
        live_config[video_keys["error"]] = str(exc)
        live_config["videoStatus"] = "failed"
    write_live_status(live_config)


def make_env(
    env_name="CableThreading",
    robot="UR5e",
    horizon=400,
    seed=None,
    has_renderer=False,
    has_offscreen_renderer=False,
    render_camera="frontview",
    cable_model="rmb",
    grasp_mode="attachment",
    difficulty="easy",
    use_camera_obs=False,
    controller_configs=None,
    **kwargs,
):
    controller_config = controller_configs or load_composite_controller_config(robot=robot)
    render_camera = _resolve_robot_camera(render_camera)
    return suite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        render_camera=render_camera,
        use_camera_obs=use_camera_obs,
        use_object_obs=True,
        horizon=horizon,
        reward_shaping=True,
        seed=seed,
        cable_model=cable_model,
        grasp_mode=grasp_mode,
        difficulty=difficulty,
        **kwargs,
    )


def obs_to_vector(obs, obs_keys=DEFAULT_OBS_KEYS):
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in obs_keys], axis=0)


def action_dim(env):
    low, _ = env.action_spec
    return int(low.shape[0])


def clip_action(env, action):
    low, high = env.action_spec
    return np.clip(np.asarray(action, dtype=np.float32), low, high)


def _threading_targets(env):
    cable_end = env._get_cable_end_pos()
    pole1 = env._get_pole1_pos()
    pole2 = env._get_pole2_pos()
    gap_mid = 0.5 * (pole1 + pole2)
    endpoint_goal = env._get_endpoint_goal()
    table_z = env.anchor_pos[2] if env.anchor_pos is not None else cable_end[2]
    pole_top_z = pole1[2] + env.pole_height / 2.0
    panda_flex_mode = getattr(env, "_robot_name", "") == "Panda" and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp", "composite_cable", "composite_soft", "composite_softened"}
    safe_z = pole_top_z + (0.105 if panda_flex_mode else 0.08)
    insert_z = pole_top_z + (0.022 if panda_flex_mode else 0.012)
    low_thread_z = min(table_z + 0.018, pole_top_z - env.low_thread_height_margin - 0.004)
    post_thread_z = low_thread_z
    laydown_z = table_z + 0.004
    pull_direction = endpoint_goal[:2] - gap_mid[:2]
    if env.anchor_pos is not None and abs(endpoint_goal[1] - env.anchor_pos[1]) > 1e-6:
        t = (gap_mid[1] - env.anchor_pos[1]) / (endpoint_goal[1] - env.anchor_pos[1])
        if 0.0 <= t <= 1.0:
            thread_xy = env.anchor_pos[:2] + t * (endpoint_goal[:2] - env.anchor_pos[:2])
            pull_direction = endpoint_goal[:2] - thread_xy
        else:
            thread_xy = gap_mid[:2]
    else:
        thread_xy = gap_mid[:2]
    pull_direction_norm = float(np.linalg.norm(pull_direction))
    if pull_direction_norm < 1e-6:
        pull_direction = np.array([0.0, -1.0], dtype=float)
    else:
        pull_direction = pull_direction / pull_direction_norm

    gap_entry = gap_mid.copy()
    gap_entry[:2] = thread_xy - pull_direction * 0.055
    gap_entry[2] = safe_z
    backoff_target = gap_mid.copy()
    backoff_target[:2] = thread_xy - pull_direction * 0.18
    backoff_target[2] = safe_z
    gap_insert = gap_mid.copy()
    gap_insert[:2] = thread_xy + pull_direction * 0.025
    gap_insert[2] = insert_z
    lower_thread_target = gap_insert.copy()
    lower_thread_target[2] = low_thread_z
    # composite 系列刚性较大，线缆沿 anchor→gripper 直线走。
    # 必须计算 gripper XY 使直线穿过杆间隙中心。
    if panda_flex_mode and getattr(env, "cable_model", "") in {"composite_cable", "composite_soft", "composite_softened"}:
        anchor_xy = env.anchor_pos[:2]
        pole_y = pole1[1]
        gap_cx = gap_mid[0]
        ya = anchor_xy[1] - pole_y
        if abs(ya) > 1e-6:
            # 令 anchor→gripper 直线在 y=pole_y 处 x=gap_cx
            # gripper_x = anchor_x + (gap_cx - anchor_x) * (gripper_y - pole_y) / (anchor_y - pole_y)
            # 简化：gripper 放在使 cable 通过 gap 中心的位置
            t_cross = -ya / (0.0 - ya)  # t where cable crosses y=pole_y (gripper_y ≈ pole_y)
            if 0 < t_cross < 1:
                needed_gripper_x = gap_cx  # gripper x 使 crossing 在 gap center
                gap_insert[:2] = np.array([needed_gripper_x, pole_y])
                lower_thread_target[:2] = gap_insert[:2].copy()
    pull_target = endpoint_goal.copy()
    pull_target[2] = post_thread_z
    if panda_flex_mode:
        cable_model = getattr(env, "cable_model", "")
        is_composite = cable_model in {"composite_cable", "composite_soft", "composite_softened"}
        if cable_model == "composite_cable":
            pull_overshoot = 0.0
        elif is_composite:
            pull_overshoot = 0.050  # composite_soft 弹性回弹较大，需要更多过冲
        else:
            pull_overshoot = 0.040
        pull_target[:2] = endpoint_goal[:2] + pull_direction * pull_overshoot
        if is_composite:
            # composite 系列刚性较大，夹爪高度直接影响线缆路径。
            # 必须低于杆顶，使线缆穿过间隙而非越过杆顶。
            pull_target[2] = pole_top_z - 0.01
        else:
            pull_target[2] = max(pole_top_z + 0.005, insert_z + 0.015)
    table_straighten_target = endpoint_goal.copy()
    straighten_overshoot = float(getattr(env, "straightening_goal_distance", 0.26)) if panda_flex_mode else 0.04
    if getattr(env, "cable_model", "") == "composite_cable":
        straighten_overshoot = max(straighten_overshoot, 0.18)
    table_straighten_target[:2] = endpoint_goal[:2] + pull_direction * straighten_overshoot
    table_straighten_target[2] = table_z + (0.002 if panda_flex_mode else 0.001)
    press_table_target = table_straighten_target.copy()
    if panda_flex_mode and getattr(env, "grasp_mode", "attachment") == "physical":
        press_table_target[:2] = table_straighten_target[:2]
    elif panda_flex_mode:
        press_table_target[:2] = endpoint_goal[:2] - pull_direction * 0.015
    press_table_target[2] = table_z + 0.0005
    laydown_target = endpoint_goal.copy()
    if getattr(env, "_robot_name", "") == "Panda":
        laydown_target = pull_target.copy()
        laydown_target[2] = table_z + 0.012
    else:
        laydown_pull_extra = 0.012
        laydown_target[:2] = endpoint_goal[:2] + pull_direction * laydown_pull_extra
        laydown_target[2] = laydown_z
    release_target = press_table_target.copy()
    if getattr(env, "_robot_name", "") == "Panda" and not (
        panda_flex_mode and getattr(env, "grasp_mode", "attachment") == "physical"
    ):
        release_target[:2] = endpoint_goal[:2]
        release_target[2] = table_z + 0.004
    else:
        release_target[2] = table_z + 0.0005
    if panda_flex_mode:
        # Release near endpoint goal at safe height (near pull_target z).
        # Don't drive to table surface -- just let go after threading.
        release_target[:2] = endpoint_goal[:2]
        release_target[2] = max(pull_target[2] - 0.005, table_z + 0.008)
    retreat_target = release_target.copy()
    settle_target = release_target.copy()
    if panda_flex_mode:
        retreat_target[:2] = release_target[:2]
        retreat_target[2] = table_z + 0.012
        settle_target[:2] = retreat_target[:2]
        settle_target[2] = table_z + 0.012
    else:
        retreat_target[:2] = release_target[:2]
        retreat_target[2] = table_z + 0.012
        settle_target = retreat_target.copy()
    return {
        "cable_end": cable_end,
        "pole1": pole1,
        "pole2": pole2,
        "gap_mid": gap_mid,
        "endpoint_goal": endpoint_goal,
        "safe_z": safe_z,
        "insert_z": insert_z,
        "low_thread_z": low_thread_z,
        "gap_entry": gap_entry,
        "backoff_target": backoff_target,
        "gap_insert": gap_insert,
        "lower_thread_target": lower_thread_target,
        "pull_target": pull_target,
        "table_straighten_target": table_straighten_target,
        "press_table_target": press_table_target,
        "laydown_target": laydown_target,
        "release_target": release_target,
        "retreat_target": retreat_target,
        "settle_target": settle_target,
    }


def _min_outer_clearance(env):
    cable_xy = env._get_cable_points()[:, :2]
    pole1_xy = env._get_pole1_pos()[:2]
    pole2_xy = env._get_pole2_pos()[:2]
    corridor_min = min(pole1_xy[0], pole2_xy[0]) + env.pole_radius - env.gap_margin
    corridor_max = max(pole1_xy[0], pole2_xy[0]) - env.pole_radius + env.gap_margin
    pole_y = pole1_xy[1]
    band = np.abs(cable_xy[:, 1] - pole_y) <= env.thread_corridor_depth * 1.5
    if not np.any(band):
        return float("inf")
    x_vals = cable_xy[band, 0]
    return float(np.min(np.minimum(np.abs(x_vals - corridor_min), np.abs(x_vals - corridor_max))))


ATTACHMENT_EXPERT_PHASES = [
    {"name": "approach_above_end", "max_steps": 30, "pos_tolerance": 0.03, "hold_steps": 0},
    {"name": "descend_to_grasp", "max_steps": 24, "pos_tolerance": 0.012, "hold_steps": 0},
    {"name": "attach", "max_steps": 10, "pos_tolerance": 0.01, "hold_steps": 6, "attach_on_step": 1},
    {"name": "lift_clear", "max_steps": 30, "pos_tolerance": 0.03, "hold_steps": 0},
    {"name": "backoff_clearance", "max_steps": 40, "pos_tolerance": 0.03, "hold_steps": 8},
    {"name": "align_to_gap_entry", "max_steps": 40, "pos_tolerance": 0.035, "hold_steps": 0},
    {"name": "enter_gap", "max_steps": 30, "pos_tolerance": 0.025, "hold_steps": 0},
    {"name": "lower_after_gap", "max_steps": 50, "pos_tolerance": 0.018, "hold_steps": 12},
    {"name": "pull_through", "max_steps": 90, "pos_tolerance": 0.025, "hold_steps": 18},
    {"name": "lay_down_endpoint", "max_steps": 60, "pos_tolerance": 0.02, "hold_steps": 24},
    {"name": "table_straighten", "max_steps": 60, "pos_tolerance": 0.02, "hold_steps": 18},
    {"name": "press_to_table", "max_steps": 40, "pos_tolerance": 0.018, "hold_steps": 16},
    {"name": "release", "max_steps": 12, "pos_tolerance": 0.018, "hold_steps": 6, "detach_on_step": 0},
    {"name": "retreat", "max_steps": 30, "pos_tolerance": 0.03, "hold_steps": 6},
    {"name": "settle_wait", "max_steps": 60, "pos_tolerance": 0.04, "hold_steps": 36},
]

PHYSICAL_EXPERT_PHASES = [
    {"name": "approach_above_end", "max_steps": 30, "pos_tolerance": 0.03, "hold_steps": 0},
    {"name": "descend_to_grasp", "max_steps": 32, "pos_tolerance": 0.01, "hold_steps": 8},
    {"name": "close_grasp", "max_steps": 24, "pos_tolerance": 0.01, "hold_steps": 10},
    {"name": "stabilize_grasp", "max_steps": 18, "pos_tolerance": 0.012, "hold_steps": 8},
    {"name": "lift_clear", "max_steps": 45, "pos_tolerance": 0.03, "hold_steps": 12},
    {"name": "backoff_clearance", "max_steps": 40, "pos_tolerance": 0.03, "hold_steps": 8},
    {"name": "align_to_gap_entry", "max_steps": 40, "pos_tolerance": 0.035, "hold_steps": 0},
    {"name": "enter_gap", "max_steps": 36, "pos_tolerance": 0.025, "hold_steps": 0},
    {"name": "lower_after_gap", "max_steps": 60, "pos_tolerance": 0.018, "hold_steps": 10},
    {"name": "pull_through", "max_steps": 110, "pos_tolerance": 0.025, "hold_steps": 14},
    {"name": "lay_down_endpoint", "max_steps": 70, "pos_tolerance": 0.02, "hold_steps": 16},
    {"name": "table_straighten", "max_steps": 70, "pos_tolerance": 0.02, "hold_steps": 14},
    {"name": "press_to_table", "max_steps": 45, "pos_tolerance": 0.018, "hold_steps": 12},
    {"name": "release", "max_steps": 16, "pos_tolerance": 0.018, "hold_steps": 6},
    {"name": "retreat", "max_steps": 30, "pos_tolerance": 0.03, "hold_steps": 6},
    {"name": "settle_wait", "max_steps": 40, "pos_tolerance": 0.04, "hold_steps": 24},
]

PANDA_POST_THREAD_REGRASP_PHASES = [
    {"name": "regrasp_table_approach", "max_steps": 36, "pos_tolerance": 0.015, "hold_steps": 6},
    {"name": "regrasp_table_close", "max_steps": 24, "pos_tolerance": 0.010, "hold_steps": 10},
    {"name": "regrasp_table_stabilize", "max_steps": 18, "pos_tolerance": 0.010, "hold_steps": 8},
]

EXPERT_PHASES = ATTACHMENT_EXPERT_PHASES


def _expert_phases_for_env(env):
    if getattr(env, "grasp_mode", "attachment") == "physical":
        panda_flex_mode = getattr(env, "_robot_name", "") == "Panda" and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp", "composite_cable", "composite_soft", "composite_softened"}
        if panda_flex_mode:
            # Simplified ending for Panda flex: after pull_through, release immediately.
            # Skip lay_down_endpoint, regrasp_table_*, table_straighten, press_to_table
            # which cause cable penetration through poles and table.
            skip_phases = {"lay_down_endpoint", "table_straighten", "press_to_table", "backoff_clearance"}
            phases = []
            for cfg in PHYSICAL_EXPERT_PHASES:
                if cfg["name"] in skip_phases:
                    continue
                phases.append(cfg)
            return phases
        return PHYSICAL_EXPERT_PHASES
    # For attachment mode with Panda flex, also simplify the ending.
    panda_flex_mode = getattr(env, "_robot_name", "") == "Panda" and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp", "composite_cable", "composite_soft", "composite_softened"}
    if panda_flex_mode:
        skip_phases = {"lay_down_endpoint", "table_straighten", "press_to_table", "backoff_clearance"}
        phases = []
        for cfg in ATTACHMENT_EXPERT_PHASES:
            if cfg["name"] in skip_phases:
                continue
            phases.append(cfg)
        return phases
    return ATTACHMENT_EXPERT_PHASES


def _phase_cfg_for_env(env, phase_cfg):
    cfg = dict(phase_cfg)
    panda_mode = getattr(env, "_robot_name", "") == "Panda"
    panda_flex_mode = panda_mode and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp", "composite_cable"}
    if panda_mode:
        if cfg["name"] == "pull_through":
            cfg["max_steps"] = max(cfg["max_steps"], 110)
            if getattr(env, "grasp_mode", "attachment") == "physical" and panda_flex_mode:
                cfg["hold_steps"] = 12
                # Slower pull to prevent cable swinging through poles.
                cfg["target_step_limit"] = 0.004
            else:
                cfg["hold_steps"] = max(cfg["hold_steps"], 24)
        elif cfg["name"] == "lay_down_endpoint":
            cfg["max_steps"] = max(cfg["max_steps"], 110)
            cfg["hold_steps"] = 18 if panda_flex_mode else max(cfg["hold_steps"], 36)
            if getattr(env, "grasp_mode", "attachment") == "physical" and panda_flex_mode:
                cfg["max_steps"] = 60
                cfg["hold_steps"] = 10
                cfg["target_step_limit"] = 0.006
        elif cfg["name"] == "table_straighten":
            if panda_flex_mode:
                cfg["max_steps"] = 60 if getattr(env, "grasp_mode", "attachment") == "physical" else max(cfg["max_steps"], 110)
                cfg["hold_steps"] = 18
                if getattr(env, "grasp_mode", "attachment") == "physical":
                    cfg["target_step_limit"] = 0.009
        elif cfg["name"] == "regrasp_table_approach":
            if panda_flex_mode:
                cfg["max_steps"] = 30
                cfg["hold_steps"] = 6
                cfg["target_step_limit"] = 0.010
        elif cfg["name"] == "regrasp_table_close":
            if panda_flex_mode:
                cfg["max_steps"] = 44
                cfg["hold_steps"] = 16
                cfg["target_step_limit"] = 0.008
        elif cfg["name"] == "regrasp_table_stabilize":
            if panda_flex_mode:
                cfg["max_steps"] = 30
                cfg["hold_steps"] = 12
                cfg["target_step_limit"] = 0.006
        elif cfg["name"] == "regrasp_table_lift":
            if panda_flex_mode:
                cfg["max_steps"] = 24
                cfg["hold_steps"] = 8
                cfg["target_step_limit"] = 0.008
        elif cfg["name"] == "press_to_table":
            if panda_flex_mode:
                cfg["max_steps"] = 90 if getattr(env, "grasp_mode", "attachment") == "physical" else 24
                cfg["hold_steps"] = 8
                if getattr(env, "grasp_mode", "attachment") == "physical":
                    cfg["target_step_limit"] = 0.008
        elif cfg["name"] == "release":
            if panda_flex_mode:
                cfg["max_steps"] = min(cfg["max_steps"], 10)
                cfg["hold_steps"] = 4
            else:
                cfg["hold_steps"] = max(cfg["hold_steps"], 12)
        elif cfg["name"] == "retreat":
            if panda_flex_mode:
                cfg["max_steps"] = max(cfg["max_steps"], 40)
                cfg["hold_steps"] = 10
            else:
                cfg["hold_steps"] = max(cfg["hold_steps"], 10)
        elif cfg["name"] == "settle_wait":
            if panda_flex_mode:
                # Shorter settle since cable is released immediately after threading.
                cfg["max_steps"] = 40
                cfg["hold_steps"] = 20
            else:
                cfg["max_steps"] = max(cfg["max_steps"], 60)
                cfg["hold_steps"] = max(cfg["hold_steps"], 36)
    if getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp"} and getattr(env, "_robot_name", "") != "Panda":
        if cfg["name"] == "table_straighten":
            cfg["max_steps"] = max(cfg["max_steps"], 90)
            cfg["hold_steps"] = max(cfg["hold_steps"], 24)
        elif cfg["name"] == "press_to_table":
            cfg["max_steps"] = max(cfg["max_steps"], 55)
            cfg["hold_steps"] = max(cfg["hold_steps"], 24)
        elif cfg["name"] == "release":
            cfg["max_steps"] = max(cfg["max_steps"], 18)
            cfg["hold_steps"] = max(cfg["hold_steps"], 10)
        elif cfg["name"] == "settle_wait":
            cfg["max_steps"] = max(cfg["max_steps"], 80)
            cfg["hold_steps"] = max(cfg["hold_steps"], 48)
    if getattr(env, "grasp_mode", "attachment") == "physical":
        if cfg["name"] == "lift_clear":
            cfg["max_steps"] = max(cfg["max_steps"], 55)
            if getattr(env, "_robot_name", "") == "UR5e":
                cfg["target_step_limit"] = 0.02
            elif getattr(env, "_robot_name", "") == "Panda":
                cfg["target_step_limit"] = 0.015
        elif cfg["name"] == "stabilize_grasp":
            cfg["hold_steps"] = max(cfg["hold_steps"], 10)
            if getattr(env, "_robot_name", "") == "UR5e":
                cfg["target_step_limit"] = 0.008
            elif getattr(env, "_robot_name", "") == "Panda":
                cfg["max_steps"] = max(cfg["max_steps"], 22)
                cfg["hold_steps"] = max(cfg["hold_steps"], 12)
                cfg["target_step_limit"] = 0.006
        elif cfg["name"] == "close_grasp":
            cfg["hold_steps"] = max(cfg["hold_steps"], 12)
            if getattr(env, "_robot_name", "") == "UR5e":
                cfg["target_step_limit"] = 0.015
            elif getattr(env, "_robot_name", "") == "Panda":
                cfg["max_steps"] = max(cfg["max_steps"], 28)
                cfg["hold_steps"] = max(cfg["hold_steps"], 14)
                cfg["target_step_limit"] = 0.010
        elif cfg["name"] == "descend_to_grasp":
            if getattr(env, "_robot_name", "") == "UR5e":
                cfg["target_step_limit"] = 0.02
            elif getattr(env, "_robot_name", "") == "Panda":
                cfg["max_steps"] = max(cfg["max_steps"], 36)
                cfg["hold_steps"] = max(cfg["hold_steps"], 10)
                cfg["target_step_limit"] = 0.015
    return cfg


def _physical_centering_correction(env, point, *, max_shift=0.008):
    if not hasattr(env, "_gripper_fingerpad_geom_groups") or not hasattr(env, "_fingerpad_group_center"):
        return np.zeros(3, dtype=float)
    left_geoms, right_geoms = env._gripper_fingerpad_geom_groups()
    left_center = env._fingerpad_group_center(left_geoms)
    right_center = env._fingerpad_group_center(right_geoms)
    if left_center is None or right_center is None:
        return np.zeros(3, dtype=float)

    point = np.asarray(point, dtype=float)
    finger_axis = np.asarray(right_center, dtype=float) - np.asarray(left_center, dtype=float)
    axis_len_sq = float(np.dot(finger_axis, finger_axis))
    if axis_len_sq < 1e-10:
        return np.zeros(3, dtype=float)

    projection = float(np.dot(point - left_center, finger_axis) / axis_len_sq)
    correction = (projection - 0.5) * finger_axis
    correction[2] = 0.0
    corr_norm = float(np.linalg.norm(correction))
    if corr_norm > max_shift and corr_norm > 1e-10:
        correction *= max_shift / corr_norm
    return correction


def _physical_single_side_contact_bias(env, *, max_shift=0.003):
    if not hasattr(env, "_physical_grasp_contact_sides"):
        return np.zeros(3, dtype=float)
    if not hasattr(env, "_gripper_fingerpad_geom_groups") or not hasattr(env, "_fingerpad_group_center"):
        return np.zeros(3, dtype=float)

    left_count, right_count = env._physical_grasp_contact_sides()
    if (left_count > 0 and right_count > 0) or (left_count <= 0 and right_count <= 0):
        return np.zeros(3, dtype=float)

    left_geoms, right_geoms = env._gripper_fingerpad_geom_groups()
    left_center = env._fingerpad_group_center(left_geoms)
    right_center = env._fingerpad_group_center(right_geoms)
    if left_center is None or right_center is None:
        return np.zeros(3, dtype=float)

    finger_axis = np.asarray(right_center, dtype=float) - np.asarray(left_center, dtype=float)
    axis_norm = float(np.linalg.norm(finger_axis))
    if axis_norm < 1e-10:
        return np.zeros(3, dtype=float)
    finger_axis = finger_axis / axis_norm

    # If only one side contacts the cable, move the clamp midpoint slightly
    # toward that contacting side so the other pad can close onto the cable.
    if left_count > 0 and right_count <= 0:
        return -finger_axis * float(max_shift)
    if right_count > 0 and left_count <= 0:
        return finger_axis * float(max_shift)
    return np.zeros(3, dtype=float)


def expert_phase_target(env, phase_name):
    targets = _threading_targets(env)
    cable_end = targets["cable_end"]
    endpoint_goal = targets["endpoint_goal"]
    table_z = float(getattr(env, "table_top_z", env.anchor_pos[2] if getattr(env, "anchor_pos", None) is not None else cable_end[2]))
    panda_mode = getattr(env, "_robot_name", "") == "Panda"
    ur5e_mode = getattr(env, "_robot_name", "") == "UR5e"
    physical_mode = getattr(env, "grasp_mode", "attachment") == "physical"
    clamp_offset = (
        env._gripper_clamp_center_offset()
        if physical_mode and hasattr(env, "_gripper_clamp_center_offset")
        else np.zeros(3, dtype=float)
    )
    target_is_site_frame = False
    grasp_point = None
    if physical_mode and hasattr(env, "_get_physical_grasp_point_pos"):
        try:
            grasp_point = np.asarray(env._get_physical_grasp_point_pos(), dtype=float).copy()
        except Exception:
            grasp_point = None
    panda_needs_regrasp = bool(
        panda_mode
        and physical_mode
        and phase_name in {"pull_through", "lay_down_endpoint", "table_straighten", "press_to_table"}
        and not bool(getattr(env, "physical_grasp_ready", lambda: False)())
    )

    if phase_name == "approach_above_end":
        target = cable_end.copy()
        target[2] = targets["safe_z"]
        grip = -1.0
    elif phase_name == "descend_to_grasp":
        if physical_mode and panda_mode:
            corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.010)
            target = corrected_point - clamp_offset
            target[2] = target[2] + 0.004
            target_is_site_frame = True
        else:
            target = cable_end.copy()
        if physical_mode and ur5e_mode:
            target[2] = cable_end[2] - 0.004
        else:
            if not (physical_mode and panda_mode):
                target[2] = cable_end[2] + 0.012
        grip = -1.0
    elif phase_name == "attach":
        target = cable_end.copy()
        target[2] = cable_end[2] + 0.01
        grip = 1.0
    elif phase_name == "close_grasp":
        if physical_mode and panda_mode:
            corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.012)
            corrected_point += _physical_single_side_contact_bias(env, max_shift=0.003)
            target = corrected_point - clamp_offset
            target[2] = target[2] + 0.001
            target_is_site_frame = True
        else:
            target = cable_end.copy()
        if physical_mode and ur5e_mode:
            target[2] = cable_end[2] - 0.008
        else:
            if not (physical_mode and panda_mode):
                target[2] = cable_end[2] + 0.006
        grip = 1.0
    elif phase_name == "stabilize_grasp":
        if physical_mode and panda_mode:
            corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.012)
            corrected_point += _physical_single_side_contact_bias(env, max_shift=0.004)
            target = corrected_point - clamp_offset
            target[2] = target[2] - 0.001
            target_is_site_frame = True
        else:
            target = cable_end.copy()
        if physical_mode and ur5e_mode:
            target[2] = cable_end[2] - 0.006
        else:
            if not (physical_mode and panda_mode):
                target[2] = cable_end[2] + 0.004
        grip = 1.0
    elif phase_name == "lift_clear":
        if physical_mode:
            current_eef = env._get_gripper_site_position().copy()
            target = np.array([current_eef[0], current_eef[1], targets["safe_z"]], dtype=float)
        else:
            target = cable_end.copy()
            target[2] = targets["safe_z"]
        grip = 1.0
    elif phase_name == "backoff_clearance":
        target = targets["backoff_target"]
        grip = 1.0
    elif phase_name == "align_to_gap_entry":
        target = targets["gap_entry"]
        grip = 1.0
    elif phase_name == "enter_gap":
        target = targets["gap_insert"]
        grip = 1.0
    elif phase_name == "lower_after_gap":
        target = targets["lower_thread_target"]
        grip = 1.0
    elif phase_name == "pull_through":
        if panda_needs_regrasp:
            corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.012)
            corrected_point += _physical_single_side_contact_bias(env, max_shift=0.004)
            target = corrected_point - clamp_offset
            target[2] = max(table_z + 0.01, cable_end[2] + 0.004)
            target_is_site_frame = True
        else:
            target = targets["pull_target"]
        grip = 1.0
    elif phase_name == "regrasp_table_approach":
        _, _, finger_axis, segment_mid, _ = _panda_regrasp_segment_pose(env, cable_end)
        desired_midpoint = np.asarray(segment_mid, dtype=float).copy() - finger_axis * 0.003
        desired_midpoint[2] = cable_end[2] + 0.006
        corrected_point = desired_midpoint + _physical_centering_correction(env, desired_midpoint, max_shift=0.012)
        fallback_target = corrected_point - clamp_offset
        target = _panda_midpoint_servo_target(env, corrected_point, fallback_target)
        target_is_site_frame = True
        grip = -1.0
    elif phase_name == "regrasp_table_close":
        _, _, finger_axis, segment_mid, _ = _panda_regrasp_segment_pose(env, cable_end)
        desired_midpoint = np.asarray(segment_mid, dtype=float).copy()
        desired_midpoint[2] = cable_end[2] - 0.006
        corrected_point = desired_midpoint + _physical_centering_correction(env, desired_midpoint, max_shift=0.012)
        corrected_point += _physical_single_side_contact_bias(env, max_shift=0.004)
        fallback_target = corrected_point - clamp_offset
        target = _panda_midpoint_servo_target(env, corrected_point, fallback_target)
        target_is_site_frame = True
        grip = 1.0
    elif phase_name == "regrasp_table_stabilize":
        _, _, finger_axis, _, stabilize_point = _panda_regrasp_segment_pose(env, cable_end)
        desired_midpoint = np.asarray(stabilize_point, dtype=float).copy()
        desired_midpoint[2] = cable_end[2] - 0.004
        corrected_point = desired_midpoint + _physical_centering_correction(env, desired_midpoint, max_shift=0.012)
        corrected_point += _physical_single_side_contact_bias(env, max_shift=0.004)
        fallback_target = corrected_point - clamp_offset
        target = _panda_midpoint_servo_target(env, corrected_point, fallback_target)
        target_is_site_frame = True
        grip = 1.0
    elif phase_name == "regrasp_table_lift":
        current_eef = env._get_gripper_site_position().copy()
        target = np.array([current_eef[0], current_eef[1], table_z + 0.030], dtype=float)
        grip = 1.0
    elif phase_name == "lay_down_endpoint":
        if panda_needs_regrasp:
            corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.012)
            corrected_point += _physical_single_side_contact_bias(env, max_shift=0.004)
            target = corrected_point - clamp_offset
            target[2] = max(table_z + 0.002, cable_end[2] + 0.002)
            target_is_site_frame = True
        else:
            target = targets["laydown_target"]
        grip = 1.0
    elif phase_name == "table_straighten":
        if panda_needs_regrasp:
            metrics = getattr(env, "_compute_metrics", lambda: {})()
            post_thread_ready = _metric_bool(metrics.get("settled_on_table_final", False)) and (
                _metric_bool(metrics.get("endpoint_past_gap_final", False))
                or _metric_bool(metrics.get("threaded_final", False))
                or _metric_bool(metrics.get("cable_low_intersects_pole_segment", False))
                or _metric_bool(metrics.get("final_line_crosses_gap", False))
            )
            if post_thread_ready:
                corrected_point = np.asarray(targets["table_straighten_target"], dtype=float).copy()
                corrected_point[2] = max(table_z + 0.0015, cable_end[2] + 0.001)
            else:
                corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.012)
            corrected_point += _physical_single_side_contact_bias(env, max_shift=0.004)
            target = corrected_point - clamp_offset
            target[2] = max(table_z + 0.002, cable_end[2] + 0.001)
            target_is_site_frame = True
        else:
            target = targets["table_straighten_target"]
        grip = 1.0
    elif phase_name == "press_to_table":
        if panda_needs_regrasp:
            metrics = getattr(env, "_compute_metrics", lambda: {})()
            post_thread_ready = _metric_bool(metrics.get("settled_on_table_final", False)) and (
                _metric_bool(metrics.get("endpoint_past_gap_final", False))
                or _metric_bool(metrics.get("threaded_final", False))
                or _metric_bool(metrics.get("cable_low_intersects_pole_segment", False))
                or _metric_bool(metrics.get("final_line_crosses_gap", False))
            )
            if post_thread_ready:
                corrected_point = np.asarray(cable_end, dtype=float).copy()
                goal_delta = np.asarray(targets["press_table_target"][:2], dtype=float) - corrected_point[:2]
                goal_dist = float(np.linalg.norm(goal_delta))
                if goal_dist > 1e-8:
                    corrected_point[:2] += goal_delta / goal_dist * min(goal_dist + 0.02, 0.08)
                corrected_point[2] = max(table_z + 0.001, cable_end[2] + 0.0005)
            else:
                corrected_point = np.asarray(cable_end, dtype=float).copy()
            corrected_point += _physical_centering_correction(env, corrected_point, max_shift=0.010)
            corrected_point += _physical_single_side_contact_bias(env, max_shift=0.003)
            target = corrected_point - clamp_offset
            target[2] = max(table_z + 0.001, cable_end[2] + 0.001)
            target_is_site_frame = True
        else:
            target = targets["press_table_target"]
        grip = 1.0
    elif phase_name == "release":
        target = targets["release_target"]
        grip = -1.0
    elif phase_name == "retreat":
        target = targets["retreat_target"]
        grip = -1.0
    elif phase_name == "settle_wait":
        target = targets["settle_target"]
        grip = -1.0
    else:
        raise ValueError(f"Unknown expert phase: {phase_name}")

    if panda_mode and phase_name in {"pull_through", "lay_down_endpoint", "table_straighten", "press_to_table", "release"}:
        endpoint_reference = cable_end
        if physical_mode and grasp_point is not None and getattr(env, "_physical_grasp_point_idx", -1) >= 0:
            endpoint_reference = grasp_point
        endpoint_error = endpoint_goal - endpoint_reference
        target = np.asarray(target, dtype=float).copy()
        phase_xy_limit = {
            "pull_through": 0.09,
            "lay_down_endpoint": 0.08,
            "table_straighten": 0.10,
            "press_to_table": 0.08,
            "release": 0.06,
        }[phase_name]
        phase_z_clip = {
            "pull_through": (-0.01, 0.03),
            "lay_down_endpoint": (-0.03, 0.01),
            "table_straighten": (-0.02, 0.008),
            "press_to_table": (-0.015, 0.006),
            "release": (-0.015, 0.005),
        }[phase_name]
        target[:2] += np.clip(endpoint_error[:2], -phase_xy_limit, phase_xy_limit)
        target[2] += np.clip(endpoint_error[2], phase_z_clip[0], phase_z_clip[1])
        if physical_mode and grasp_point is not None and phase_name in {"pull_through", "lay_down_endpoint", "table_straighten", "press_to_table"}:
            target += _physical_centering_correction(env, grasp_point, max_shift=0.008)

    if physical_mode and phase_name in {"descend_to_grasp", "close_grasp", "stabilize_grasp"} and not target_is_site_frame:
        target = np.asarray(target, dtype=float).copy()
        target += _physical_centering_correction(env, cable_end)

    target = np.asarray(target, dtype=float)
    if physical_mode and phase_name not in {"release", "retreat", "settle_wait"} and not target_is_site_frame:
        target = target - clamp_offset

    return target, float(grip)


def expert_phase_action(env, phase_name):
    target, grip = expert_phase_target(env, phase_name)
    action = RobosuiteControllerAdapter(env).action_to_eef_position(target, gripper=grip)
    return action, target


def _smooth_action_target(raw_target, previous_target, max_step):
    raw_target = np.asarray(raw_target, dtype=float)
    if previous_target is None:
        return raw_target.copy()
    previous_target = np.asarray(previous_target, dtype=float)
    delta = raw_target - previous_target
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_step) or distance < 1e-9:
        return raw_target.copy()
    return previous_target + delta / distance * float(max_step)


def _metric_bool(value):
    return bool(np.asarray(value).item())


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _panda_post_thread_regrasp_candidate_ready(env):
    if getattr(env, "_robot_name", "") != "Panda":
        return False
    if getattr(env, "cable_model", "") not in {"flex", "flex_cable", "flexcomp"}:
        return False
    if not bool(getattr(env, "_is_flex_cable", False)):
        return False
    left_count, right_count = env._physical_grasp_contact_sides()
    contact_count = int(left_count + right_count)
    point_idx = int(getattr(env, "_physical_grasp_point_idx", -1))
    grasp_point = None
    if point_idx >= 0 and getattr(env, "_flex_vertnum", None) is not None:
        point_idx = int(np.clip(point_idx, 0, env._flex_vertnum - 1))
        grasp_point = np.asarray(env.sim.data.flexvert_xpos[point_idx], dtype=float).copy()
    endpoint_pos = np.asarray(env._get_cable_end_pos(), dtype=float).copy()
    point_between = bool(grasp_point is not None and env._is_point_between_gripper_fingerpads(grasp_point))
    endpoint_between = bool(env._is_point_between_gripper_fingerpads(endpoint_pos))
    metrics = getattr(env, "_compute_metrics", lambda: {})()
    return bool(
        contact_count > 0
        and env._fingerpad_gap_width() <= 0.05
        and (
            point_between
            or endpoint_between
            or (
                bool(metrics.get("endpoint_past_gap_final", False))
                and bool(metrics.get("settled_on_table_final", False))
            )
        )
    )


def _panda_regrasp_target_point(env, fallback_point):
    point = np.asarray(fallback_point, dtype=float).copy()
    if getattr(env, "_robot_name", "") != "Panda":
        return point
    if getattr(env, "cable_model", "") not in {"flex", "flex_cable", "flexcomp"}:
        return point
    if not bool(getattr(env, "_is_flex_cable", False)):
        return point
    if not hasattr(env, "_physical_tail_candidate_indices") or getattr(env, "_flex_vertnum", None) is None:
        return point

    candidate_indices = env._physical_tail_candidate_indices()
    if getattr(candidate_indices, "size", 0) == 0:
        return point
    candidate_positions = np.asarray(env.sim.data.flexvert_xpos[candidate_indices], dtype=float)
    eef_pos = np.asarray(env._get_gripper_site_position(), dtype=float)
    table_z = float(getattr(env, "table_top_z", point[2]))

    finger_axis_xy = None
    if hasattr(env, "_gripper_fingerpad_geom_groups") and hasattr(env, "_fingerpad_group_center"):
        left_geoms, right_geoms = env._gripper_fingerpad_geom_groups()
        left_center = env._fingerpad_group_center(left_geoms)
        right_center = env._fingerpad_group_center(right_geoms)
        if left_center is not None and right_center is not None:
            finger_axis_xy = np.asarray(right_center, dtype=float)[:2] - np.asarray(left_center, dtype=float)[:2]
            axis_norm = float(np.linalg.norm(finger_axis_xy))
            if axis_norm > 1e-8:
                finger_axis_xy = finger_axis_xy / axis_norm
            else:
                finger_axis_xy = None

    if candidate_indices.size >= 3:
        best_idx = None
        best_score = np.inf
        inboard_candidates = candidate_indices[:-1]
        if inboard_candidates.size >= 3:
            inboard_candidates = inboard_candidates[:3]
        for rank, idx in enumerate(inboard_candidates):
            idx = int(idx)
            pos = np.asarray(env.sim.data.flexvert_xpos[idx], dtype=float)
            prev_idx = max(idx - 1, 0)
            next_idx = min(idx + 1, env._flex_vertnum - 1)
            tangent = np.asarray(env.sim.data.flexvert_xpos[next_idx], dtype=float) - np.asarray(
                env.sim.data.flexvert_xpos[prev_idx], dtype=float
            )
            tangent_xy = tangent[:2]
            tangent_norm = float(np.linalg.norm(tangent_xy))
            tangent_align_penalty = 0.5
            if tangent_norm > 1e-8 and finger_axis_xy is not None:
                tangent_xy = tangent_xy / tangent_norm
                tangent_align_penalty = abs(float(np.dot(tangent_xy, finger_axis_xy)))
            xy_dist = float(np.linalg.norm(pos[:2] - eef_pos[:2]))
            table_penalty = abs(float(pos[2]) - table_z)
            endpoint_rank_penalty = float(rank) * 0.004
            score = 1.6 * tangent_align_penalty + 0.20 * xy_dist + 0.35 * table_penalty + endpoint_rank_penalty
            if score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            return np.asarray(env.sim.data.flexvert_xpos[best_idx], dtype=float).copy()

    best_idx = None
    best_score = np.inf
    for idx, pos in zip(candidate_indices, candidate_positions):
        xy_dist = float(np.linalg.norm(pos[:2] - eef_pos[:2]))
        table_penalty = abs(float(pos[2]) - table_z)
        score = xy_dist + 0.5 * table_penalty
        if score < best_score:
            best_score = score
            best_idx = int(idx)
    if best_idx is None:
        return point
    return np.asarray(env.sim.data.flexvert_xpos[best_idx], dtype=float).copy()


def _panda_tail_segment_direction(env, grasp_point, fallback_endpoint):
    grasp_point = np.asarray(grasp_point, dtype=float)
    endpoint = np.asarray(fallback_endpoint, dtype=float)
    direction = endpoint - grasp_point
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return direction / norm


def _panda_finger_axis(env):
    if not hasattr(env, "_gripper_fingerpad_geom_groups") or not hasattr(env, "_fingerpad_group_center"):
        return np.array([1.0, 0.0, 0.0], dtype=float)
    left_geoms, right_geoms = env._gripper_fingerpad_geom_groups()
    left_center = env._fingerpad_group_center(left_geoms)
    right_center = env._fingerpad_group_center(right_geoms)
    if left_center is None or right_center is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    axis = np.asarray(right_center, dtype=float) - np.asarray(left_center, dtype=float)
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return axis / norm


def _panda_regrasp_segment_pose(env, fallback_endpoint):
    grasp_point = _panda_regrasp_target_point(env, fallback_endpoint)
    tail_dir = _panda_tail_segment_direction(env, grasp_point, fallback_endpoint)
    # Regrasp the midpoint of a short tail segment instead of the endpoint itself.
    # This gives the finger pads actual segment length to cage on the table.
    segment_mid = np.asarray(grasp_point, dtype=float).copy() + tail_dir * 0.005
    stabilize_point = np.asarray(segment_mid, dtype=float).copy() - tail_dir * 0.003
    finger_axis = _panda_finger_axis(env)
    return grasp_point, tail_dir, finger_axis, segment_mid, stabilize_point


def _panda_midpoint_servo_target(env, desired_midpoint, fallback_target):
    desired_midpoint = np.asarray(desired_midpoint, dtype=float)
    current_midpoint = env._gripper_fingerpad_midpoint() if hasattr(env, "_gripper_fingerpad_midpoint") else None
    current_eef = np.asarray(env._get_gripper_site_position(), dtype=float)
    if current_midpoint is None:
        return np.asarray(fallback_target, dtype=float).copy()
    current_midpoint = np.asarray(current_midpoint, dtype=float)
    midpoint_error = desired_midpoint - current_midpoint
    target = current_eef + midpoint_error
    return np.asarray(target, dtype=float)


from examples.cable_threading.failure_reason import build_cable_threading_failure_reason


def summarize_episode(rows, env, total_reward, *, policy_name, episode_index, seed, phase_log=None):
    if rows:
        final = rows[-1]
    else:
        final = env._compute_metrics()

    reset_summary = dict(env.last_reset_summary)
    thread_completion_max = max(float(row["thread_completion"]) for row in rows) if rows else float(final["thread_completion"])
    post_collision_count_max = max(int(row.get("post_collision_count", 0)) for row in rows) if rows else int(final.get("post_collision_count", 0))
    ever_threaded = any(_metric_bool(row["threaded_final"]) for row in rows) if rows else _metric_bool(final["threaded_final"])
    ever_endpoint_region = (
        any(_metric_bool(row["endpoint_region_final"]) for row in rows) if rows else _metric_bool(final["endpoint_region_final"])
    )
    ever_endpoint_past_gap = (
        any(_metric_bool(row["endpoint_past_gap_final"]) for row in rows) if rows else _metric_bool(final["endpoint_past_gap_final"])
    )
    ever_straightened = (
        any(_metric_bool(row["straightened_final"]) for row in rows) if rows else _metric_bool(final["straightened_final"])
    )
    ever_final_line_crosses_gap = (
        any(_metric_bool(row["final_line_crosses_gap"]) for row in rows) if rows else _metric_bool(final["final_line_crosses_gap"])
    )
    ever_cable_intersects_pole_segment = (
        any(_metric_bool(row["cable_intersects_pole_segment"]) for row in rows)
        if rows
        else _metric_bool(final["cable_intersects_pole_segment"])
    )
    ever_cable_low_intersects_pole_segment = (
        any(_metric_bool(row["cable_low_intersects_pole_segment"]) for row in rows)
        if rows
        else _metric_bool(final["cable_low_intersects_pole_segment"])
    )
    ever_settled = (
        any(_metric_bool(row["settled_on_table_final"]) for row in rows) if rows else _metric_bool(final["settled_on_table_final"])
    )
    ever_anchor_stable = (
        any(_metric_bool(row["anchor_stable_final"]) for row in rows) if rows else _metric_bool(final["anchor_stable_final"])
    )
    ever_success = (
        any(_metric_bool(row["final_success"]) for row in rows) if rows else _metric_bool(final["final_success"])
    )

    summary = {
        "episode": int(episode_index),
        "policy": str(policy_name),
        "seed": int(seed) if seed is not None else -1,
        "difficulty": reset_summary.get("difficulty", getattr(env, "difficulty", "easy")),
        "return": float(total_reward),
        "steps": len(rows),
        "final_success": _metric_bool(final["final_success"]),
        "ever_success": ever_success,
        "threaded_final": _metric_bool(final["threaded_final"]),
        "ever_threaded": ever_threaded,
        "endpoint_region_final": _metric_bool(final["endpoint_region_final"]),
        "ever_endpoint_region": ever_endpoint_region,
        "endpoint_past_gap_final": _metric_bool(final["endpoint_past_gap_final"]),
        "ever_endpoint_past_gap": ever_endpoint_past_gap,
        "final_line_crosses_gap": _metric_bool(final["final_line_crosses_gap"]),
        "ever_final_line_crosses_gap": ever_final_line_crosses_gap,
        "cable_intersects_pole_segment": _metric_bool(final["cable_intersects_pole_segment"]),
        "ever_cable_intersects_pole_segment": ever_cable_intersects_pole_segment,
        "cable_low_intersects_pole_segment": _metric_bool(final["cable_low_intersects_pole_segment"]),
        "ever_cable_low_intersects_pole_segment": ever_cable_low_intersects_pole_segment,
        "straightened_final": _metric_bool(final["straightened_final"]),
        "ever_straightened": ever_straightened,
        "settled_on_table_final": _metric_bool(final["settled_on_table_final"]),
        "ever_settled_on_table": ever_settled,
        "anchor_stable_final": _metric_bool(final["anchor_stable_final"]),
        "ever_anchor_stable": ever_anchor_stable,
        "anchor_error_final": float(final["anchor_error_final"]),
        "min_pole_clearance_final": float(final["min_pole_clearance_final"]),
        "min_outer_clearance_final": float(final["min_outer_clearance_final"]),
        "endpoint_goal_error_final": float(final["endpoint_goal_error_final"]),
        "straightness_error_final": float(final["straightness_error_final"]),
        "tabletop_spread_final": float(final["tabletop_spread_final"]),
        "endpoint_height_error_final": float(final["endpoint_height_error_final"]),
        "thread_completion_final": float(final["thread_completion"]),
        "passed_keypoint_ratio": float(final.get("passed_keypoint_ratio", final["thread_completion"])),
        "gate_deviation": float(final.get("gate_deviation", np.nan)),
        "post_collision_count": int(final.get("post_collision_count", 0)),
        "post_collision_count_max": int(post_collision_count_max),
        "cable_on_table": _metric_bool(final.get("cable_on_table", final["settled_on_table_final"])),
        "thread_completion_max": float(thread_completion_max),
        "peak_height_excess_final": float(final["peak_height_excess"]),
        "gap_cross_x_final": float(final["gap_cross_x"]),
        "gap_cross_z_final": float(final["gap_cross_z"]),
        "thread_cross_value_final": float(final["thread_cross_value"]),
        "reset_root_x": float(reset_summary["root_pos"][0]) if "root_pos" in reset_summary else np.nan,
        "reset_root_y": float(reset_summary["root_pos"][1]) if "root_pos" in reset_summary else np.nan,
        "reset_root_z": float(reset_summary["root_pos"][2]) if "root_pos" in reset_summary else np.nan,
        "reset_shape_noise_l2": float(reset_summary.get("shape_noise_l2", np.nan)),
        "reset_endpoint_goal_init_error": float(reset_summary.get("endpoint_goal_init_error", np.nan)),
        "phase_count": len(phase_log) if phase_log is not None else 0,
    }

    summary.update(
        {
            "success": summary["final_success"],
            "success_any": summary["ever_success"],
            "threaded": summary["threaded_final"],
            "threaded_any": summary["ever_threaded"],
            "endpoint_region": summary["endpoint_region_final"],
            "endpoint_region_any": summary["ever_endpoint_region"],
            "endpoint_past_gap": summary["endpoint_past_gap_final"],
            "endpoint_past_gap_any": summary["ever_endpoint_past_gap"],
            "final_line_crosses_gap_any": summary["ever_final_line_crosses_gap"],
            "cable_intersects_pole_segment_any": summary["ever_cable_intersects_pole_segment"],
            "cable_low_intersects_pole_segment_any": summary["ever_cable_low_intersects_pole_segment"],
            "straightened": summary["straightened_final"],
            "straightened_any": summary["ever_straightened"],
            "settled_on_table": summary["settled_on_table_final"],
            "settled_on_table_any": summary["ever_settled_on_table"],
            "anchor_error": summary["anchor_error_final"],
            "min_pole_clearance": summary["min_pole_clearance_final"],
            "min_outer_clearance": summary["min_outer_clearance_final"],
            "endpoint_goal_error": summary["endpoint_goal_error_final"],
            "straightness_error": summary["straightness_error_final"],
            "tabletop_spread": summary["tabletop_spread_final"],
            "thread_completion": summary["thread_completion_final"],
            "passed_ratio": summary["passed_keypoint_ratio"],
            "max_thread_completion": summary["thread_completion_max"],
            "max_post_collision_count": summary["post_collision_count_max"],
        }
    )
    summary["failure_reason"] = (
        "" if summary["final_success"] else build_cable_threading_failure_reason(summary)
    )
    return summary


def rollout_expert_episode(
    env,
    render_sleep=0.0,
    obs_keys=DEFAULT_OBS_KEYS,
    record_trajectory=False,
    record_raw_obs=False,
    episode_index=0,
    seed=None,
    live_config=None,
):
    obs = env.reset()
    env.set_attachment_enabled(False)
    if live_config is not None:
        live_config.pop("_last_display_frame", None)
        obs = _warmup_live_after_reset(env, obs, live_config)
        if not live_config.get("has_valid_frame"):
            live_config["_consecutive_valid"] = 0
            live_config["frame_status"] = "warming_up"
    if live_config is not None and live_config.get("timeline_path"):
        record_live_timeline_event(live_config, "reset", "环境初始化")
    if getattr(env, "grasp_mode", "attachment") == "physical":
        endpoint_z = float(env._get_cable_end_pos()[2])
        if hasattr(env, "_physical_grasp_initial_endpoint_z"):
            env._physical_grasp_initial_endpoint_z = endpoint_z
        if hasattr(env, "_physical_grasp_hold_height"):
            env._physical_grasp_hold_height = endpoint_z
        if hasattr(env, "_physical_left_contact_memory"):
            env._physical_left_contact_memory = 0
        if hasattr(env, "_physical_right_contact_memory"):
            env._physical_right_contact_memory = 0
    needs_backoff = _min_outer_clearance(env) < float(
        getattr(env, "pre_thread_outer_clearance_threshold", 0.03)
    )

    total_reward = 0.0
    rows = []
    phase_log = []
    transitions = []
    done = False
    previous_action_target = env._get_gripper_site_position().copy()
    max_action_target_step = float(getattr(env, "expert_target_step_limit", 0.075))

    phase_plan = []
    for base_phase_cfg in _expert_phases_for_env(env):
        phase_cfg = _phase_cfg_for_env(env, base_phase_cfg)
        if phase_cfg["name"] == "backoff_clearance" and not needs_backoff:
            continue
        phase_plan.append(phase_cfg)

    phase_name_to_index = {cfg["name"]: idx for idx, cfg in enumerate(phase_plan)}
    regrasp_attempts = 0
    max_regrasp_attempts = 3 if getattr(env, "grasp_mode", "attachment") == "physical" else 0
    post_thread_regrasp_attempts = 0
    panda_flex_physical = (
        getattr(env, "grasp_mode", "attachment") == "physical"
        and getattr(env, "_robot_name", "") == "Panda"
        and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp"}
    )
    max_post_thread_regrasp_attempts = 1 if panda_flex_physical else 2
    transport_regrasp_phases = {"align_to_gap_entry", "enter_gap", "lower_after_gap", "pull_through"}
    step_recorder = _create_step_metric_recorder(env, episode_index, live_config)
    phase_idx = 0
    while phase_idx < len(phase_plan):
        phase_cfg = phase_plan[phase_idx]
        phase_name = phase_cfg["name"]
        if live_config is not None:
            maybe_record_live_timeline_expert_phase(live_config, phase_name)
        phase_target_step_limit = float(phase_cfg.get("target_step_limit", max_action_target_step))
        phase_completed = False
        lost_physical_grasp = False
        for local_step in range(phase_cfg["max_steps"]):
            if local_step == phase_cfg.get("attach_on_step", -1):
                env.set_attachment_enabled(True)
            if local_step == phase_cfg.get("detach_on_step", -1):
                env.set_attachment_enabled(False)
            raw_target, grip = expert_phase_target(env, phase_name)
            target = _smooth_action_target(raw_target, previous_action_target, phase_target_step_limit)
            previous_action_target = target.copy()
            action = RobosuiteControllerAdapter(env).action_to_eef_position(target, gripper=grip)
            prev_obs = obs
            step_t0 = time.perf_counter()
            obs, reward, done, info = env.step(action)
            step_wall_sec = time.perf_counter() - step_t0
            total_reward += reward
            rows.append(dict(info))
            phase_log.append(phase_name)
            _record_step_metric(
                step_recorder, len(rows) - 1, action, reward, done, info, step_wall_sec=step_wall_sec
            )

            if live_config is not None:
                live_config["step"] = int(live_config.get("step", 0)) + 1
                live_config["phase"] = phase_name
                maybe_capture_and_persist_live_frame(env, obs, live_config)
                write_live_status(live_config)

            if record_trajectory:
                eef_after = env._get_gripper_site_position()
                step_entry = {
                    "obs": obs_to_vector(prev_obs, obs_keys),
                    "next_obs": obs_to_vector(obs, obs_keys),
                    "action": action.copy(),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": dict(info),
                    "phase": phase_name,
                    "eef_pos": eef_after.copy(),
                    "eef_target": np.asarray(target, dtype=float).copy(),
                    "metrics": dict(info),
                    "attachment_enabled": bool(getattr(env, "attachment_enabled", False)),
                    "_attach_pending": bool(getattr(env, "_attach_pending", False)),
                }
                if record_raw_obs:
                    step_entry["raw_obs"] = dict(obs)
                transitions.append(step_entry)

            if env.has_renderer:
                env.render()
                if render_sleep > 0:
                    time.sleep(render_sleep)

            if phase_name == "settle_wait" and _metric_bool(info.get("final_success", False)):
                done = True
                break

            if (
                getattr(env, "grasp_mode", "attachment") == "physical"
                and phase_name in transport_regrasp_phases
                and local_step + 1 >= 8
                and not bool(getattr(env, "physical_grasp_ready", lambda: False)())
                and int(getattr(env, "_physical_left_contact_memory", 0)) <= 0
                and int(getattr(env, "_physical_right_contact_memory", 0)) <= 0
                and not (
                    getattr(env, "_robot_name", "") == "Panda"
                    and (
                        _metric_bool(info.get("endpoint_past_gap_final", False))
                        or float(info.get("thread_completion", 0.0)) >= 0.98
                    )
                )
            ):
                lost_physical_grasp = True
                break

            # pull_through: composite_soft 柔性好，线缆穿过间隙且端点超过间隙后，
            # 等夹爪接近目标再完成（确保线缆被拉到目标位置）。
            if phase_name == "pull_through" and local_step >= 4:
                _cable_model = getattr(env, "cable_model", "")
                if _cable_model in {"composite_soft", "composite_softened"}:
                    _low_int = _metric_bool(info.get("cable_low_intersects_pole_segment", False))
                    _past = _metric_bool(info.get("endpoint_past_gap_final", False))
                    _dist = float(np.linalg.norm(env._get_gripper_site_position() - raw_target))
                    if _low_int and _past and _dist < 0.05:
                        phase_completed = True
                        break

            if phase_name == "settle_wait":
                if local_step + 1 >= phase_cfg["hold_steps"]:
                    phase_completed = True
                    break
            elif phase_name == "release":
                if getattr(env, "grasp_mode", "attachment") == "physical":
                    if local_step + 1 >= phase_cfg["hold_steps"]:
                        phase_completed = True
                        break
                elif local_step + 1 >= phase_cfg["hold_steps"] and not getattr(env, "attachment_enabled", False):
                    phase_completed = True
                    break
            elif local_step + 1 >= phase_cfg["hold_steps"]:
                distance = float(np.linalg.norm(env._get_gripper_site_position() - raw_target))
                phase_ready = True
                if phase_name == "lower_after_gap":
                    phase_ready = _metric_bool(info["cable_low_intersects_pole_segment"])
                elif phase_name in {"close_grasp", "stabilize_grasp", "regrasp_table_close", "regrasp_table_stabilize"}:
                    phase_ready = bool(getattr(env, "physical_grasp_ready", lambda: False)())
                    if not phase_ready and phase_name in {"regrasp_table_close", "regrasp_table_stabilize"}:
                        phase_ready = _panda_post_thread_regrasp_candidate_ready(env)
                elif phase_name == "lift_clear" and getattr(env, "grasp_mode", "attachment") == "physical":
                    phase_ready = bool(getattr(env, "physical_grasp_lift_ready", lambda: False)())
                elif phase_name == "regrasp_table_lift":
                    phase_ready = (
                        bool(getattr(env, "physical_grasp_ready", lambda: False)())
                        or _panda_post_thread_regrasp_candidate_ready(env)
                    ) and float(info.get("physical_grasp_lift_height", 0.0)) >= 0.008
                elif phase_name == "regrasp_table_approach":
                    phase_ready = _metric_bool(info["endpoint_past_gap_final"]) and _metric_bool(info["settled_on_table_final"])
                elif phase_name == "pull_through":
                    phase_ready = _metric_bool(info["cable_low_intersects_pole_segment"]) and _metric_bool(
                        info["endpoint_past_gap_final"]
                    )
                    if getattr(env, "_robot_name", "") == "Panda" and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp", "composite_cable"}:
                        phase_ready = phase_ready and float(info["endpoint_goal_error_final"]) <= 0.18
                elif phase_name == "lay_down_endpoint":
                    phase_ready = (
                        _metric_bool(info["cable_low_intersects_pole_segment"])
                        and _metric_bool(info["settled_on_table_final"])
                    )
                    if getattr(env, "_robot_name", "") == "Panda" and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp", "composite_cable"}:
                        phase_ready = (
                            (
                                _metric_bool(info["endpoint_past_gap_final"])
                                and float(info["endpoint_goal_error_final"]) <= 0.20
                                and bool(getattr(env, "physical_grasp_ready", lambda: False)())
                            )
                            or (
                                _metric_bool(info["cable_low_intersects_pole_segment"])
                                and _metric_bool(info["settled_on_table_final"])
                                and bool(getattr(env, "physical_grasp_ready", lambda: False)())
                            )
                            or _metric_bool(info["straightened_final"])
                        )
                elif phase_name == "table_straighten":
                    phase_ready = (
                        _metric_bool(info["cable_low_intersects_pole_segment"])
                        and _metric_bool(info["settled_on_table_final"])
                        and _metric_bool(info["straightened_final"])
                    )
                elif phase_name == "press_to_table":
                    phase_ready = (
                        _metric_bool(info["cable_low_intersects_pole_segment"])
                        and float(info["endpoint_height_error_final"]) <= 0.006
                    )
                elif phase_name == "backoff_clearance":
                    phase_ready = _min_outer_clearance(env) >= float(
                        getattr(env, "pre_thread_outer_clearance_threshold", 0.03)
                    )
                if (
                    getattr(env, "grasp_mode", "attachment") == "physical"
                    and phase_name in {"close_grasp", "stabilize_grasp", "lift_clear", "regrasp_table_close", "regrasp_table_stabilize", "regrasp_table_lift"}
                    and phase_ready
                ):
                    phase_completed = True
                    break
                if (
                    getattr(env, "grasp_mode", "attachment") == "physical"
                    and phase_name == "pull_through"
                    and getattr(env, "_robot_name", "") == "Panda"
                    and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp"}
                    and phase_ready
                    and bool(getattr(env, "physical_grasp_ready", lambda: False)())
                ):
                    phase_completed = True
                    break
                if (
                    getattr(env, "grasp_mode", "attachment") == "physical"
                    and getattr(env, "_robot_name", "") == "Panda"
                    and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp"}
                    and phase_name in {"table_straighten", "press_to_table"}
                    and phase_ready
                ):
                    phase_completed = True
                    break
                if distance < phase_cfg["pos_tolerance"] and phase_ready:
                    phase_completed = True
                    break

            if done:
                break
        if done:
            break

        if (
            getattr(env, "grasp_mode", "attachment") == "physical"
            and phase_name in {"close_grasp", "stabilize_grasp", "lift_clear"}
            and not phase_completed
            and regrasp_attempts < max_regrasp_attempts
        ):
            regrasp_attempts = min(regrasp_attempts + 1, max_regrasp_attempts)
            env.set_attachment_enabled(False)
            if hasattr(env, "_physical_left_contact_memory"):
                env._physical_left_contact_memory = 0
            if hasattr(env, "_physical_right_contact_memory"):
                env._physical_right_contact_memory = 0
            current_eef = env._get_gripper_site_position().copy()
            current_eef[2] = max(current_eef[2], float(getattr(env, "table_top_z", 0.8)) + 0.09)
            previous_action_target = current_eef
            phase_idx = phase_name_to_index.get("approach_above_end", 0)
            continue

        if (
            getattr(env, "grasp_mode", "attachment") == "physical"
            and phase_name in {"regrasp_table_close", "regrasp_table_stabilize", "regrasp_table_lift"}
            and not phase_completed
        ):
            env.set_attachment_enabled(False)
            if hasattr(env, "_physical_left_contact_memory"):
                env._physical_left_contact_memory = 0
            if hasattr(env, "_physical_right_contact_memory"):
                env._physical_right_contact_memory = 0
            post_thread_regrasp_attempts += 1
            if post_thread_regrasp_attempts <= max_post_thread_regrasp_attempts:
                phase_idx = phase_name_to_index.get("regrasp_table_approach", phase_idx + 1)
            else:
                phase_idx = phase_name_to_index.get("table_straighten", phase_idx + 1)
            continue

        if (
            getattr(env, "grasp_mode", "attachment") == "physical"
            and phase_name in transport_regrasp_phases
            and lost_physical_grasp
            and regrasp_attempts < max_regrasp_attempts
        ):
            regrasp_attempts += 1
            env.set_attachment_enabled(False)
            if hasattr(env, "_physical_left_contact_memory"):
                env._physical_left_contact_memory = 0
            if hasattr(env, "_physical_right_contact_memory"):
                env._physical_right_contact_memory = 0
            current_eef = env._get_gripper_site_position().copy()
            current_eef[2] = max(current_eef[2], float(getattr(env, "table_top_z", 0.8)) + 0.09)
            previous_action_target = current_eef
            phase_idx = phase_name_to_index.get("approach_above_end", 0)
            continue

        if (
            getattr(env, "grasp_mode", "attachment") == "physical"
            and phase_name == "lift_clear"
            and phase_completed
        ):
            regrasp_attempts = 0

        if (
            getattr(env, "grasp_mode", "attachment") == "physical"
            and phase_name == "pull_through"
            and getattr(env, "_robot_name", "") == "Panda"
            and getattr(env, "cable_model", "") in {"flex", "flex_cable", "flexcomp"}
            and bool(getattr(env, "physical_grasp_ready", lambda: False)())
        ):
            post_thread_regrasp_attempts = 0
            # Skip table_straighten/press_to_table -- release immediately after threading.
            phase_idx = phase_name_to_index.get("release", phase_idx + 1)
            continue

        phase_idx += 1

    summary = summarize_episode(
        rows,
        env,
        total_reward,
        policy_name="robot_endpoint_oracle",
        episode_index=episode_index,
        seed=seed,
        phase_log=phase_log,
    )
    _finish_step_metric_recorder(step_recorder, summary, env)
    if live_config is not None and live_config.get("timeline_path"):
        if summary.get("final_success"):
            record_live_timeline_event(live_config, "success_check", "成功条件判定")
            record_live_timeline_event(live_config, "completed", "任务完成")
        write_live_timeline(live_config)
    if live_config is not None:
        _ensure_episode_live_frames(env, obs, live_config)
    return summary, phase_log, transitions


def rollout_policy_episode(
    env,
    policy,
    *,
    render_sleep=0.0,
    obs_keys=None,
    record_trajectory=False,
    episode_index=0,
    seed=None,
    policy_name="policy",
    live_config=None,
    attachment_mode="policy",
    attachment_schedule=None,
):
    from .attachment_controller import build_attachment_controller

    obs = env.reset()
    attach_ctrl = None
    if attachment_mode != "none" and getattr(env, "grasp_mode", "attachment") == "attachment":
        attach_ctrl = build_attachment_controller(
            env,
            replay_mode="recorded" if attachment_mode == "recorded" else "policy",
            attachment_schedule=attachment_schedule,
        )
        if attachment_mode == "recorded":
            attach_ctrl.reset(attachment_schedule)
        else:
            attach_ctrl.reset()
    if live_config is not None:
        live_config.pop("_last_display_frame", None)
        obs = _warmup_live_after_reset(env, obs, live_config)
        if not live_config.get("has_valid_frame"):
            live_config["_consecutive_valid"] = 0
            live_config["frame_status"] = "warming_up"
    if live_config is not None and live_config.get("timeline_path"):
        record_live_timeline_event(live_config, "reset", "环境初始化")
    if hasattr(policy, "reset"):
        policy.reset()

    total_reward = 0.0
    rows = []
    transitions = []

    done = False
    step_recorder = _create_step_metric_recorder(env, episode_index, live_config)
    while not done and len(rows) < env.horizon:
        action = policy.act(obs) if hasattr(policy, "act") else policy(obs)
        action = clip_action(env, action)
        if attach_ctrl is not None:
            attach_ctrl.pre_step(action, info=rows[-1] if rows else None)
        prev_obs = obs
        step_t0 = time.perf_counter()
        obs, reward, done, info = env.step(action)
        step_wall_sec = time.perf_counter() - step_t0
        total_reward += reward
        rows.append(dict(info))
        _record_step_metric(
            step_recorder, len(rows) - 1, action, reward, done, info, step_wall_sec=step_wall_sec
        )

        if live_config is not None:
            live_config["step"] = int(live_config.get("step", 0)) + 1
            maybe_capture_and_persist_live_frame(env, obs, live_config)
            write_live_status(live_config)

        if record_trajectory:
            eef_after = env._get_gripper_site_position()
            transitions.append(
                {
                    "obs": obs_to_vector(prev_obs, obs_keys or DEFAULT_OBS_KEYS),
                    "next_obs": obs_to_vector(obs, obs_keys or DEFAULT_OBS_KEYS),
                    "action": action.copy(),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": dict(info),
                    "phase": "policy",
                    "eef_pos": eef_after.copy(),
                    "eef_target": eef_after.copy(),
                    "metrics": dict(info),
                    "attachment_enabled": bool(getattr(env, "attachment_enabled", False)),
                }
            )

        if env.has_renderer:
            env.render()
            if render_sleep > 0:
                time.sleep(render_sleep)

    summary = summarize_episode(
        rows,
        env,
        total_reward,
        policy_name=policy_name,
        episode_index=episode_index,
        seed=seed,
    )
    _finish_step_metric_recorder(step_recorder, summary, env)
    if attach_ctrl is not None and hasattr(attach_ctrl, "attachment_stats"):
        summary.update(attach_ctrl.attachment_stats())
    if live_config is not None and live_config.get("timeline_path"):
        write_live_timeline(live_config)
    if live_config is not None:
        _ensure_episode_live_frames(env, obs, live_config)
    return summary, transitions


def save_dataset(path, trajectories, obs_keys=DEFAULT_OBS_KEYS, metadata=None, episode_metadata=None):
    validate_transition_trajectories(trajectories, metadata=metadata)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    observations = np.concatenate([np.stack([step["obs"] for step in traj], axis=0) for traj in trajectories], axis=0)
    next_observations = np.concatenate([np.stack([step["next_obs"] for step in traj], axis=0) for traj in trajectories], axis=0)
    actions = np.concatenate([np.stack([step["action"] for step in traj], axis=0) for traj in trajectories], axis=0)
    rewards = np.concatenate([np.asarray([step["reward"] for step in traj], dtype=np.float32) for traj in trajectories], axis=0)
    dones = np.concatenate([np.asarray([step["done"] for step in traj], dtype=bool) for traj in trajectories], axis=0)
    eef_positions = np.concatenate(
        [np.stack([np.asarray(step.get("eef_pos", np.full(3, np.nan)), dtype=np.float32) for step in traj], axis=0) for traj in trajectories],
        axis=0,
    )
    eef_targets = np.concatenate(
        [np.stack([np.asarray(step.get("eef_target", np.full(3, np.nan)), dtype=np.float32) for step in traj], axis=0) for traj in trajectories],
        axis=0,
    )
    phases = np.concatenate([np.asarray([str(step.get("phase", "")) for step in traj], dtype=object) for traj in trajectories], axis=0)
    step_metrics = np.concatenate(
        [
            np.asarray([json.dumps(_jsonable(step.get("metrics", {})), ensure_ascii=False) for step in traj], dtype=object)
            for traj in trajectories
        ],
        axis=0,
    )
    episode_lengths = np.array([len(traj) for traj in trajectories], dtype=np.int32)
    episode_metadata = episode_metadata or [{} for _ in trajectories]

    np.savez_compressed(
        path,
        observations=observations.astype(np.float32),
        next_observations=next_observations.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rewards,
        dones=dones,
        eef_positions=eef_positions.astype(np.float32),
        eef_targets=eef_targets.astype(np.float32),
        phases=phases,
        step_metrics=step_metrics,
        episode_lengths=episode_lengths,
        obs_keys=np.array(obs_keys),
        metadata=json.dumps(metadata or {}),
        episode_metadata=np.array([json.dumps(item) for item in episode_metadata], dtype=object),
    )


RESULT_FIELDNAMES = [
    "episode",
    "policy",
    "seed",
    "difficulty",
    "return",
    "steps",
    "final_success",
    "ever_success",
    "threaded_final",
    "ever_threaded",
    "endpoint_region_final",
    "ever_endpoint_region",
    "endpoint_past_gap_final",
    "ever_endpoint_past_gap",
    "final_line_crosses_gap",
    "ever_final_line_crosses_gap",
    "cable_intersects_pole_segment",
    "ever_cable_intersects_pole_segment",
    "cable_low_intersects_pole_segment",
    "ever_cable_low_intersects_pole_segment",
    "straightened_final",
    "ever_straightened",
    "settled_on_table_final",
    "ever_settled_on_table",
    "anchor_stable_final",
    "ever_anchor_stable",
    "anchor_error_final",
    "min_pole_clearance_final",
    "min_outer_clearance_final",
    "endpoint_goal_error_final",
    "straightness_error_final",
    "tabletop_spread_final",
    "endpoint_height_error_final",
    "thread_completion_final",
    "passed_keypoint_ratio",
    "gate_deviation",
    "post_collision_count",
    "cable_on_table",
    "thread_completion_max",
    "peak_height_excess_final",
    "gap_cross_x_final",
    "gap_cross_z_final",
    "thread_cross_value_final",
    "reset_root_x",
    "reset_root_y",
    "reset_root_z",
    "reset_shape_noise_l2",
    "reset_endpoint_goal_init_error",
    "phase_count",
]


def write_results_csv(path, rows):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    if not rows:
        return {
            "episodes": 0,
            "final_success_rate": 0.0,
            "ever_success_rate": 0.0,
            "mean_endpoint_goal_error_final": 0.0,
            "mean_straightness_error_final": 0.0,
            "mean_anchor_error_final": 0.0,
            "mean_tabletop_spread_final": 0.0,
            "mean_thread_completion_max": 0.0,
            "attachment_mode": None,
            "mean_attach_count": 0.0,
            "mean_detach_count": 0.0,
            "mean_attachment_enabled_ratio": 0.0,
        }
    attach_modes = [row.get("attachment_mode") for row in rows if row.get("attachment_mode")]
    attach_counts = [row.get("attach_count", row.get("attach_transitions")) for row in rows]
    detach_counts = [row.get("detach_count", row.get("detach_transitions")) for row in rows]
    attach_ratios = [row.get("attachment_enabled_ratio") for row in rows if row.get("attachment_enabled_ratio") is not None]
    return {
        "episodes": len(rows),
        "final_success_rate": float(np.mean([row["final_success"] for row in rows])),
        "ever_success_rate": float(np.mean([row["ever_success"] for row in rows])),
        "mean_endpoint_goal_error_final": float(np.mean([row["endpoint_goal_error_final"] for row in rows])),
        "mean_straightness_error_final": float(np.mean([row["straightness_error_final"] for row in rows])),
        "mean_anchor_error_final": float(np.mean([row["anchor_error_final"] for row in rows])),
        "mean_tabletop_spread_final": float(np.mean([row["tabletop_spread_final"] for row in rows])),
        "mean_thread_completion_max": float(np.mean([row["thread_completion_max"] for row in rows])),
        "attachment_mode": attach_modes[0] if attach_modes else None,
        "mean_attach_count": float(np.mean([float(v) for v in attach_counts if v is not None])) if any(v is not None for v in attach_counts) else 0.0,
        "mean_detach_count": float(np.mean([float(v) for v in detach_counts if v is not None])) if any(v is not None for v in detach_counts) else 0.0,
        "mean_attachment_enabled_ratio": float(np.mean(attach_ratios)) if attach_ratios else 0.0,
    }


class RandomPolicy:
    def __init__(self, env, seed=None):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def reset(self):
        return None

    def act(self, obs):
        low, high = self.env.action_spec
        return self.rng.uniform(low, high).astype(np.float32)


class RobomimicPolicyAdapter:
    def __init__(self, checkpoint, obs_keys=None, device="cpu"):
        self.checkpoint = str(Path(checkpoint).expanduser())
        self.device = device
        try:
            import torch
            import robomimic.utils.file_utils as FileUtils
        except ImportError as exc:
            raise ImportError("robomimic is not installed in the robosuite environment.") from exc

        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=self.checkpoint,
            device=torch.device(device),
            verbose=False,
        )
        if obs_keys is None:
            inferred_obs_keys = self.ckpt_dict.get("shape_metadata", {}).get("all_obs_keys", None)
            if inferred_obs_keys is None:
                inferred_obs_keys = DEFAULT_OBS_KEYS
            self.obs_keys = [str(key) for key in inferred_obs_keys]
        else:
            self.obs_keys = list(obs_keys)

        shape_metadata = self.ckpt_dict.get("shape_metadata", {})
        self.obs_shapes = dict(shape_metadata.get("all_shapes") or {})

    def reset(self):
        if hasattr(self.policy, "start_episode"):
            self.policy.start_episode()

    def act(self, obs):
        missing = [key for key in self.obs_keys if key not in obs]
        if missing:
            raise KeyError(
                f"Missing observation keys required by robomimic policy: {missing}. "
                f"Available keys: {sorted(obs.keys())}"
            )
        obs_dict = {}
        for key in self.obs_keys:
            value = np.asarray(obs[key])
            if key.endswith("_image"):
                expected_shape = self.obs_shapes.get(key)
                if expected_shape and value.ndim == 3:
                    expected_shape = tuple(int(dim) for dim in expected_shape)
                    if expected_shape[0] in (1, 3, 4):
                        target_height, target_width = expected_shape[1:3]
                    else:
                        target_height, target_width = expected_shape[0:2]
                    if value.shape[0:2] != (target_height, target_width):
                        from PIL import Image

                        source = value.astype(np.uint8) if value.dtype != np.uint8 else value
                        value = np.asarray(
                            Image.fromarray(source).resize(
                                (target_width, target_height), Image.Resampling.LANCZOS
                            )
                        )
                obs_dict[key] = value.astype(np.uint8) if value.dtype == np.uint8 else value
            else:
                expected_shape = self.obs_shapes.get(key)
                if expected_shape:
                    expected_shape = tuple(int(dim) for dim in expected_shape)
                    if value.shape != expected_shape and value.size == int(np.prod(expected_shape)):
                        value = value.reshape(expected_shape)
                obs_dict[key] = value.astype(np.float32)
        import robomimic.utils.obs_utils as ObsUtils

        obs_dict = ObsUtils.process_obs_dict(obs_dict)
        if hasattr(self.policy, "__call__"):
            try:
                action = self.policy(ob=obs_dict)
            except TypeError:
                action = self.policy(obs_dict)
        elif hasattr(self.policy, "act"):
            action = self.policy.act(obs_dict)
        else:
            raise AttributeError("Loaded robomimic policy does not expose a callable action interface.")

        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 1:
            action = action[0]
        return action
