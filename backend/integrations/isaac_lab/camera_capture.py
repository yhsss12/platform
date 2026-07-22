"""Isaac Lab Stack Cube viewport camera capture utilities."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

STACK_CUBE_VIEWER_EYE = (1.35, 1.35, 1.05)
STACK_CUBE_VIEWER_LOOKAT = (0.5, 0.0, 0.35)

# Default camera keys written to HDF5 obs/ (extensible for additional sensors).
DEFAULT_AGENTVIEW_KEY = "agentview_image"
DEFAULT_WRIST_CAMERA_KEY = "robot0_eye_in_hand_image"


def configure_single_env_live_viewer(env_cfg, env_index: int = 0) -> None:
    """Configure Isaac env viewer for third-person agentview capture."""
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = max(0, int(env_index))
    env_cfg.viewer.eye = STACK_CUBE_VIEWER_EYE
    env_cfg.viewer.lookat = STACK_CUBE_VIEWER_LOOKAT


def focus_live_viewport(env, env_index: int = 0) -> None:
    env_index = max(0, min(int(env_index), max(0, env.num_envs - 1)))
    vcc = getattr(env, "viewport_camera_controller", None)
    if vcc is not None:
        vcc.set_view_env_index(env_index)
        vcc.update_view_location(eye=STACK_CUBE_VIEWER_EYE, lookat=STACK_CUBE_VIEWER_LOOKAT)
        return
    if hasattr(env, "sim") and hasattr(env, "scene"):
        origin = env.scene.env_origins[env_index].detach().cpu().numpy()
        eye = origin + np.array(STACK_CUBE_VIEWER_EYE, dtype=float)
        target = origin + np.array(STACK_CUBE_VIEWER_LOOKAT, dtype=float)
        env.sim.set_camera_view(eye=eye, target=target)


def _rgb_is_valid(arr: np.ndarray, *, min_mean: float = 2.0, min_std: float = 2.0) -> bool:
    if arr.size == 0:
        return False
    return float(arr.mean()) > min_mean and float(arr.std()) > min_std


def capture_viewport_rgb(
    env,
    simulation_app: Any = None,
    *,
    env_index: int = 0,
    max_attempts: int = 12,
) -> Optional[np.ndarray]:
    """Capture RGB frame from Isaac viewport (H, W, 3) uint8."""
    if not hasattr(env, "render"):
        return None
    for attempt in range(max_attempts):
        focus_live_viewport(env, env_index)
        if hasattr(env, "sim"):
            env.sim.render()
        if simulation_app is not None:
            simulation_app.update()
        for recompute in (False, True):
            try:
                rgb = env.render(recompute=recompute)
            except Exception as exc:
                logger.debug("viewport capture attempt %s failed: %s", attempt, exc)
                continue
            if rgb is None:
                continue
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[2] >= 3 and _rgb_is_valid(arr):
                return arr[:, :, :3].copy()
    return None


def warmup_viewport_capture(
    env,
    simulation_app: Any = None,
    *,
    env_index: int = 0,
    max_attempts: int = 60,
) -> bool:
    """Prime viewport rendering before per-step capture during recording."""
    frame = capture_viewport_rgb(
        env,
        simulation_app,
        env_index=env_index,
        max_attempts=max_attempts,
    )
    return frame is not None


def resize_rgb_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize RGB uint8 frame to (height, width, 3)."""
    target_h = max(1, int(height))
    target_w = max(1, int(width))
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB frame (H,W,3), got shape={arr.shape}")
    rgb = arr[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
        return rgb
    try:
        import cv2

        return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Nearest-neighbor fallback without OpenCV.
        y_idx = (np.linspace(0, rgb.shape[0] - 1, target_h)).astype(int)
        x_idx = (np.linspace(0, rgb.shape[1] - 1, target_w)).astype(int)
        return rgb[np.ix_(y_idx, x_idx)]
