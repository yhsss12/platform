from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_video(frames: list[np.ndarray], output_path: Path, *, fps: int = 20) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return {"ok": False, "error": "no_frames", "path": str(output_path)}

    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError as exc:
            return {"ok": False, "error": f"imageio_missing: {exc}", "path": str(output_path)}

    rgb_frames = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb_frames.append(arr)

    try:
        imageio.mimsave(str(output_path), rgb_frames, fps=fps)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "path": str(output_path)}

    return {
        "ok": True,
        "path": str(output_path),
        "frameCount": len(rgb_frames),
        "fps": fps,
    }
