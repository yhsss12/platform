"""Isaac Lab live frame 有效性检测（排除全黑 warmup 缓冲）。"""

from __future__ import annotations

from pathlib import Path


def frame_image_is_valid(path: Path, *, min_mean: float = 2.0, min_std: float = 2.0) -> bool:
    if not path.is_file() or path.stat().st_size <= 64:
        return False
    try:
        import numpy as np
        from PIL import Image

        arr = np.array(Image.open(path).convert("RGB"))
        if arr.size == 0:
            return False
        mean = float(arr.mean())
        std = float(arr.std())
        return mean > min_mean and std > min_std
    except OSError:
        return False
