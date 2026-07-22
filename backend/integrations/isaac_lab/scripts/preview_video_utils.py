"""从 live frames 合成浏览器可播放 preview.mp4（H.264 / yuv420p / faststart）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def resolve_ffmpeg_executable() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def build_preview_from_frames(frames_dir: Path, preview_out: Path, fps: int = 10) -> bool:
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return False

    preview_out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg:
        pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview_out),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and preview_out.is_file() and preview_out.stat().st_size > 0:
            return True

    try:
        import cv2
    except ImportError:
        return False

    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(preview_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return False
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        if img.mean() < 2.0 and img.std() < 2.0:
            continue
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        writer.write(img)
    writer.release()
    return preview_out.is_file() and preview_out.stat().st_size > 0
