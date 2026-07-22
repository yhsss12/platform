"""Isaac Lab MP4 浏览器兼容探测与转码（mp4v → H.264/yuv420p）。"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def browser_cached_video_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}.browser.mp4")


def resolve_ffmpeg_executable() -> Optional[str]:
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found

    python = (settings.ISAACLAB_PYTHON or "").strip()
    if not python:
        return None

    try:
        result = subprocess.run(
            [python, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            candidate = result.stdout.strip()
            if candidate and Path(candidate).is_file():
                return candidate
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("resolve ffmpeg via ISAACLAB_PYTHON failed: %s", exc)
    return None


def probe_mp4_codec(path: Path) -> dict[str, object]:
    """轻量探测 MP4 容器与常见 codec 标记。"""
    info: dict[str, object] = {
        "exists": False,
        "size": 0,
        "hasVideoStreamHint": False,
        "codec": "unknown",
        "browserCompatible": False,
        "moovBeforeMdat": False,
    }
    if not path.is_file():
        return info

    try:
        size = path.stat().st_size
        data = path.read_bytes() if size <= 64 * 1024 * 1024 else path.read_bytes()[:8_000_000]
    except OSError:
        return info

    info["exists"] = True
    info["size"] = size
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    info["moovBeforeMdat"] = moov >= 0 and mdat >= 0 and moov < mdat

    if b"avc1" in data or b"h264" in data or b"avc3" in data:
        info["codec"] = "h264"
        info["hasVideoStreamHint"] = True
        info["browserCompatible"] = True
    elif b"mp4v" in data:
        info["codec"] = "mp4v"
        info["hasVideoStreamHint"] = True
        info["browserCompatible"] = False
    elif b"ftyp" in data and b"mdat" in data:
        info["hasVideoStreamHint"] = True

    return info


def is_browser_compatible_mp4(path: Path) -> bool:
    return bool(probe_mp4_codec(path).get("browserCompatible"))


def transcode_to_browser_mp4(source: Path, dest: Path, *, timeout: int = 600) -> tuple[bool, Optional[str]]:
    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg is None:
        return False, "ffmpeg not available for browser transcode"

    dest.parent.mkdir(parents=True, exist_ok=True)
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
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"ffmpeg transcode failed: {exc}"

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-400:]
        return False, f"ffmpeg exit={result.returncode}: {tail}"

    if not dest.is_file() or dest.stat().st_size <= 0:
        return False, "transcoded mp4 missing or empty"

    if not is_browser_compatible_mp4(dest):
        return False, "transcoded mp4 still not browser-compatible"

    return True, None


def ensure_browser_playable_mp4(source: Path) -> tuple[Optional[Path], Optional[str]]:
    """返回浏览器可播放 MP4；必要时转码并缓存为 *.browser.mp4。"""
    if not source.is_file() or source.stat().st_size <= 0:
        return None, "video file missing"

    if is_browser_compatible_mp4(source):
        return source, None

    cached = browser_cached_video_path(source)
    if cached.is_file():
        try:
            if cached.stat().st_mtime >= source.stat().st_mtime and is_browser_compatible_mp4(cached):
                return cached, "transcoded_cache"
        except OSError:
            pass

    ok, note = transcode_to_browser_mp4(source, cached)
    if ok:
        return cached, note or "transcoded"

    probe = probe_mp4_codec(source)
    if probe.get("codec") == "mp4v":
        return None, "回放视频编码不兼容（mp4v），且自动转码失败"
    return None, note or "video not browser-compatible"
