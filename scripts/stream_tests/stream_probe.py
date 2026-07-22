import argparse
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class ProbeResult:
    ok: bool
    duration_sec: float
    frames: int
    bytes: int
    fps: float
    stalls: int
    max_stall_sec: float
    error: Optional[str] = None


BOUNDARY_RE = re.compile(br"\r?\n--frame\r?\n", re.IGNORECASE)


def probe_mjpeg(url: str, *, duration_sec: float, stall_timeout_sec: float, headers: dict) -> ProbeResult:
    start = time.time()
    last_frame_ts = start
    frames = 0
    total_bytes = 0
    stalls = 0
    max_stall = 0.0

    try:
        with requests.get(url, stream=True, timeout=(5, stall_timeout_sec), headers=headers) as r:
            if r.status_code // 100 != 2:
                return ProbeResult(
                    ok=False,
                    duration_sec=0.0,
                    frames=0,
                    bytes=0,
                    fps=0.0,
                    stalls=0,
                    max_stall_sec=0.0,
                    error=f"HTTP {r.status_code}",
                )

            buf = b""
            for chunk in r.iter_content(chunk_size=64 * 1024):
                now = time.time()
                if now - start >= duration_sec:
                    break
                if not chunk:
                    continue
                total_bytes += len(chunk)
                buf += chunk
                parts = BOUNDARY_RE.split(buf)
                if len(parts) <= 1:
                    if now - last_frame_ts >= stall_timeout_sec:
                        stalls += 1
                        max_stall = max(max_stall, now - last_frame_ts)
                        last_frame_ts = now
                    continue
                buf = parts[-1]
                new_frames = max(0, len(parts) - 1)
                frames += new_frames
                last_frame_ts = now

    except requests.exceptions.ReadTimeout:
        now = time.time()
        stalls += 1
        max_stall = max(max_stall, now - last_frame_ts)
    except Exception as e:
        return ProbeResult(
            ok=False,
            duration_sec=time.time() - start,
            frames=frames,
            bytes=total_bytes,
            fps=(frames / max(1e-6, time.time() - start)),
            stalls=stalls,
            max_stall_sec=max_stall,
            error=str(e),
        )

    dur = time.time() - start
    return ProbeResult(
        ok=True,
        duration_sec=dur,
        frames=frames,
        bytes=total_bytes,
        fps=(frames / max(1e-6, dur)),
        stalls=stalls,
        max_stall_sec=max_stall,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="MJPEG URL, e.g. http://localhost:8000/api/stream/camera1?device_id=1")
    ap.add_argument("--duration-sec", type=float, default=30.0)
    ap.add_argument("--stall-timeout-sec", type=float, default=3.0)
    ap.add_argument("--auth", default="", help="Bearer token (optional)")
    args = ap.parse_args()

    headers = {}
    if args.auth.strip():
        headers["Authorization"] = f"Bearer {args.auth.strip()}"

    res = probe_mjpeg(
        args.url,
        duration_sec=float(args.duration_sec),
        stall_timeout_sec=float(args.stall_timeout_sec),
        headers=headers,
    )
    line = (
        f"ok={res.ok} duration_sec={res.duration_sec:.2f} frames={res.frames} fps={res.fps:.2f} "
        f"bytes={res.bytes} stalls={res.stalls} max_stall_sec={res.max_stall_sec:.2f}"
    )
    if res.error:
        line += f" error={res.error}"
    print(line)
    return 0 if res.ok else 2


if __name__ == "__main__":
    sys.exit(main())

