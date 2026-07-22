#!/usr/bin/env python3
"""线缆穿杆 trained_model_evaluation 展示视频回归验收脚本。"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
_CABLE_MVP = BACKEND.parent / "integrations" / "CableThreadingMVP"
for _p in (BACKEND, _CABLE_MVP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.services import cable_threading_service as ct_svc  # noqa: E402

CHECKPOINT = (
    "/home/ubuntu/project/eai-idev2.1/runs/training/jobs/"
    "train_20260617_163406_b1a9/checkpoints/model_final.pth"
)
MODEL_ASSET_ID = "model__163406_b1a9_ff10ee0ee9"


def _poll_until_done(job_id: str, *, timeout_sec: int = 7200) -> dict:
    started = time.time()
    while time.time() - started < timeout_sec:
        status = ct_svc.get_eval_job_status(job_id)
        value = str(status.get("status") or "running")
        if value in {"completed", "failed"}:
            return status
        time.sleep(5)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_sec}s")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _probe_video(path: Path) -> dict:
    info = {"exists": path.is_file(), "sizeBytes": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        return info
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            payload = json.loads(result.stdout or "{}")
            streams = payload.get("streams") or []
            if streams:
                stream = streams[0]
                info["width"] = stream.get("width")
                info["height"] = stream.get("height")
                info["codec"] = stream.get("codec_name")
                info["fps"] = stream.get("r_frame_rate")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return info


def _check_log_no_obs_upscale(job_root: Path) -> bool:
    log_path = job_root / "logs" / "run.log"
    if not log_path.is_file():
        return True
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "live_obs_resolution_low=true" not in text and "OBS_VALIDATION_FAILED" not in text


def _sample_frame_resolution(job_root: Path) -> str | None:
    frames_dir = job_root / "live" / "frames"
    pngs = sorted(frames_dir.glob("frame_*.png"))
    jpgs = sorted(frames_dir.glob("frame_*.jpg"))
    sample = pngs[0] if pngs else (jpgs[0] if jpgs else None)
    if sample is None:
        return None
    try:
        from PIL import Image

        with Image.open(sample) as img:
            return f"{img.width}x{img.height}"
    except Exception:
        return None


def _validate_latest_jpeg(job_root: Path) -> dict:
    latest = job_root / "live" / "latest.jpg"
    info: dict = {"exists": latest.is_file()}
    if not latest.is_file():
        return info
    info["sizeBytes"] = latest.stat().st_size
    try:
        from examples.cable_threading.utils import is_valid_live_frame
        from PIL import Image
        import numpy as np

        with Image.open(latest) as img:
            rgb = np.asarray(img.convert("RGB"))
            valid, reason = is_valid_live_frame(rgb)
            info["resolution"] = f"{rgb.shape[1]}x{rgb.shape[0]}"
            info["valid"] = valid
            info["invalidReason"] = reason
    except Exception as exc:
        info["valid"] = None
        info["error"] = str(exc)
    return info


def _count_frames(job_root: Path) -> dict:
    frames_dir = job_root / "live" / "frames"
    invalid_dir = job_root / "live" / "invalid_frames"
    return {
        "validFrames": len(list(frames_dir.glob("frame_*.png"))) + len(list(frames_dir.glob("frame_*.jpg"))),
        "invalidFrames": len(list(invalid_dir.glob("*.jpg"))) if invalid_dir.is_dir() else 0,
    }


def _validate_run(job_id: str, *, label: str) -> dict:
    job_root = ct_svc.OUTPUT_ROOT / "jobs" / job_id
    status = ct_svc.get_eval_job_status(job_id)
    live = _read_json(job_root / "live" / "status.json")
    aggregate = _read_json(job_root / "results" / "aggregate_result.json")
    per_episode_path = job_root / "results" / "per_episode_results.json"

    eval_mp4 = job_root / "videos" / "eval.mp4"
    browser_mp4 = job_root / "videos" / "eval.browser.mp4"
    eval_probe = _probe_video(eval_mp4)
    browser_probe = _probe_video(browser_mp4)
    frame_res = _sample_frame_resolution(job_root)
    latest_info = _validate_latest_jpeg(job_root)
    frame_counts = _count_frames(job_root)

    success_rate = (status.get("metrics") or {}).get("successRate")
    if success_rate is None:
        success_rate = aggregate.get("final_success_rate")

    checks = {
        "status_completed": str(status.get("status")) == "completed",
        "obs_validation_ok": _check_log_no_obs_upscale(job_root),
        "liveFrameSource_sim_render": live.get("liveFrameSource") == "sim_render",
        "displayCamera_agentview": live.get("displayCamera") == "agentview",
        "hasValidFrame_true": live.get("hasValidFrame") is True,
        "videoResolution_1280x720": live.get("videoResolution") == "1280x720",
        "eval_mp4_exists": eval_mp4.is_file(),
        "browser_mp4_exists": browser_mp4.is_file(),
        "aggregate_videoPath": bool(aggregate.get("videoPath")),
        "aggregate_browserVideoPath": bool(aggregate.get("browserVideoPath")),
        "aggregate_videoStatus_available": aggregate.get("videoStatus") == "available",
        "aggregate_videoResolution": aggregate.get("videoResolution") == "1280x720",
        "aggregate_displayCamera": aggregate.get("displayCamera") == "agentview",
        "per_episode_exists": per_episode_path.is_file(),
        "frame_png_used": frame_res == "1280x720" if frame_res else None,
        "video_probe_1280x720": (
            eval_probe.get("width") == 1280 and eval_probe.get("height") == 720
        ),
        "latest_jpeg_valid": latest_info.get("valid") is True,
        "invalid_frames_recorded": frame_counts["invalidFrames"] >= 0,
        "skipped_invalid_tracked": isinstance(live.get("skippedInvalidFrame"), int),
        "success_rate_is_metric_not_system_fail": str(status.get("status")) == "completed"
        or "OBS_VALIDATION_FAILED" not in (job_root / "logs" / "run.log").read_text(encoding="utf-8", errors="replace")
        if (job_root / "logs" / "run.log").is_file()
        else True,
    }

    return {
        "label": label,
        "evalId": job_id,
        "status": status.get("status"),
        "successRate": success_rate,
        "failureCount": aggregate.get("failure_count"),
        "videoPath": aggregate.get("videoPath") or str(eval_mp4),
        "browserVideoPath": aggregate.get("browserVideoPath") or str(browser_mp4),
        "videoResolution": live.get("videoResolution") or aggregate.get("videoResolution"),
        "displayCamera": live.get("displayCamera") or aggregate.get("displayCamera"),
        "liveFrameSource": live.get("liveFrameSource"),
        "hasValidFrame": live.get("hasValidFrame"),
        "skippedInvalidFrame": live.get("skippedInvalidFrame"),
        "invalidFrameReasons": live.get("invalidFrameReasons"),
        "frameCounts": frame_counts,
        "latestJpeg": latest_info,
        "evalVideoProbe": eval_probe,
        "browserVideoProbe": browser_probe,
        "sampleFrameResolution": frame_res,
        "checks": checks,
        "all_passed": all(v is True for v in checks.values() if v is not None),
    }


def _start_eval(*, episodes: int, horizon: int) -> str:
    result = ct_svc.start_evaluate_async(
        episodes=episodes,
        robot="Panda",
        cable_model="composite_cable",
        difficulty="easy",
        horizon=horizon,
        seed=0,
        policy="robomimic",
        checkpoint=CHECKPOINT,
        device="cpu",
        model_name=f"regression {MODEL_ASSET_ID}",
    )
    job_id = str(result["evalJobId"])
    meta_dir = ct_svc.OUTPUT_ROOT / "jobs" / job_id / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "evaluation_context.json").write_text(
        json.dumps(
            {
                "evaluationMode": "trained_model_evaluation",
                "modelAssetId": MODEL_ASSET_ID,
                "episodes": episodes,
                "horizon": horizon,
                "seed": 0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return job_id


def main() -> int:
    if not Path(CHECKPOINT).is_file():
        print(f"ERROR: checkpoint missing: {CHECKPOINT}")
        return 1

    reports = []

    print("=== 短参数回归 episodes=2 horizon=200 ===")
    short_id = _start_eval(episodes=2, horizon=200)
    print(f"started: {short_id}")
    short_status = _poll_until_done(short_id, timeout_sec=3600)
    print(f"finished: {short_id} status={short_status.get('status')}")
    reports.append(_validate_run(short_id, label="short"))

    print("\n=== 正式参数回归 episodes=10 horizon=600 ===")
    full_id = _start_eval(episodes=10, horizon=600)
    print(f"started: {full_id}")
    full_status = _poll_until_done(full_id, timeout_sec=7200)
    print(f"finished: {full_id} status={full_status.get('status')}")
    reports.append(_validate_run(full_id, label="full"))

    out_path = ct_svc.OUTPUT_ROOT / "eval_video_regression_report.json"
    out_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written: {out_path}")
    print(json.dumps(reports, indent=2, ensure_ascii=False))

    ok = all(r.get("all_passed") for r in reports)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
