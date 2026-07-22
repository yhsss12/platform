#!/usr/bin/env python3
"""Organize official adapter outputs into platform job layout for acceptance."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_ID = "isaacsim_franka_pick_place"
TASK_NAME = "Franka 物体搬运"
SIMULATOR = "Isaac Sim"
ROBOT = "Franka Panda"
EXPERT_SOURCE = "NVIDIA Isaac Sim 官方 FrankaPickPlace controller"
FORBIDDEN_VIDEO_HINTS = ("cable", "thread", "dual_arm", "dac_gen", "ct_gen", "threading")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def contains_forbidden_video_hint(path: str | Path) -> bool:
    lowered = str(path).replace("\\", "/").lower()
    return any(hint in lowered for hint in FORBIDDEN_VIDEO_HINTS)


def organize_job_dir(job_dir: Path, *, episode_id: str = "ep_000001") -> dict[str, Any]:
    job_dir = job_dir.resolve()
    ep_dir = job_dir / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "videos").mkdir(parents=True, exist_ok=True)
    (job_dir / "results").mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    (job_dir / "metadata").mkdir(parents=True, exist_ok=True)

    run_result = _read_json(job_dir / "run_result.json")
    adapter_metrics = _read_json(job_dir / "metrics.json")
    adapter_manifest = _read_json(job_dir / "episode_manifest.json")
    adapter_trajectory = _read_json(job_dir / "trajectory.json")

    metrics = adapter_metrics or run_result
    manifest = adapter_manifest or {}
    if run_result:
        manifest.setdefault("episode_id", episode_id)
        manifest.setdefault("task_id", TASK_ID)
        manifest.setdefault("video_status", run_result.get("video_status"))
        manifest.setdefault("video_available", run_result.get("video_available"))
        manifest.setdefault("video_path", run_result.get("video_path"))

    if adapter_trajectory:
        _write_json(ep_dir / "trajectory.json", adapter_trajectory)
    if metrics:
        _write_json(ep_dir / "metrics.json", metrics)
    if manifest:
        rel_video = f"videos/{episode_id}.mp4" if manifest.get("video_available") else None
        manifest["metrics_path"] = f"episodes/{episode_id}/metrics.json"
        manifest["trajectory_path"] = f"episodes/{episode_id}/trajectory.json"
        manifest["video_path"] = rel_video
        _write_json(ep_dir / "episode_manifest.json", manifest)

    video_path = job_dir / "videos" / f"{episode_id}.mp4"
    success = bool(metrics.get("success", run_result.get("success")))
    video_available = bool(
        manifest.get("video_available", run_result.get("video_available", video_path.is_file()))
    )
    video_status = str(
        manifest.get("video_status") or run_result.get("video_status") or ("available" if video_available else "pending")
    )

    aggregate = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "simulator": SIMULATOR,
        "robot": ROBOT,
        "expert_source": EXPERT_SOURCE,
        "num_episodes": 1,
        "success_episodes": 1 if success else 0,
        "failed_episodes": 0 if success else 1,
        "success_rate": 1.0 if success else 0.0,
        "runtime_mode": "isaacsim",
        "created_at": _utc_now_iso(),
    }
    _write_json(job_dir / "results" / "aggregate_metrics.json", aggregate)
    _write_json(
        job_dir / "results" / "per_episode_results.json",
        {
            "episodes": [
                {
                    "episode_id": episode_id,
                    "success": success,
                    "video_available": video_available,
                    "video_status": video_status,
                }
            ]
        },
    )

    dataset_manifest = {
        "datasetId": f"dataset_{TASK_ID}_verify",
        "jobId": job_dir.name,
        "task_id": TASK_ID,
        "source_task_id": TASK_ID,
        "task_name": TASK_NAME,
        "taskType": TASK_ID,
        "simulator": SIMULATOR,
        "simulatorBackend": "isaacsim",
        "robot": ROBOT,
        "expert_source": EXPERT_SOURCE,
        "sourceType": "simulation_generated",
        "sourceJobId": job_dir.name,
        "episodes": 1,
        "episode_count": 1,
        "totalEpisodes": 1,
        "successfulEpisodes": 1 if success else 0,
        "failedEpisodes": 0 if success else 1,
        "success_rate": aggregate["success_rate"],
        "datasetFormat": "episode_manifest",
        "created_at": _utc_now_iso(),
        "episode_manifests": [f"episodes/{episode_id}/episode_manifest.json"],
        "video_available": video_available,
        "video_status": video_status,
        "videoStatus": video_status,
        "runtime_mode": "isaacsim",
    }
    _write_json(job_dir / "dataset_manifest.json", dataset_manifest)

    status = {
        "jobId": job_dir.name,
        "taskId": TASK_ID,
        "status": "completed",
        "progress": 100,
        "totalEpisodes": 1,
        "completedEpisodes": 1,
        "successEpisodes": 1 if success else 0,
        "failedEpisodes": 0 if success else 1,
        "outputDir": str(job_dir),
        "datasetId": dataset_manifest["datasetId"],
        "runtimeMode": "isaacsim",
        "videoAvailable": video_available,
        "video_status": video_status,
        "videoStatus": video_status,
        "message": "数据生成任务已完成",
    }
    _write_json(job_dir / "status.json", status)

    log_path = job_dir / "logs" / "run.log"
    if not log_path.is_file():
        log_path.write_text(
            f"[verify] organized adapter outputs at {_utc_now_iso()}\n",
            encoding="utf-8",
        )

    episode_manifest = _read_json(ep_dir / "episode_manifest.json")
    return {
        "job_dir": str(job_dir),
        "episode_id": episode_id,
        "video_path": str(video_path),
        "video_available": video_available,
        "video_status": video_status,
        "success": success,
        "pick_success": metrics.get("pick_success"),
        "place_success": metrics.get("place_success"),
        "controller_done": metrics.get("controller_done", run_result.get("controller_done")),
        "episode_task_id": episode_manifest.get("task_id"),
        "dataset_task_id": dataset_manifest.get("task_id"),
        "metrics": metrics,
        "run_result": run_result,
    }


def extract_preview(video_path: Path, preview_path: Path) -> bool:
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        return False
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        import subprocess

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(preview_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
            if result.returncode == 0 and preview_path.is_file() and preview_path.stat().st_size > 0:
                return True
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(video_path))
        frame = reader.get_data(0)
        imageio.imwrite(str(preview_path), frame)
        reader.close()
        return preview_path.is_file() and preview_path.stat().st_size > 0
    except Exception:
        return False


def write_acceptance_md(
    job_dir: Path,
    *,
    summary: dict[str, Any],
    preview_path: Path | None,
    checks: list[tuple[str, bool, str]],
    conclusion: str,
    skipped: bool = False,
    skip_reason: str | None = None,
    diagnosis: dict[str, Any] | None = None,
) -> Path:
    acceptance_path = job_dir / "ACCEPTANCE.md"
    job_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Isaac Sim Franka Pick and Place E2E Acceptance",
        "",
        f"- Generated at: {_utc_now_iso()}",
        "",
        "## Task",
        "",
        f"- task_id: `{TASK_ID}`",
        f"- task_name: {TASK_NAME}",
        f"- simulator: {SIMULATOR}",
        f"- robot: {ROBOT}",
        f"- expert_source: {EXPERT_SOURCE}",
        "",
    ]

    if skipped:
        lines.extend(
            [
                "## Runtime",
                "",
            ]
        )
        if diagnosis:
            lines.extend(
                [
                    f"- isaac_lab_runtime_available: `{diagnosis.get('isaac_lab_runtime_available')}`",
                    f"- isaac_sim_runtime_available: `{diagnosis.get('isaac_sim_runtime_available')}`",
                    f"- simulation_app_available: `{diagnosis.get('simulation_app_available')}`",
                    f"- can_import_franka_pick_place: `{diagnosis.get('can_import_franka_pick_place')}`",
                    f"- franka_pick_place_import_path: `{diagnosis.get('franka_pick_place_import_path')}`",
                    f"- detected_isaac_version: `{diagnosis.get('detected_isaac_version')}`",
                    f"- recommended_runner: `{((diagnosis.get('recommended_runner') or {}).get('label'))}`",
                    f"- diagnosis: `{diagnosis.get('diagnosis')}`",
                    "",
                ]
            )
        lines.extend(
            [
                f"- runtime_mode: `skipped`",
                f"- note: {skip_reason or 'E2E acceptance skipped.'}",
                "",
                "## Conclusion",
                "",
                conclusion,
                "",
            ]
        )
        acceptance_path.write_text("\n".join(lines), encoding="utf-8")
        return acceptance_path

    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    lines.extend(
        [
            "## Runtime",
            "",
        ]
    )
    if diagnosis:
        lines.extend(
            [
                f"- isaac_lab_runtime_available: `{diagnosis.get('isaac_lab_runtime_available')}`",
                f"- isaac_sim_runtime_available: `{diagnosis.get('isaac_sim_runtime_available')}`",
                f"- can_import_franka_pick_place: `{diagnosis.get('can_import_franka_pick_place')}`",
                f"- franka_pick_place_import_path: `{diagnosis.get('franka_pick_place_import_path')}`",
                f"- recommended_runner: `{((diagnosis.get('recommended_runner') or {}).get('label'))}`",
                "",
            ]
        )
    lines.extend(
        [
            "- runtime_mode: `isaacsim`",
            f"- video_status: `{summary.get('video_status')}`",
            "",
            "## Episode Metrics",
            "",
            f"- success: `{summary.get('success')}`",
            f"- pick_success: `{summary.get('pick_success')}`",
            f"- place_success: `{summary.get('place_success')}`",
            f"- controller_done: `{summary.get('controller_done')}`",
            "",
            "## Artifacts",
            "",
            f"- mp4: `{summary.get('video_path')}`",
            f"- preview: `{preview_path}`" if preview_path else "- preview: `(not generated)`",
            "",
            "## Task ID Validation",
            "",
            f"- episode_manifest.task_id: `{summary.get('episode_task_id')}`",
            f"- dataset_manifest.task_id: `{summary.get('dataset_task_id')}`",
            "",
            "## Checks",
            "",
        ]
    )
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        lines.append(f"- [{mark}] {name}: {detail}")
    lines.extend(["", "## Conclusion", "", conclusion, ""])
    acceptance_path.write_text("\n".join(lines), encoding="utf-8")
    return acceptance_path


def _load_diagnosis(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    diag_path = Path(path).expanduser()
    if not diag_path.is_file():
        return None
    try:
        data = json.loads(diag_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Isaac Sim Franka E2E acceptance artifacts")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--episode-id", default="ep_000001")
    parser.add_argument("--write-acceptance", action="store_true")
    parser.add_argument("--write-skipped", action="store_true")
    parser.add_argument("--skip-reason", default="E2E acceptance skipped.")
    parser.add_argument("--diagnose-json", default=None)
    parser.add_argument("--extract-preview", action="store_true")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).expanduser().resolve()
    diagnosis = _load_diagnosis(args.diagnose_json)

    if args.write_skipped:
        job_dir.mkdir(parents=True, exist_ok=True)
        write_acceptance_md(
            job_dir,
            summary={},
            preview_path=None,
            checks=[],
            conclusion=f"SKIPPED: {args.skip_reason}",
            skipped=True,
            skip_reason=args.skip_reason,
            diagnosis=diagnosis,
        )
        print(json.dumps({"accepted": False, "skipped": True, "acceptance": str(job_dir / "ACCEPTANCE.md")}))
        return 2

    summary = organize_job_dir(job_dir, episode_id=args.episode_id)
    preview_path = job_dir / "videos" / f"{args.episode_id}_preview.png"
    preview_ok = False
    if args.extract_preview:
        preview_ok = extract_preview(Path(summary["video_path"]), preview_path)

    if args.write_acceptance:
        video_path = Path(summary["video_path"])
        episode_manifest = _read_json(job_dir / "episodes" / args.episode_id / "episode_manifest.json")
        dataset_manifest = _read_json(job_dir / "dataset_manifest.json")
        checks = [
            ("status.json", (job_dir / "status.json").is_file(), str(job_dir / "status.json")),
            ("dataset_manifest.json", (job_dir / "dataset_manifest.json").is_file(), "present"),
            (
                "aggregate_metrics.json",
                (job_dir / "results" / "aggregate_metrics.json").is_file(),
                "present",
            ),
            (
                "episode_manifest.json",
                (job_dir / "episodes" / args.episode_id / "episode_manifest.json").is_file(),
                "present",
            ),
            (
                "metrics.json",
                (job_dir / "episodes" / args.episode_id / "metrics.json").is_file(),
                "present",
            ),
            (
                "trajectory.json",
                (job_dir / "episodes" / args.episode_id / "trajectory.json").is_file(),
                "present",
            ),
            ("videos/ep_000001.mp4", video_path.is_file() and video_path.stat().st_size > 0, str(video_path)),
            (
                "video_status=available",
                summary.get("video_status") == "available",
                str(summary.get("video_status")),
            ),
            (
                "task_id consistency",
                episode_manifest.get("task_id") == dataset_manifest.get("task_id") == TASK_ID,
                f"episode={episode_manifest.get('task_id')} dataset={dataset_manifest.get('task_id')}",
            ),
            (
                "forbidden video hints",
                not contains_forbidden_video_hint(video_path),
                str(video_path),
            ),
            (
                "preview.png",
                preview_ok,
                str(preview_path),
            ),
        ]
        all_pass = all(item[1] for item in checks)
        conclusion = (
            "ACCEPTED: real Isaac Sim Franka Pick-and-Place video E2E verification passed."
            if all_pass
            else "REJECTED: real Isaac Sim Franka Pick-and-Place video E2E verification failed."
        )
        write_acceptance_md(
            job_dir,
            summary=summary,
            preview_path=preview_path if preview_ok else None,
            checks=checks,
            conclusion=conclusion,
            diagnosis=diagnosis,
        )
        print(json.dumps({"accepted": all_pass, "summary": summary, "preview": str(preview_path)}, ensure_ascii=False))
        return 0 if all_pass else 1

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
