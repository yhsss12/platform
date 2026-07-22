from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation_replay_info import write_evaluation_video_metadata
from app.services.evaluation.base import utc_now_iso
from app.services import dual_arm_cable_service as dac_svc
from app.services.workspace_model_asset_service import get_model_asset_by_id

logger = logging.getLogger(__name__)

JOB_RESULT = dac_svc.JOB_RESULT
JOB_VIDEO = dac_svc.JOB_VIDEO
ALLOWED_STRETCH_MODES = dac_svc.ALLOWED_STRETCH_MODES
ALLOWED_RELEASE_MODES = dac_svc.ALLOWED_RELEASE_MODES

EpisodeRunner = Callable[..., dict[str, Any]]

_DUAL_ARM_TEMPLATE_IDS = frozenset(
    {"dual_arm_cable_manipulation", "task_dual_arm_cable_manipulation_v1"}
)

DEFAULT_DUAL_ARM_EPISODE_TIMEOUT_SECONDS = int(
    os.environ.get("DUAL_ARM_EPISODE_TIMEOUT_SECONDS", "1800")
)


def resolve_eval_params(request: EvaluateAsyncRequest) -> dict[str, Any]:
    params = request.dualArmCable or {}
    seeds = list(request.seeds or params.get("seeds") or [])
    if not seeds:
        if request.seed is None:
            raise ValueError("seeds required for episode_stability evaluation (or provide seed)")
        seeds = [int(request.seed) + i for i in range(request.numEpisodes)]
    stretch_mode = str(
        params.get("stretchMode") or params.get("stretch_mode") or "fixed_distance"
    )
    release_mode = str(
        params.get("releaseMode") or params.get("release_mode") or "three_phase"
    )
    return {
        "seeds": seeds,
        "max_cables": int(params.get("maxCables") if params.get("maxCables") is not None else request.maxCables or 1),
        "record": bool(params.get("record") if "record" in params else request.record),
        "headless": bool(params.get("headless") if "headless" in params else request.headless),
        "stretch_mode": stretch_mode,
        "release_mode": release_mode,
    }


def resolve_policy_eval_params(request: EvaluateAsyncRequest) -> dict[str, Any]:
    params = request.dualArmCable or {}
    seeds = list(request.seeds or params.get("seeds") or [])
    if not seeds:
        if request.seed is None:
            raise ValueError("seeds required for trained_model_evaluation (or provide seed)")
        seeds = [int(request.seed) + i for i in range(request.numEpisodes)]
    checkpoint_path = str(
        params.get("checkpointPath")
        or request.checkpointPath
        or request.checkpointId
        or ""
    ).strip()
    model_asset_id = str(params.get("modelAssetId") or request.modelAssetId or "").strip()
    if not checkpoint_path and model_asset_id:
        asset = get_model_asset_by_id(model_asset_id)
        if asset:
            checkpoint_path = str(asset.get("checkpointPath") or "").strip()
    return {
        "seeds": seeds,
        "checkpoint_path": checkpoint_path,
        "model_asset_id": model_asset_id,
        "max_steps": int(params.get("maxPolicySteps") or params.get("maxSteps") or 500),
        "record": bool(params.get("record") if "record" in params else True if request.record is None else request.record),
        "headless": bool(params.get("headless") if "headless" in params else request.headless),
    }


def build_policy_rollout_command(
    *,
    episode_dir: Path,
    checkpoint_path: str,
    seed: int,
    max_steps: int,
    record: bool,
    model_asset_id: str,
    python_bin: Path,
    eval_script: Path,
    scene_xml: Path,
) -> list[str]:
    cmd = [
        str(python_bin),
        str(eval_script),
        "--checkpoint",
        checkpoint_path,
        "--output-dir",
        str(episode_dir),
        "--scene",
        str(scene_xml),
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
        "--device",
        "cpu",
    ]
    if record:
        cmd.append("--record")
    if model_asset_id:
        cmd.extend(["--model-asset-id", model_asset_id])
    return cmd


def policy_episode_record_from_result(
    *,
    eval_job_id: str,
    episode_index: int,
    seed: int,
    episode_dir: Path,
    episode_status: str,
    episode_result: dict[str, Any],
    return_code: int,
    runtime_sec: float,
    failure_reason: Optional[str],
) -> dict[str, Any]:
    episode_success = bool(episode_result.get("episode_success"))
    steps_executed = int(episode_result.get("steps_executed") or 0)
    mean_reward = episode_result.get("mean_reward")
    video_rel = f"videos/episode_{episode_index:02d}.mp4"
    video_path = episode_dir.parent.parent / video_rel
    resolved_failure = failure_reason
    if episode_success is not True and not resolved_failure:
        resolved_failure = (
            episode_result.get("failure_reason")
            or episode_result.get("failure_summary")
            or "episode_not_successful"
        )
    return {
        "episodeIndex": episode_index,
        "seed": seed,
        "status": episode_status,
        "episodeStatus": episode_status,
        "episodeSuccess": episode_success,
        "taskSuccess": bool(episode_result.get("task_success")),
        "graspSuccess": bool(episode_result.get("grasp_success")),
        "stretchSuccess": bool(episode_result.get("stretch_success")),
        "leftContact": episode_result.get("left_contact"),
        "rightContact": episode_result.get("right_contact"),
        "stretchReached": episode_result.get("stretch_reached"),
        "stepsExecuted": steps_executed,
        "meanReward": mean_reward,
        "totalReward": episode_result.get("total_reward"),
        "backendType": episode_result.get("backend_type") or "torch_bc",
        "policyMode": episode_result.get("policyMode") or "torch_bc_policy",
        "videoPath": video_rel if video_path.is_file() else None,
        "resultPath": f"episodes/episode_{episode_index:02d}/results/episode_result.json",
        "failureReason": resolved_failure if episode_success is not True else None,
        "metrics": {
            "leftContact": episode_result.get("left_contact"),
            "rightContact": episode_result.get("right_contact"),
            "stretchReached": episode_result.get("stretch_reached"),
            "aborted": episode_result.get("aborted"),
            "finalLeftContact": episode_result.get("final_left_contact"),
            "finalRightContact": episode_result.get("final_right_contact"),
            "finalStretchReached": episode_result.get("final_stretch_reached"),
        },
        "runtimeSec": round(runtime_sec, 2),
        "returnCode": return_code,
        "evalJobId": eval_job_id,
    }


def aggregate_policy_episode_records(
    eval_job_id: str,
    episodes: list[dict[str, Any]],
    *,
    model_asset_id: str,
    backend_type: str = "torch_bc",
) -> dict[str, Any]:
    total = len(episodes)
    success_episodes = sum(1 for ep in episodes if ep.get("episodeSuccess") is True)
    success_rate = (success_episodes / total) if total else 0.0
    failure_count = total - success_episodes
    rewards = [
        float(ep["meanReward"])
        for ep in episodes
        if isinstance(ep.get("meanReward"), (int, float))
    ]
    lengths = [
        int(ep["stepsExecuted"])
        for ep in episodes
        if isinstance(ep.get("stepsExecuted"), int)
    ]
    runtimes = [
        float(ep["runtimeSec"])
        for ep in episodes
        if isinstance(ep.get("runtimeSec"), (int, float))
    ]
    videos = [ep["videoPath"] for ep in episodes if ep.get("videoPath")]

    return {
        "evalJobId": eval_job_id,
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "trained_model_evaluation",
        "modelAssetId": model_asset_id,
        "backendType": backend_type,
        "episodeCount": total,
        "successRate": round(success_rate, 4),
        "meanReward": round(sum(rewards) / len(rewards), 6) if rewards else None,
        "meanEpisodeLength": round(sum(lengths) / len(lengths), 2) if lengths else None,
        "failureCount": failure_count,
        "completedAt": utc_now_iso(),
        "summary": {
            "totalEpisodes": total,
            "successEpisodes": success_episodes,
            "successRate": round(success_rate, 4),
        },
        "perEpisode": episodes,
        "artifacts": {
            "perEpisodeResults": "results/per_episode_results.json",
            "aggregateResult": "results/aggregate_result.json",
            "log": "logs/eval.log",
            "videos": videos,
        },
    }


def build_episode_command(
    *,
    episode_job_id: str,
    episode_dir: Path,
    max_cables: int,
    seed: int,
    record: bool,
    headless: bool,
    stretch_mode: str,
    release_mode: str,
    python_bin: Path,
    platform_runner: Path,
    scene_xml: Path,
) -> list[str]:
    cmd = [
        str(python_bin),
        str(platform_runner),
        "--job-id",
        episode_job_id,
        "--output-dir",
        str(episode_dir),
        "--scene",
        str(scene_xml),
        "--max-cables",
        str(max_cables),
        "--seed",
        str(seed),
        "--stretch-mode",
        stretch_mode,
        "--release-mode",
        release_mode,
    ]
    if record:
        cmd.append("--record")
    if headless:
        cmd.append("--headless")
    return cmd


def parse_episode_result(episode_dir: Path) -> dict[str, Any]:
    candidates = (
        episode_dir / JOB_RESULT,
        episode_dir / "episode" / "episode_result.json",
    )
    for result_path in candidates:
        if not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload:
                return payload
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def _episode_run_record_path(episode_dir: Path) -> Path:
    return episode_dir / "run_status.json"


def write_episode_run_record(
    episode_dir: Path,
    *,
    eval_job_id: str,
    episode_index: int,
    seed: int,
    status: str,
    pid: Optional[int] = None,
    return_code: Optional[int] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "evalJobId": eval_job_id,
        "episodeIndex": episode_index,
        "seed": seed,
        "status": status,
        "updatedAt": utc_now_iso(),
    }
    if pid is not None:
        payload["pid"] = pid
    if return_code is not None:
        payload["returnCode"] = return_code
    if started_at:
        payload["startedAt"] = started_at
    if finished_at:
        payload["finishedAt"] = finished_at
    if error:
        payload["error"] = error
    _episode_run_record_path(episode_dir).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _persist_interim_results(
    job_root: Path,
    eval_job_id: str,
    *,
    evaluation_mode: str,
    per_episode: list[dict[str, Any]],
    total_episodes: int,
    started_at: str,
    phase: str = "episode_running",
) -> None:
    results_dir = job_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_episode_payload = {
        "evalJobId": eval_job_id,
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": evaluation_mode,
        "episodes": per_episode,
        "completedEpisodes": len(per_episode),
        "requestedEpisodes": total_episodes,
        "updatedAt": utc_now_iso(),
    }
    (results_dir / "per_episode_results.json").write_text(
        json.dumps(per_episode_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    completed = len(per_episode)
    progress = (completed / total_episodes) if total_episodes else 0.0
    _write_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": evaluation_mode,
            "status": "running",
            "phase": phase,
            "progress": min(0.99, progress),
            "currentEpisode": completed,
            "totalEpisodes": total_episodes,
            "completedEpisodes": completed,
            "requestedEpisodes": total_episodes,
            "message": f"已完成 {completed}/{total_episodes} 个 episode",
            "metrics": {"completedEpisodes": completed, "requestedEpisodes": total_episodes},
            "artifacts": {},
            "startedAt": started_at,
        },
    )


def _finalize_episode_stability_job(
    job_root: Path,
    eval_job_id: str,
    request: EvaluateAsyncRequest,
    *,
    per_episode: list[dict[str, Any]],
    total: int,
    started_at: str,
    aborted: bool = False,
    abort_message: Optional[str] = None,
) -> None:
    results_dir = job_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_episode_payload = {
        "evalJobId": eval_job_id,
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "episode_stability",
        "episodes": per_episode,
    }
    (results_dir / "per_episode_results.json").write_text(
        json.dumps(per_episode_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not per_episode:
        _fail_job(job_root, eval_job_id, request, abort_message or "评测未产生任何 episode 结果")
        return

    aggregate = aggregate_episode_records(eval_job_id, per_episode)
    (results_dir / "aggregate_result.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = aggregate["summary"]
    task_metrics = aggregate["taskMetrics"]
    completed = len(per_episode)
    if aborted or completed < total:
        message = abort_message or f"评测在第 {completed + 1} 轮前异常退出（已完成 {completed}/{total}）"
        _write_status(
            job_root,
            {
                "evalJobId": eval_job_id,
                "taskType": "dual_arm_cable_manipulation",
                "evaluationMode": "episode_stability",
                "status": "failed",
                "phase": "failed",
                "progress": completed / total if total else 0.0,
                "currentEpisode": completed,
                "totalEpisodes": total,
                "completedEpisodes": completed,
                "requestedEpisodes": total,
                "message": message,
                "error": message,
                "metrics": {
                    "successRate": summary.get("successRate"),
                    "successEpisodes": summary.get("successEpisodes"),
                    "totalEpisodes": summary.get("totalEpisodes"),
                    "completedEpisodes": completed,
                    "requestedEpisodes": total,
                    "meanFinalSag": task_metrics.get("meanFinalSag"),
                    "meanFinalSpan": task_metrics.get("meanFinalSpan"),
                },
                "artifacts": aggregate.get("artifacts") or {},
                "startedAt": started_at,
                "failedAt": utc_now_iso(),
            },
        )
        _append_eval_log(job_root, f"[eval_worker] FAILED partial evalJobId={eval_job_id} completed={completed}/{total}")
        return

    _write_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": "episode_stability",
            "status": "completed",
            "phase": "completed",
            "progress": 1.0,
            "currentEpisode": total,
            "totalEpisodes": total,
            "completedEpisodes": total,
            "requestedEpisodes": total,
            "message": "episode 稳定性评测完成",
            "error": None,
            "metrics": {
                "successRate": summary.get("successRate"),
                "successEpisodes": summary.get("successEpisodes"),
                "totalEpisodes": summary.get("totalEpisodes"),
                "meanFinalSag": task_metrics.get("meanFinalSag"),
                "meanFinalSpan": task_metrics.get("meanFinalSpan"),
                "contactSuccessRate": task_metrics.get("contactSuccessRate"),
                "stretchReachedRate": task_metrics.get("stretchReachedRate"),
            },
            "artifacts": aggregate.get("artifacts") or {},
            "startedAt": started_at,
        },
    )
    _append_eval_log(job_root, f"[eval_worker] completed evalJobId={eval_job_id}")
    write_evaluation_video_metadata(
        job_root,
        evaluation_mode="episode_stability",
        source_file_name=str(JOB_VIDEO.name),
    )


def episode_record_from_result(
    *,
    eval_job_id: str,
    episode_index: int,
    seed: int,
    episode_dir: Path,
    episode_status: str,
    episode_result: dict[str, Any],
    return_code: int,
    runtime_sec: float,
    failure_reason: Optional[str],
) -> dict[str, Any]:
    step_metrics = dac_svc._extract_step_metrics(episode_result)
    episode_success = bool(episode_result.get("episode_success"))
    succeeded = int(episode_result.get("num_cables_succeeded") or 0)
    max_cables = int(episode_result.get("max_cables") or 1)
    video_rel = f"videos/episode_{episode_index:02d}.mp4"
    video_path = episode_dir.parent.parent / video_rel
    return {
        "episodeIndex": episode_index,
        "seed": seed,
        "status": episode_status,
        "episodeSuccess": episode_success,
        "succeededCables": succeeded,
        "maxCables": max_cables,
        "leftContact": step_metrics.get("left_contact"),
        "rightContact": step_metrics.get("right_contact"),
        "stretchReached": step_metrics.get("stretch_reached"),
        "sagM": step_metrics.get("sag_m"),
        "spanM": step_metrics.get("span_m"),
        "finalSagM": step_metrics.get("final_sag_m"),
        "finalSpanM": step_metrics.get("final_span_m"),
        "videoPath": video_rel if video_path.is_file() else None,
        "resultPath": f"episodes/episode_{episode_index:02d}/results/episode_result.json",
        "failureReason": failure_reason,
        "runtimeSec": round(runtime_sec, 2),
        "returnCode": return_code,
        "evalJobId": eval_job_id,
    }


def aggregate_episode_records(
    eval_job_id: str,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(episodes)
    success_episodes = sum(1 for ep in episodes if ep.get("episodeSuccess") is True)
    success_rate = (success_episodes / total) if total else 0.0

    contact_hits = 0
    stretch_hits = 0
    final_sags: list[float] = []
    final_spans: list[float] = []
    sags: list[float] = []
    spans: list[float] = []
    runtimes: list[float] = []
    failure_seeds: list[int] = []
    failure_reasons: dict[str, int] = {}

    for ep in episodes:
        seed = ep.get("seed")
        if ep.get("episodeSuccess") is not True:
            if seed is not None:
                failure_seeds.append(int(seed))
            reason = str(ep.get("failureReason") or "episode_not_successful")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        if ep.get("leftContact") is True and ep.get("rightContact") is True:
            contact_hits += 1
        if ep.get("stretchReached") is True:
            stretch_hits += 1
        for key, bucket in (
            ("finalSagM", final_sags),
            ("finalSpanM", final_spans),
            ("sagM", sags),
            ("spanM", spans),
        ):
            val = ep.get(key)
            if isinstance(val, (int, float)):
                bucket.append(float(val))
        rt = ep.get("runtimeSec")
        if isinstance(rt, (int, float)):
            runtimes.append(float(rt))

    def _mean(values: list[float]) -> Optional[float]:
        return round(sum(values) / len(values), 6) if values else None

    videos = [ep["videoPath"] for ep in episodes if ep.get("videoPath")]

    return {
        "evalJobId": eval_job_id,
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "episode_stability",
        "completedAt": utc_now_iso(),
        "summary": {
            "totalEpisodes": total,
            "successEpisodes": success_episodes,
            "successRate": round(success_rate, 4),
        },
        "taskMetrics": {
            "contactSuccessRate": round(contact_hits / total, 4) if total else 0.0,
            "stretchReachedRate": round(stretch_hits / total, 4) if total else 0.0,
            "meanFinalSag": _mean(final_sags),
            "meanFinalSpan": _mean(final_spans),
            "meanSag": _mean(sags),
            "meanSpan": _mean(spans),
            "failureSeeds": failure_seeds,
            "failureReasons": failure_reasons,
            "meanRuntimeSec": round(sum(runtimes) / len(runtimes), 2) if runtimes else None,
        },
        "perEpisode": episodes,
        "artifacts": {
            "perEpisodeResults": "results/per_episode_results.json",
            "aggregateResult": "results/aggregate_result.json",
            "log": "logs/eval.log",
            "videos": videos,
        },
    }


def _append_eval_log(job_root: Path, message: str) -> None:
    log_path = job_root / "logs" / "eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(message)
        if not message.endswith("\n"):
            fh.write("\n")


def _write_status(job_root: Path, payload: dict[str, Any]) -> None:
    payload["updatedAt"] = utc_now_iso()
    (job_root / "status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_source_jobs(job_root: Path, episodes_meta: list[dict[str, Any]]) -> None:
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dual_arm_cable_manipulation": {"episodes": episodes_meta}}
    (meta_dir / "source_jobs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_episode_subprocess(
    *,
    cmd: list[str],
    episode_dir: Path,
    integration_dir: Path,
    env: dict[str, str],
    eval_job_id: str,
    episode_index: int,
    seed: int,
    timeout_seconds: Optional[int] = None,
) -> tuple[int, str]:
    episode_dir.mkdir(parents=True, exist_ok=True)
    log_path = episode_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    started = time.time()
    write_episode_run_record(
        episode_dir,
        eval_job_id=eval_job_id,
        episode_index=episode_index,
        seed=seed,
        status="running",
        started_at=started_at,
    )
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[eval_worker] command={' '.join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(integration_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        write_episode_run_record(
            episode_dir,
            eval_job_id=eval_job_id,
            episode_index=episode_index,
            seed=seed,
            status="running",
            pid=proc.pid,
            started_at=started_at,
        )
        try:
            proc.wait(timeout=timeout_seconds)
            return_code = int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            return_code = -9
            finished_at = utc_now_iso()
            write_episode_run_record(
                episode_dir,
                eval_job_id=eval_job_id,
                episode_index=episode_index,
                seed=seed,
                status="failed",
                pid=proc.pid,
                return_code=return_code,
                started_at=started_at,
                finished_at=finished_at,
                error=f"episode timeout after {timeout_seconds}s",
            )
            return return_code, f"episode timeout after {timeout_seconds}s"

    finished_at = utc_now_iso()
    failure_reason = None
    if return_code != 0:
        failure_reason = f"rollout subprocess exited with code {return_code}"
    elif not parse_episode_result(episode_dir):
        failure_reason = "episode_result.json not found"
    write_episode_run_record(
        episode_dir,
        eval_job_id=eval_job_id,
        episode_index=episode_index,
        seed=seed,
        status="failed" if failure_reason else "completed",
        pid=proc.pid,
        return_code=return_code,
        started_at=started_at,
        finished_at=finished_at,
        error=failure_reason,
    )
    return return_code, failure_reason or ""


def run_evaluation_worker(
    eval_job_id: str,
    job_root: Path,
    request: EvaluateAsyncRequest,
    *,
    episode_runner: Optional[EpisodeRunner] = None,
) -> None:
    """Sequential multi-seed evaluation worker."""
    evaluation_mode = str(request.evaluationMode or "episode_stability")
    if evaluation_mode == "trained_model_evaluation":
        _run_policy_evaluation_worker(
            eval_job_id,
            job_root,
            request,
            episode_runner=episode_runner,
        )
        return
    _run_episode_stability_worker(
        eval_job_id,
        job_root,
        request,
        episode_runner=episode_runner,
    )


def _run_policy_evaluation_worker(
    eval_job_id: str,
    job_root: Path,
    request: EvaluateAsyncRequest,
    *,
    episode_runner: Optional[EpisodeRunner] = None,
) -> None:
    params = resolve_policy_eval_params(request)
    seeds: list[int] = [int(s) for s in params["seeds"]]
    total = len(seeds)
    checkpoint_path = params["checkpoint_path"]
    model_asset_id = params["model_asset_id"]

    if not dac_svc.PYTHON_BIN.is_file():
        _fail_job(job_root, eval_job_id, request, f"Python interpreter not found: {dac_svc.PYTHON_BIN}")
        return
    if not dac_svc.EVAL_POLICY_ROLLOUT.is_file():
        _fail_job(
            job_root,
            eval_job_id,
            request,
            f"eval_policy_rollout not found: {dac_svc.EVAL_POLICY_ROLLOUT}",
        )
        return
    if not dac_svc.SCENE_XML.is_file():
        _fail_job(job_root, eval_job_id, request, f"scene not found: {dac_svc.SCENE_XML}")
        return
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        _fail_job(
            job_root,
            eval_job_id,
            request,
            f"checkpoint not found for trained_model_evaluation: {checkpoint_path or '(empty)'}",
        )
        return

    episodes_meta: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []

    _write_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": "trained_model_evaluation",
            "status": "running",
            "phase": "policy_rollout",
            "progress": 0.0,
            "currentEpisode": 0,
            "totalEpisodes": total,
            "message": "torch_bc 策略 rollout 评测 worker 已启动",
            "metrics": {},
            "artifacts": {"checkpointPath": checkpoint_path, "modelAssetId": model_asset_id},
            "startedAt": utc_now_iso(),
        },
    )

    env = dac_svc._build_child_env()

    for idx, seed in enumerate(seeds):
        episode_index = idx
        episode_key = f"episode_{episode_index:02d}"
        episode_dir = job_root / "episodes" / episode_key
        episodes_meta.append(
            {
                "episodeIndex": episode_index,
                "episodeDir": str(episode_dir.relative_to(job_root)),
                "seed": seed,
                "checkpointPath": checkpoint_path,
            }
        )
        _write_source_jobs(job_root, episodes_meta)

        progress = (idx / total) * 0.9 if total else 0.0
        _write_status(
            job_root,
            {
                "evalJobId": eval_job_id,
                "taskType": "dual_arm_cable_manipulation",
                "evaluationMode": "trained_model_evaluation",
                "status": "running",
                "phase": "policy_rollout",
                "progress": progress,
                "currentEpisode": idx + 1,
                "totalEpisodes": total,
                "message": f"正在运行 torch_bc rollout 第 {idx + 1}/{total} 个 episode，seed={seed}",
                "metrics": {},
                "artifacts": {
                    "currentEpisodeDir": str(episode_dir.relative_to(job_root)),
                    "checkpointPath": checkpoint_path,
                },
                "startedAt": _read_started_at(job_root),
            },
        )
        _append_eval_log(
            job_root,
            f"[eval_worker] policy_rollout episode={episode_index} seed={seed} checkpoint={checkpoint_path}",
        )

        for sub in ("logs", "live", "videos", "results"):
            (episode_dir / sub).mkdir(parents=True, exist_ok=True)

        cmd = build_policy_rollout_command(
            episode_dir=episode_dir,
            checkpoint_path=checkpoint_path,
            seed=seed,
            max_steps=params["max_steps"],
            record=params["record"],
            model_asset_id=model_asset_id,
            python_bin=dac_svc.PYTHON_BIN,
            eval_script=dac_svc.EVAL_POLICY_ROLLOUT,
            scene_xml=dac_svc.SCENE_XML,
        )

        started = time.time()
        if episode_runner is not None:
            runner_result = episode_runner(
                eval_job_id=eval_job_id,
                episode_index=episode_index,
                seed=seed,
                episode_dir=episode_dir,
                cmd=cmd,
            )
            return_code = int(runner_result.get("returnCode", 0))
            failure_reason = runner_result.get("failureReason")
            episode_result = runner_result.get("episodeResult") or parse_episode_result(episode_dir)
            runtime_sec = float(runner_result.get("runtimeSec", time.time() - started))
        else:
            return_code, failure_reason = run_episode_subprocess(
                cmd=cmd,
                episode_dir=episode_dir,
                integration_dir=dac_svc.INTEGRATION_DIR,
                env=env,
                eval_job_id=eval_job_id,
                episode_index=episode_index,
                seed=seed,
                timeout_seconds=DEFAULT_DUAL_ARM_EPISODE_TIMEOUT_SECONDS,
            )
            runtime_sec = time.time() - started
            episode_result = parse_episode_result(episode_dir)

        src_video = episode_dir / JOB_VIDEO
        dst_video = job_root / "videos" / f"episode_{episode_index:02d}.mp4"
        if src_video.is_file():
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)

        if return_code != 0 or not episode_result:
            episode_status = "failed"
            if not failure_reason:
                failure_reason = episode_result.get("failure_reason") or "policy rollout failed"
        else:
            episode_status = "completed"
            if episode_result.get("episode_success") is not True and not failure_reason:
                failure_reason = episode_result.get("failure_reason") or "episode_not_successful"

        record = policy_episode_record_from_result(
            eval_job_id=eval_job_id,
            episode_index=episode_index,
            seed=seed,
            episode_dir=episode_dir,
            episode_status=episode_status,
            episode_result=episode_result,
            return_code=return_code,
            runtime_sec=runtime_sec,
            failure_reason=failure_reason,
        )
        per_episode.append(record)
        _append_eval_log(
            job_root,
            f"[eval_worker] policy episode={episode_index} status={episode_status} success={record.get('episodeSuccess')}",
        )

    per_episode_payload = {
        "evalJobId": eval_job_id,
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "trained_model_evaluation",
        "modelAssetId": model_asset_id,
        "backendType": "torch_bc",
        "episodes": per_episode,
    }
    results_dir = job_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "per_episode_results.json").write_text(
        json.dumps(per_episode_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    aggregate = aggregate_policy_episode_records(
        eval_job_id,
        per_episode,
        model_asset_id=model_asset_id,
        backend_type="torch_bc",
    )
    (results_dir / "aggregate_result.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = aggregate["summary"]
    _write_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": "trained_model_evaluation",
            "status": "completed",
            "phase": "completed",
            "progress": 1.0,
            "currentEpisode": total,
            "totalEpisodes": total,
            "message": "torch_bc 训练模型 rollout 评测完成",
            "metrics": {
                "successRate": aggregate.get("successRate"),
                "meanReward": aggregate.get("meanReward"),
                "meanEpisodeLength": aggregate.get("meanEpisodeLength"),
                "failureCount": aggregate.get("failureCount"),
                "successEpisodes": summary.get("successEpisodes"),
                "totalEpisodes": summary.get("totalEpisodes"),
            },
            "artifacts": aggregate.get("artifacts") or {},
            "startedAt": _read_started_at(job_root),
        },
    )
    _append_eval_log(job_root, f"[eval_worker] completed policy eval evalJobId={eval_job_id}")
    write_evaluation_video_metadata(
        job_root,
        evaluation_mode="trained_model_evaluation",
        source_file_name=str(JOB_VIDEO.name),
    )


def _run_episode_stability_worker(
    eval_job_id: str,
    job_root: Path,
    request: EvaluateAsyncRequest,
    *,
    episode_runner: Optional[EpisodeRunner] = None,
) -> None:
    """Sequential multi-seed episode stability evaluation worker."""
    params = resolve_eval_params(request)
    seeds: list[int] = [int(s) for s in params["seeds"]]
    total = len(seeds)

    if not dac_svc.PYTHON_BIN.is_file():
        _fail_job(job_root, eval_job_id, request, f"Python interpreter not found: {dac_svc.PYTHON_BIN}")
        return
    if not dac_svc.PLATFORM_RUNNER.is_file():
        _fail_job(job_root, eval_job_id, request, f"platform_runner not found: {dac_svc.PLATFORM_RUNNER}")
        return
    if not dac_svc.SCENE_XML.is_file():
        _fail_job(job_root, eval_job_id, request, f"scene not found: {dac_svc.SCENE_XML}")
        return
    if params["stretch_mode"] not in ALLOWED_STRETCH_MODES:
        _fail_job(job_root, eval_job_id, request, f"invalid stretchMode: {params['stretch_mode']}")
        return
    if params["release_mode"] not in ALLOWED_RELEASE_MODES:
        _fail_job(job_root, eval_job_id, request, f"invalid releaseMode: {params['release_mode']}")
        return

    episodes_meta: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []

    _write_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": "episode_stability",
            "status": "running",
            "phase": "episode_running",
            "progress": 0.0,
            "currentEpisode": 0,
            "totalEpisodes": total,
            "message": "评测 worker 已启动",
            "metrics": {},
            "artifacts": {},
            "startedAt": utc_now_iso(),
        },
    )

    env = dac_svc._build_child_env()
    started_at = utc_now_iso()
    episode_timeout = DEFAULT_DUAL_ARM_EPISODE_TIMEOUT_SECONDS

    try:
        for idx, seed in enumerate(seeds):
            episode_index = idx
            episode_key = f"episode_{episode_index:02d}"
            episode_dir = job_root / "episodes" / episode_key
            episode_job_id = f"{eval_job_id}_{episode_key}"
            episodes_meta.append(
                {
                    "episodeIndex": episode_index,
                    "episodeJobId": episode_job_id,
                    "episodeDir": str(episode_dir.relative_to(job_root)),
                    "seed": seed,
                }
            )
            _write_source_jobs(job_root, episodes_meta)

            progress = (idx / total) * 0.9 if total else 0.0
            _write_status(
                job_root,
                {
                    "evalJobId": eval_job_id,
                    "taskType": "dual_arm_cable_manipulation",
                    "evaluationMode": "episode_stability",
                    "status": "running",
                    "phase": "episode_running",
                    "progress": progress,
                    "currentEpisode": idx + 1,
                    "totalEpisodes": total,
                    "message": f"正在运行第 {idx + 1}/{total} 个 episode，seed={seed}",
                    "metrics": {},
                    "artifacts": {"currentEpisodeDir": str(episode_dir.relative_to(job_root))},
                    "startedAt": started_at,
                },
            )
            _append_eval_log(
                job_root,
                f"[eval_worker] episode={episode_index} seed={seed} dir={episode_dir}",
            )

            for sub in ("logs", "live", "videos", "results"):
                (episode_dir / sub).mkdir(parents=True, exist_ok=True)

            cmd = build_episode_command(
                episode_job_id=episode_job_id,
                episode_dir=episode_dir,
                max_cables=params["max_cables"],
                seed=seed,
                record=params["record"],
                headless=params["headless"],
                stretch_mode=params["stretch_mode"],
                release_mode=params["release_mode"],
                python_bin=dac_svc.PYTHON_BIN,
                platform_runner=dac_svc.PLATFORM_RUNNER,
                scene_xml=dac_svc.SCENE_XML,
            )

            started = time.time()
            if episode_runner is not None:
                runner_result = episode_runner(
                    eval_job_id=eval_job_id,
                    episode_index=episode_index,
                    seed=seed,
                    episode_dir=episode_dir,
                    cmd=cmd,
                )
                return_code = int(runner_result.get("returnCode", 0))
                failure_reason = runner_result.get("failureReason")
                episode_result = runner_result.get("episodeResult") or parse_episode_result(episode_dir)
                runtime_sec = float(runner_result.get("runtimeSec", time.time() - started))
            else:
                return_code, failure_reason = run_episode_subprocess(
                    cmd=cmd,
                    episode_dir=episode_dir,
                    integration_dir=dac_svc.INTEGRATION_DIR,
                    env=env,
                    eval_job_id=eval_job_id,
                    episode_index=episode_index,
                    seed=seed,
                    timeout_seconds=episode_timeout,
                )
                runtime_sec = time.time() - started
                episode_result = parse_episode_result(episode_dir)

            src_video = episode_dir / JOB_VIDEO
            dst_video = job_root / "videos" / f"episode_{episode_index:02d}.mp4"
            if src_video.is_file():
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)

            if return_code != 0 or not episode_result:
                episode_status = "failed"
                if not failure_reason:
                    failure_reason = "episode failed"
            else:
                episode_status = "completed"
                if episode_result.get("episode_success") is not True and not failure_reason:
                    failure_reason = "episode_not_successful"

            record = episode_record_from_result(
                eval_job_id=eval_job_id,
                episode_index=episode_index,
                seed=seed,
                episode_dir=episode_dir,
                episode_status=episode_status,
                episode_result=episode_result,
                return_code=return_code,
                runtime_sec=runtime_sec,
                failure_reason=failure_reason if episode_status == "failed" or not episode_result.get("episode_success") else None,
            )
            per_episode.append(record)
            _persist_interim_results(
                job_root,
                eval_job_id,
                evaluation_mode="episode_stability",
                per_episode=per_episode,
                total_episodes=total,
                started_at=started_at,
            )
            _append_eval_log(
                job_root,
                f"[eval_worker] episode={episode_index} status={episode_status} success={record.get('episodeSuccess')} returnCode={return_code}",
            )
    except Exception as exc:
        logger.exception("dual_arm eval worker aborted evalJobId=%s", eval_job_id)
        _append_eval_log(job_root, f"[eval_worker] ABORTED: {exc}")
        _finalize_episode_stability_job(
            job_root,
            eval_job_id,
            request,
            per_episode=per_episode,
            total=total,
            started_at=started_at,
            aborted=True,
            abort_message=f"评测 worker 异常退出: {exc}",
        )
        return

    _finalize_episode_stability_job(
        job_root,
        eval_job_id,
        request,
        per_episode=per_episode,
        total=total,
        started_at=started_at,
        aborted=False,
    )


def _read_started_at(job_root: Path) -> str:
    status_path = job_root / "status.json"
    if status_path.is_file():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if raw.get("startedAt"):
                return str(raw["startedAt"])
        except (OSError, json.JSONDecodeError):
            pass
    return utc_now_iso()


def _fail_job(
    job_root: Path,
    eval_job_id: str,
    request: EvaluateAsyncRequest,
    message: str,
) -> None:
    _append_eval_log(job_root, f"[eval_worker] FAILED: {message}")
    _write_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": request.evaluationMode,
            "status": "failed",
            "phase": "failed",
            "progress": 0.0,
            "currentEpisode": 0,
            "totalEpisodes": request.numEpisodes,
            "message": message,
            "metrics": {},
            "artifacts": {},
            "startedAt": utc_now_iso(),
        },
    )


def spawn_evaluation_worker(
    eval_job_id: str,
    job_root: Path,
    request: EvaluateAsyncRequest,
) -> None:
    thread = threading.Thread(
        target=run_evaluation_worker,
        args=(eval_job_id, job_root, request),
        daemon=False,
        name=f"dual-arm-eval-{eval_job_id}",
    )
    thread.start()
