"""Isaac Lab Stack Cube 训练模型 rollout 评测（subprocess + isaaclab.sh）。"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.evaluation_request_resolver import NormalizedEvaluateRequest
from app.services.evaluation.job_paths import make_isaac_eval_job_id, prepare_eval_job_root
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner
from app.services.isaac_lab.paths import (
    policy_eval_isaaclab_relative_path,
    policy_eval_platform_script,
    PROJECT_ROOT,
)
from app.services.isaac_lab.training_service import (
    ISAAC_STACK_ACTION_DIM,
    ISAAC_STACK_TASK_ENV,
    ISAAC_STACK_TASK_TYPE,
    ISAAC_STACK_TEMPLATE_ID,
)
from app.services.isaac_lab.video_compat import ensure_browser_playable_mp4
from app.services.workspace_job_service import record_workspace_job_start, sync_workspace_job_from_runtime
from app.services.evaluation.display_name import (
    build_evaluation_display_name,
    resolve_evaluation_task_name,
    resolve_evaluation_type_label,
    resolve_task_display_name,
)
from app.services.evaluation_metric_service import attach_isaac_eval_metric_metadata
from app.services.workspace_model_asset_service import get_model_asset_by_id

logger = logging.getLogger(__name__)

POLICY_EVAL_SCRIPT = policy_eval_isaaclab_relative_path()
ISAAC_EVAL_BACKEND = "isaac_robomimic_bc"
DEFAULT_TASK_ENV = ISAAC_STACK_TASK_ENV
DEFAULT_HORIZON = 400

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_JOBS: set[str] = set()


def _resolve_isaac_eval_model_name(request: EvaluateAsyncRequest) -> Optional[str]:
    for block in (request.cableThreading, request.dualArmCable):
        if isinstance(block, dict):
            for key in ("modelName", "taskName"):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for value in (request.modelName, request.taskName):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_model_manifest(asset: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(asset.get("manifestPath") or ""))
    if manifest_path.is_file():
        return _read_json(manifest_path)
    return {}


def _backend_type_incompatible_detail(actual: str, *, expected: list[str], task_label: str) -> dict[str, Any]:
    actual_display = actual.strip() or "未知"
    return {
        "code": "MODEL_ASSET_BACKEND_TYPE_INCOMPATIBLE",
        "message": (
            f"{task_label}评测当前仅支持 Isaac Robomimic BC 模型，"
            f"当前模型类型为 {actual_display}。"
        ),
        "expectedBackendTypes": expected,
        "actualBackendType": actual_display,
    }


def validate_isaac_robomimic_model_asset(
    model_asset_id: str,
    checkpoint_path: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    asset = get_model_asset_by_id(model_asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAssetId not found: {model_asset_id}",
        )

    manifest = _load_model_manifest(asset)
    framework = str(manifest.get("framework") or asset.get("framework") or "").strip()
    backend_type = str(manifest.get("backendType") or manifest.get("trainingBackend") or "").strip()
    if backend_type != ISAAC_EVAL_BACKEND and framework not in {
        "Isaac Robomimic BC",
        "Robomimic BC",
        ISAAC_EVAL_BACKEND,
    }:
        actual = backend_type or framework or "未知"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_backend_type_incompatible_detail(
                actual,
                expected=[ISAAC_EVAL_BACKEND],
                task_label="物块堆叠",
            ),
        )

    task_template_id = str(manifest.get("taskTemplateId") or asset.get("taskTemplateId") or "").strip()
    if task_template_id and task_template_id not in {ISAAC_STACK_TEMPLATE_ID, "task_isaac_block_stacking_v1"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset taskTemplateId must be isaac_block_stacking, got {task_template_id!r}",
        )

    task_type = str(manifest.get("taskType") or "").strip()
    if task_type and task_type not in {ISAAC_STACK_TASK_TYPE, "block_stacking"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset taskType must be isaac_block_stacking, got {task_type!r}",
        )

    action_dim = manifest.get("actionDim")
    if action_dim is not None and int(action_dim) != ISAAC_STACK_ACTION_DIM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset actionDim must be {ISAAC_STACK_ACTION_DIM}, got {action_dim}",
        )

    task_env = str(manifest.get("taskEnv") or DEFAULT_TASK_ENV).strip()
    if task_env != DEFAULT_TASK_ENV:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset taskEnv must be {DEFAULT_TASK_ENV}, got {task_env!r}",
        )

    ckpt_raw = checkpoint_path or str(manifest.get("checkpointPath") or asset.get("checkpointPath") or "")
    ckpt = Path(ckpt_raw).expanduser()
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"checkpointPath not found or empty: {ckpt_raw}",
        )

    if framework in {"robomimic_bc", "torch_bc"} or backend_type in {"robomimic_bc", "torch_bc"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MuJoCo 线缆模型不可用于 Isaac 物块堆叠评测",
        )

    return asset, manifest, ckpt.resolve()


def probe_isaac_eval_capability() -> dict[str, Any]:
    runner = IsaacLabCliRunner.from_settings()
    script_src, script_ok = policy_eval_platform_script()
    issues: list[str] = []
    evidence: list[str] = []

    if runner.root is not None:
        evidence.append(str(runner.root))
    if script_ok:
        evidence.append(str(script_src))
    else:
        issues.append("stack_cube_policy_eval.py missing")

    if not runner.is_ready():
        issues.append("isaaclab.sh is not configured")
    if not getattr(settings, "ISAACLAB_RUNTIME_ENABLED", False):
        issues.append("ISAACLAB_RUNTIME_ENABLED is false")

    return {"ready": runner.is_ready() and script_ok and not issues, "evidence": evidence, "issues": issues}


def _sync_platform_script(runner: IsaacLabCliRunner) -> Path:
    source, ok = policy_eval_platform_script()
    if not ok or runner.root is None:
        raise FileNotFoundError("Platform Isaac policy eval script is missing")
    dest = runner.root / POLICY_EVAL_SCRIPT
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    preview_src = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/preview_video_utils.py"
    if preview_src.is_file():
        preview_dest = dest.parent / "preview_video_utils.py"
        shutil.copy2(preview_src, preview_dest)
    return dest


def _build_cli_args(
    *,
    checkpoint: Path,
    job_root: Path,
    task_env: str,
    num_rollouts: int,
    horizon: int,
    seed: int,
    headless: bool,
) -> list[str]:
    args = [
        "--task",
        task_env,
        "--checkpoint",
        str(checkpoint),
        "--horizon",
        str(max(1, int(horizon))),
        "--num_rollouts",
        str(max(1, int(num_rollouts))),
        "--seed",
        str(int(seed)),
        "--output_dir",
        str(job_root.resolve()),
        "--enable_cameras",
    ]
    if headless:
        args.append("--headless")
    return args


def _update_status(job_root: Path, payload: dict[str, Any]) -> None:
    existing = _read_json(job_root / "status.json")
    merged = {**existing, **payload, "updatedAt": _now_label()}
    _write_json(job_root / "status.json", merged)


def _load_preview_video_builder():
    import importlib.util

    script_path = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/preview_video_utils.py"
    spec = importlib.util.spec_from_file_location("isaac_preview_video_utils", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "build_preview_from_frames", None)


def _synthesize_eval_videos_from_frames(job_root: Path, per_episode: dict[str, Any]) -> None:
    episodes = per_episode.get("episodes")
    if not isinstance(episodes, list):
        return

    build_preview = _load_preview_video_builder()
    if build_preview is None:
        return

    videos_dir = job_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for row in episodes:
        if not isinstance(row, dict):
            continue
        episode_index = int(row.get("episodeIndex", 0))
        frames_dir = job_root / "artifacts" / f"episode_{episode_index:02d}_frames"
        if not frames_dir.is_dir() or not any(frames_dir.glob("frame_*.jpg")):
            continue
        video_rel = f"videos/episode_{episode_index:02d}.mp4"
        video_path = job_root / video_rel
        if build_preview(frames_dir, video_path, fps=10):
            row["videoPath"] = video_rel


def _transcode_eval_videos(job_root: Path, per_episode: dict[str, Any]) -> None:
    episodes = per_episode.get("episodes")
    if not isinstance(episodes, list):
        return
    for row in episodes:
        if not isinstance(row, dict):
            continue
        rel = row.get("videoPath")
        if not rel:
            continue
        source = job_root / str(rel)
        playable, note = ensure_browser_playable_mp4(source)
        if playable is not None:
            try:
                row["browserVideoPath"] = str(playable.relative_to(job_root))
            except ValueError:
                row["browserVideoPath"] = str(playable)
            row["videoTranscoded"] = note in {"transcoded", "transcoded_cache"}
        else:
            row["videoTranscodeError"] = note
    _write_json(job_root / "results" / "per_episode_results.json", per_episode)


def _enrich_aggregate_result(
    job_root: Path,
    *,
    model_asset_id: str,
    model_manifest: dict[str, Any],
    asset: dict[str, Any],
) -> dict[str, Any]:
    aggregate = _read_json(job_root / "results" / "aggregate_result.json")
    if not aggregate:
        return {}

    aggregate.setdefault("taskType", ISAAC_STACK_TASK_TYPE)
    aggregate.setdefault("taskTemplateId", ISAAC_STACK_TEMPLATE_ID)
    aggregate["evaluationMode"] = "trained_model_evaluation"
    aggregate["modelAssetId"] = model_asset_id
    aggregate["backendType"] = ISAAC_EVAL_BACKEND
    aggregate["framework"] = "Robomimic BC"
    aggregate["simulatorBackend"] = "isaac_lab"
    aggregate["obsKeys"] = model_manifest.get("obsKeys") or ["eef_pos", "eef_quat", "gripper_pos", "object"]
    aggregate["actionDim"] = model_manifest.get("actionDim") or ISAAC_STACK_ACTION_DIM
    aggregate["datasetEnv"] = model_manifest.get("datasetEnv")
    aggregate["sourceTrainJobId"] = model_manifest.get("sourceTrainJobId") or asset.get("sourceTrainingJobId")
    aggregate["sourceDatasetId"] = model_manifest.get("sourceDatasetId") or asset.get("sourceDatasetId")
    per_episode = _read_json(job_root / "results" / "per_episode_results.json")
    aggregate = attach_isaac_eval_metric_metadata(
        aggregate,
        per_episode if isinstance(per_episode, dict) else {},
    )
    _write_json(job_root / "results" / "aggregate_result.json", aggregate)
    return aggregate


def execute_isaac_evaluation(
    eval_job_id: str,
    job_root: Path,
    request: EvaluateAsyncRequest,
    normalized: NormalizedEvaluateRequest,
    *,
    asset: dict[str, Any],
    model_manifest: dict[str, Any],
    checkpoint: Path,
) -> None:
    logs_dir = job_root / "logs"
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    capability = probe_isaac_eval_capability()
    if not capability.get("ready"):
        message = "Isaac 评测未就绪：" + "; ".join(capability.get("issues") or [])
        _update_status(job_root, {"status": "failed", "message": message, "progress": 0.0})
        stderr_path.write_text(message + "\n", encoding="utf-8")
        sync_workspace_job_from_runtime(eval_job_id)
        return

    runner = IsaacLabCliRunner.from_settings()
    if runner.root is None:
        _update_status(job_root, {"status": "failed", "message": "ISAACLAB_ROOT 未配置", "progress": 0.0})
        sync_workspace_job_from_runtime(eval_job_id)
        return

    try:
        _sync_platform_script(runner)
    except OSError as exc:
        _update_status(job_root, {"status": "failed", "message": str(exc), "progress": 0.0})
        sync_workspace_job_from_runtime(eval_job_id)
        return

    task_env = str(model_manifest.get("taskEnv") or DEFAULT_TASK_ENV)
    horizon = int(getattr(request, "horizon", None) or DEFAULT_HORIZON)
    seed = int(request.seed if request.seed is not None else 0)
    num_rollouts = int(request.numEpisodes or 1)
    headless = bool(request.headless)

    cli_args = _build_cli_args(
        checkpoint=checkpoint,
        job_root=job_root,
        task_env=task_env,
        num_rollouts=num_rollouts,
        horizon=horizon,
        seed=seed,
        headless=headless,
    )
    command = runner.build_command(POLICY_EVAL_SCRIPT, *cli_args)

    _write_json(
        job_root / "metadata" / "evaluation_request.json",
        {
            **request.model_dump(mode="json", by_alias=True),
            "evalJobId": eval_job_id,
            "taskType": normalized.task_type,
            "taskTemplateId": normalized.task_template_id,
            "evaluationMode": normalized.public_evaluation_mode,
            "modelAssetId": normalized.model_asset_id,
            "checkpointPath": str(checkpoint),
            "command": command,
            "submittedAt": _now_label(),
        },
    )

    _update_status(
        job_root,
        {
            "evalJobId": eval_job_id,
            "status": "running",
            "phase": "policy_rollout",
            "progress": 0.05,
            "message": "Isaac Robomimic BC rollout 评测进行中",
            "taskType": normalized.task_type,
            "evaluationMode": normalized.public_evaluation_mode,
            "totalEpisodes": num_rollouts,
            "currentEpisode": 0,
            "command": command,
        },
    )

    timeout = int(getattr(settings, "ISAACLAB_EVAL_TIMEOUT", 3600) or 3600)
    result = runner.run_to_files(
        POLICY_EVAL_SCRIPT,
        *cli_args,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=timeout,
    )

    aggregate_path = job_root / "results" / "aggregate_result.json"
    per_episode_path = job_root / "results" / "per_episode_results.json"

    if result.timed_out:
        _update_status(
            job_root,
            {"status": "failed", "message": f"Isaac 评测超时（{timeout}s）", "progress": 0.99},
        )
        sync_workspace_job_from_runtime(eval_job_id)
        return

    if result.returncode != 0:
        _update_status(
            job_root,
            {
                "status": "failed",
                "message": f"Isaac 评测进程失败（exit={result.returncode}）",
                "progress": 1.0,
            },
        )
        sync_workspace_job_from_runtime(eval_job_id)
        return

    if not aggregate_path.is_file() or not per_episode_path.is_file():
        _update_status(
            job_root,
            {
                "status": "failed",
                "message": "评测结束但未生成 aggregate_result.json / per_episode_results.json",
                "progress": 1.0,
            },
        )
        sync_workspace_job_from_runtime(eval_job_id)
        return

    per_episode = _read_json(per_episode_path)
    if "episodes" not in per_episode:
        per_episode = {"episodes": per_episode if isinstance(per_episode, list) else []}

    _synthesize_eval_videos_from_frames(job_root, per_episode)
    _transcode_eval_videos(job_root, per_episode)
    _write_json(per_episode_path, per_episode)

    aggregate = _enrich_aggregate_result(
        job_root,
        model_asset_id=str(normalized.model_asset_id or ""),
        model_manifest=model_manifest,
        asset=asset,
    )

    _update_status(
        job_root,
        {
            "status": "completed",
            "phase": "completed",
            "progress": 1.0,
            "message": "Isaac Robomimic BC 评测完成",
            "metrics": aggregate,
            "artifacts": {
                "aggregateResult": str(aggregate_path),
                "perEpisodeResults": str(per_episode_path),
                "runtimePath": str(job_root),
            },
        },
    )
    sync_workspace_job_from_runtime(eval_job_id)


def _spawn_worker(
    eval_job_id: str,
    job_root: Path,
    request: EvaluateAsyncRequest,
    normalized: NormalizedEvaluateRequest,
    asset: dict[str, Any],
    model_manifest: dict[str, Any],
    checkpoint: Path,
) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_JOBS.add(eval_job_id)
    try:
        execute_isaac_evaluation(
            eval_job_id,
            job_root,
            request,
            normalized,
            asset=asset,
            model_manifest=model_manifest,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        logger.exception("isaac eval worker failed: %s", exc)
        _update_status(job_root, {"status": "failed", "message": str(exc), "progress": 0.0})
        sync_workspace_job_from_runtime(eval_job_id)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_JOBS.discard(eval_job_id)


def start_isaac_evaluation_async(
    request: EvaluateAsyncRequest,
    normalized: NormalizedEvaluateRequest,
) -> dict[str, Any]:
    if normalized.public_evaluation_mode != "trained_model_evaluation":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Isaac 物块堆叠仅支持 trained_model_evaluation",
        )
    model_asset_id = str(normalized.model_asset_id or request.modelAssetId or "").strip()
    if not model_asset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="modelAssetId required for Isaac trained_model_evaluation",
        )

    asset, model_manifest, checkpoint = validate_isaac_robomimic_model_asset(
        model_asset_id,
        normalized.checkpoint_path,
    )

    eval_job_id = make_isaac_eval_job_id()
    job_root = prepare_eval_job_root(eval_job_id)
    user_task_name = _resolve_isaac_eval_model_name(request)
    task_name, generated_display_name = resolve_evaluation_task_name(
        request,
        normalized.task_type,
        normalized.public_evaluation_mode,
    )

    _write_json(
        job_root / "status.json",
        {
            "evalJobId": eval_job_id,
            "taskType": normalized.task_type,
            "taskTemplateId": normalized.task_template_id,
            "evaluationMode": normalized.public_evaluation_mode,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "message": "Isaac 评测任务已排队",
            "createdAt": _now_label(),
            "displayName": generated_display_name,
            "taskDisplayName": resolve_task_display_name(normalized.task_type),
            "evaluationTypeLabel": resolve_evaluation_type_label(normalized.public_evaluation_mode),
            "taskName": task_name,
            **({"originalName": user_task_name} if user_task_name else {}),
        },
    )

    record_workspace_job_start(
        job_id=eval_job_id,
        job_type="evaluation",
        task_type=normalized.task_type,
        runtime_path=str(job_root.relative_to(PROJECT_ROOT) if job_root.is_relative_to(PROJECT_ROOT) else job_root),
        runner="stack_cube_policy_eval.py",
        status="running",
        task_name=task_name,
        metadata={
            "taskTemplateId": normalized.task_template_id,
            "evaluationMode": normalized.public_evaluation_mode,
            "displayName": generated_display_name,
            "templateDisplayName": generated_display_name,
            "taskDisplayName": resolve_task_display_name(normalized.task_type),
            "evaluationTypeLabel": resolve_evaluation_type_label(normalized.public_evaluation_mode),
            "originalName": user_task_name,
            "modelAssetId": model_asset_id,
            "numEpisodes": request.numEpisodes,
            **({"modelName": user_task_name} if user_task_name else {}),
        },
    )

    thread = threading.Thread(
        target=_spawn_worker,
        args=(eval_job_id, job_root, request, normalized, asset, model_manifest, checkpoint),
        daemon=True,
        name=f"isaac-eval-{eval_job_id}",
    )
    thread.start()

    runtime_rel = str(job_root.relative_to(PROJECT_ROOT)) if job_root.is_relative_to(PROJECT_ROOT) else str(job_root)
    return {
        "evalJobId": eval_job_id,
        "taskType": normalized.task_type,
        "taskTemplateId": normalized.task_template_id,
        "evaluationMode": normalized.public_evaluation_mode,
        "status": "running",
        "runtimePath": runtime_rel,
        "resultPath": str(job_root / "results" / "aggregate_result.json"),
    }


def load_isaac_eval_job_root(eval_job_id: str) -> Path:
    from app.services.evaluation.job_paths import eval_job_dir

    return eval_job_dir(eval_job_id)


def read_isaac_eval_status(eval_job_id: str) -> dict[str, Any]:
    job_root = load_isaac_eval_job_root(eval_job_id)
    if not job_root.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Isaac evaluation job not found")
    status_payload = _read_json(job_root / "status.json")
    if not status_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Isaac evaluation job not found")
    sync_workspace_job_from_runtime(eval_job_id)
    from app.services.evaluation_workbench_basic_info import attach_workbench_basic_info

    return attach_workbench_basic_info(status_payload, eval_job_id=eval_job_id, job_root=job_root)


def read_isaac_eval_result(eval_job_id: str) -> dict[str, Any]:
    job_root = load_isaac_eval_job_root(eval_job_id)
    aggregate = _read_json(job_root / "results" / "aggregate_result.json")
    per_episode = _read_json(job_root / "results" / "per_episode_results.json")
    if not aggregate:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="evaluation result not ready")
    return {"aggregate": aggregate, "episodes": per_episode.get("episodes") or []}


def read_isaac_eval_log(eval_job_id: str) -> str:
    job_root = load_isaac_eval_job_root(eval_job_id)
    parts: list[str] = []
    for name in ("stdout.log", "stderr.log"):
        path = job_root / "logs" / name
        if path.is_file():
            try:
                parts.append(f"--- {name} ---\n" + path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n\n".join(parts)


def resolve_isaac_eval_video_path(eval_job_id: str, episode_id: Optional[int] = None) -> Optional[Path]:
    job_root = load_isaac_eval_job_root(eval_job_id)
    per_episode = _read_json(job_root / "results" / "per_episode_results.json")
    episodes = per_episode.get("episodes") or []
    if not episodes:
        return None

    index = 0 if episode_id is None else int(episode_id)
    row = next((item for item in episodes if isinstance(item, dict) and item.get("episodeIndex") == index), None)
    if row is None and 0 <= index < len(episodes) and isinstance(episodes[index], dict):
        row = episodes[index]
    if not isinstance(row, dict):
        return None

    for key in ("browserVideoPath", "videoPath"):
        rel = row.get(key)
        if not rel:
            continue
        candidate = Path(str(rel))
        path = candidate if candidate.is_absolute() else job_root / candidate
        if path.is_file() and path.stat().st_size > 0:
            playable, _ = ensure_browser_playable_mp4(path)
            return playable or path
    return None
