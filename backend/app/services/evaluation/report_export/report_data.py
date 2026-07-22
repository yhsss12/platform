from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from fastapi import HTTPException, status

from app.services.evaluation.job_paths import PROJECT_ROOT, eval_job_dir, validate_eval_job_id
from app.services.evaluation.report_export.exporters._utils import dash, format_file_size, utc_now_iso
from app.services.evaluation.metric_policy import (
    is_report_body_metric,
    partition_metric_results_for_report,
)
from app.services.evaluation.dual_arm_runtime_metrics import (
    DUAL_ARM_TASK_TYPE,
    build_dual_arm_episode_metric_rows,
)
from app.services.evaluation.sim_time_metrics import compute_episode_sim_time_sec, enrich_run_metrics_sim_time
from app.services.evaluation.selected_evaluation_metrics import finalize_selected_evaluation_metrics
from app.services.evaluation_workbench_basic_info import build_evaluation_workbench_basic_info

logger = logging.getLogger(__name__)

from app.core.platform_paths import platform_paths

REPORT_OUTPUT_ROOT = platform_paths.runs_root / "evaluation_reports"

RUNTIME_FILE_SPECS: list[tuple[str, str, str]] = [
    ("evaluation_context.json", "metadata/evaluation_context.json", "评测上下文"),
    ("evaluation_request.json", "metadata/evaluation_request.json", "评测请求"),
    ("aggregate_result.json", "results/aggregate_result.json", "聚合结果"),
    ("per_episode_results.json", "results/per_episode_results.json", "逐 episode 结果"),
    ("status.json", "status.json", "任务状态"),
    ("live_status.json", "live/status.json", "实时状态"),
    ("eval.log", "logs/eval.log", "评测日志"),
    ("eval.results.json", "results/eval.results.json", "评测结果"),
]


class EvaluationReportData(TypedDict, total=False):
    reportMeta: dict[str, Any]
    basicInfo: dict[str, Any]
    evaluationConfig: dict[str, Any]
    evaluationObject: dict[str, Any]
    metricResults: dict[str, Any]
    selectedMetricIds: list[str]
    runMetrics: dict[str, Any]
    episodeResults: list[dict[str, Any]]
    videoInfo: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    runtimeFiles: list[dict[str, Any]]
    rawSources: dict[str, Any]
    legacyNotice: str | None
    rawRunMetrics: dict[str, Any]
    deprecatedMetrics: list[dict[str, Any]]
    unknownMetrics: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_json_list(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("episodes", "items", "results", "perEpisode"):
                nested = data.get(key)
                if isinstance(nested, list):
                    return nested
        return []
    except (OSError, json.JSONDecodeError):
        return []


def _pick_str(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _iso_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(microsecond=0).isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.replace(microsecond=0).isoformat()
    except ValueError:
        return text


def _load_aggregate(job_root: Path, status: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    for source in (
        result,
        status.get("metrics") if isinstance(status.get("metrics"), dict) else {},
        _read_json(job_root / "results" / "aggregate_result.json"),
        _read_json(job_root / "results" / "eval.results.json"),
    ):
        if not isinstance(source, dict) or not source:
            continue
        nested = source.get("aggregate")
        if isinstance(nested, dict) and nested:
            merged = dict(nested)
            for key, value in source.items():
                if key != "aggregate" and key not in merged:
                    merged[key] = value
            return merged
        if source.get("metricResults") or source.get("success_rate") is not None or source.get("successRate") is not None:
            return dict(source)
    return {}


def _resolve_selected_metric_ids(
    *,
    status: dict[str, Any],
    result: dict[str, Any],
    aggregate: dict[str, Any],
    job_root: Path,
) -> list[str]:
    for source in (
        status.get("selectedMetricIds"),
        result.get("selectedMetricIds"),
        aggregate.get("selectedMetricIds"),
        status.get("metrics", {}).get("selectedMetricIds") if isinstance(status.get("metrics"), dict) else None,
        _read_json(job_root / "metadata" / "evaluation_context.json").get("selectedMetricIds"),
        _read_json(job_root / "metadata" / "evaluation_request.json").get("selectedMetricIds"),
        _read_json(job_root / "metadata" / "evaluation_context.json").get("metrics"),
        _read_json(job_root / "metadata" / "evaluation_request.json").get("metrics"),
    ):
        if isinstance(source, list) and source:
            return [str(item).strip() for item in source if str(item).strip()]
    return []


def _resolve_metric_results(
    *,
    status: dict[str, Any],
    result: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    for source in (
        status.get("metricResults"),
        result.get("metricResults"),
        aggregate.get("metricResults"),
        status.get("metrics", {}).get("metricResults") if isinstance(status.get("metrics"), dict) else None,
    ):
        if isinstance(source, dict) and source:
            return {str(k): v for k, v in source.items() if isinstance(v, dict)}
    return {}


def _resolve_run_metrics(
    *,
    status: dict[str, Any],
    result: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    for source in (
        aggregate.get("runMetrics"),
        result.get("runMetrics"),
        status.get("runMetrics"),
        status.get("metrics", {}).get("runMetrics") if isinstance(status.get("metrics"), dict) else None,
    ):
        if isinstance(source, dict) and source:
            return dict(source)
    return {}


def _filter_body_metric_results(
    selected_metric_ids: list[str],
    metric_results: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return partition_metric_results_for_report(selected_metric_ids, metric_results)


def _build_evaluation_config(
    *,
    status: dict[str, Any],
    result: dict[str, Any],
    aggregate: dict[str, Any],
    job_root: Path,
    selected_metric_ids: list[str],
) -> dict[str, Any]:
    context = _read_json(job_root / "metadata" / "evaluation_context.json")
    request = _read_json(job_root / "metadata" / "evaluation_request.json")
    nested_request = request.get("evaluationRequest")
    if isinstance(nested_request, dict):
        request = {**request, **nested_request}
    config = context.get("config") if isinstance(context.get("config"), dict) else {}
    if not config and isinstance(request.get("config"), dict):
        config = request.get("config")

    return {
        "episodes": _pick_str(
            status.get("totalEpisodes"),
            status.get("numEpisodes"),
            aggregate.get("total_episodes"),
            aggregate.get("episodes"),
            aggregate.get("episodeCount"),
            request.get("numEpisodes"),
            config.get("numEpisodes"),
        )
        or "-",
        "horizon": dash(
            _pick_str(
                request.get("horizon"),
                context.get("horizon"),
                config.get("horizon"),
                aggregate.get("horizon"),
            )
        ),
        "seed": dash(
            _pick_str(request.get("seed"), context.get("seed"), config.get("seed"))
        ),
        "recordVideo": dash(
            _pick_str(
                request.get("record"),
                context.get("record"),
                config.get("record"),
            )
            or ("true" if aggregate.get("recorded_video_count") is not None else "")
        ),
        "selectedMetricIds": selected_metric_ids,
        "simulationPlatform": dash(
            _pick_str(
                status.get("simulationPlatform"),
                context.get("simulationPlatform"),
                request.get("simulationPlatform"),
            )
        ),
        "robotType": dash(
            _pick_str(status.get("robotType"), context.get("robotType"), request.get("robotType"))
        ),
        "taskType": dash(_pick_str(status.get("taskType"), result.get("taskType"), aggregate.get("taskType"))),
        "modelAssetId": dash(
            _pick_str(
                status.get("modelAssetId"),
                result.get("modelAssetId"),
                request.get("modelAssetId"),
                context.get("modelAssetId"),
            )
        ),
        "modelAssetName": dash(
            _pick_str(
                status.get("modelAssetName"),
                result.get("modelAssetName"),
                request.get("modelAssetName"),
            )
        ),
        "policyName": dash(
            _pick_str(request.get("policyType"), context.get("policyType"), result.get("policyMode"))
        ),
        "datasetName": dash(
            _pick_str(status.get("datasetName"), request.get("datasetName"), context.get("datasetName"))
        ),
    }


def _load_step_metric_summaries(job_root: Path) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    step_root = job_root / "results" / "step_metrics"
    if not step_root.is_dir():
        return summaries
    for episode_dir in sorted(step_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        summary_path = episode_dir / "summary.json"
        summary = _read_json(summary_path)
        if not summary:
            continue
        episode_index = summary.get("episodeIndex")
        if episode_index is None:
            name = episode_dir.name
            digits = "".join(ch for ch in name if ch.isdigit())
            episode_index = int(digits) if digits else None
        if episode_index is None:
            continue
        summaries[int(episode_index)] = summary
    return summaries


def _normalize_episode_row(
    item: dict[str, Any],
    *,
    step_summary: dict[str, Any] | None,
    control_frequency_hz: Any = None,
) -> dict[str, Any]:
    episode_index = item.get("episodeIndex", item.get("episode", item.get("index")))
    success_raw = item.get("success")
    if success_raw is None:
        success_raw = item.get("episodeSuccess")
    if success_raw is None:
        success_label = "-"
    elif success_raw:
        success_label = "成功"
    else:
        success_label = "失败"

    step_count_raw = _pick_str(
        item.get("stepCount"),
        item.get("stepsExecuted"),
        item.get("numTransitions"),
        item.get("steps"),
        step_summary.get("stepCount") if step_summary else "",
    )
    control_hz = _pick_str(
        control_frequency_hz,
        step_summary.get("controlFrequencyHz") if step_summary else "",
        item.get("controlFrequencyHz"),
    )
    sim_time_value = compute_episode_sim_time_sec(
        step_count=step_count_raw,
        control_frequency_hz=control_hz,
        existing_sim_time=item.get("simTimeSec") or (step_summary.get("simTimeSec") if step_summary else None),
    )
    sim_time = f"{sim_time_value:.4f}".rstrip("0").rstrip(".") if sim_time_value is not None else "-"

    max_action_norm = _pick_str(
        item.get("maxActionNorm"),
        step_summary.get("maxActionNorm") if step_summary else "",
    )
    smoothness = _pick_str(
        item.get("smoothnessScore"),
        step_summary.get("smoothnessScore") if step_summary else "",
    )
    failure_reason = _pick_str(item.get("failureReason"), item.get("error"), step_summary.get("error") if step_summary else "")

    return {
        "episodeIndex": episode_index if episode_index is not None else "-",
        "successLabel": success_label,
        "stepCount": step_count_raw or "-",
        "simTimeSec": sim_time,
        "maxActionNorm": max_action_norm or "-",
        "smoothnessScore": smoothness or "-",
        "failureReason": failure_reason or "-",
        "videoFile": _pick_str(item.get("videoPath"), item.get("videoFile")) or "-",
    }


def _format_sim_time_sec(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _apply_dual_arm_episode_overlay(row: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    step_count = overlay.get("stepCount")
    if isinstance(step_count, (int, float)) and step_count >= 0:
        merged["stepCount"] = str(int(step_count)) if float(step_count).is_integer() else str(step_count)
    sim_time = overlay.get("simTimeSec")
    if isinstance(sim_time, (int, float)):
        merged["simTimeSec"] = _format_sim_time_sec(float(sim_time))
    max_action_norm = overlay.get("maxActionNorm")
    if isinstance(max_action_norm, (int, float)):
        merged["maxActionNorm"] = str(round(float(max_action_norm), 4))
    smoothness = overlay.get("smoothnessScore")
    if isinstance(smoothness, (int, float)):
        merged["smoothnessScore"] = str(round(float(smoothness), 4))
    return merged


def _build_episode_results(
    job_root: Path,
    status: dict[str, Any],
    aggregate: dict[str, Any],
    result: dict[str, Any],
    run_metrics: dict[str, Any],
    *,
    task_type: str = "",
) -> list[dict[str, Any]]:
    step_summaries = _load_step_metric_summaries(job_root)
    control_frequency_hz = run_metrics.get("controlFrequencyHz")
    episodes: list[dict[str, Any]] = []
    dual_arm_overlays: dict[int, dict[str, Any]] = {}
    if task_type == DUAL_ARM_TASK_TYPE:
        dual_arm_overlays = build_dual_arm_episode_metric_rows(job_root)

    per_episode_path = job_root / "results" / "per_episode_results.json"
    raw_items = _read_json_list(per_episode_path)
    if not raw_items:
        for source in (aggregate.get("perEpisode"), result.get("perEpisode"), status.get("episodeResults")):
            if isinstance(source, list) and source:
                raw_items = source
                break

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        episode_index = item.get("episodeIndex", item.get("episode", item.get("index")))
        step_summary = None
        if episode_index is not None:
            step_summary = step_summaries.get(int(episode_index))
        row = _normalize_episode_row(
            item,
            step_summary=step_summary,
            control_frequency_hz=control_frequency_hz,
        )
        if episode_index is not None:
            overlay = dual_arm_overlays.get(int(episode_index))
            if overlay:
                row = _apply_dual_arm_episode_overlay(row, overlay)
        episodes.append(row)

    if episodes:
        return episodes

    for episode_index, summary in sorted(step_summaries.items()):
        episodes.append(
            _normalize_episode_row(
                {"episodeIndex": episode_index, "success": summary.get("success")},
                step_summary=summary,
                control_frequency_hz=control_frequency_hz,
            )
        )
    return episodes


def _probe_video_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if fps and fps > 0:
                meta["fps"] = round(float(fps), 3)
            if frame_count and fps and fps > 0:
                meta["durationSec"] = round(float(frame_count) / float(fps), 3)
            if width and height:
                meta["resolution"] = f"{int(width)}x{int(height)}"
        capture.release()
    except Exception:
        pass
    return meta


def _build_video_info(job_root: Path, status: dict[str, Any], aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    representative = status.get("representativeVideoEpisode")
    replay_uris = status.get("replayUris") if isinstance(status.get("replayUris"), list) else []
    indexed: dict[int, dict[str, Any]] = {}

    for item in replay_uris:
        if not isinstance(item, dict):
            continue
        episode_index = item.get("episodeIndex")
        if episode_index is None:
            continue
        indexed[int(episode_index)] = {
            "episodeIndex": int(episode_index),
            "videoFilename": dash(item.get("fileName")),
            "videoPath": dash(item.get("uri")),
            "durationSec": "-",
            "fps": "-",
            "resolution": "-",
            "isRepresentative": representative is not None and int(episode_index) == int(representative),
        }

    videos_dir = job_root / "videos"
    if videos_dir.is_dir():
        for path in sorted(videos_dir.glob("*.mp4")):
            stem = path.stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            episode_index = int(digits) if digits else None
            meta = _probe_video_metadata(path)
            row = indexed.get(episode_index or -1, {})
            indexed[episode_index if episode_index is not None else len(indexed)] = {
                "episodeIndex": episode_index if episode_index is not None else "-",
                "videoFilename": path.name,
                "videoPath": str(path.relative_to(job_root)),
                "durationSec": dash(meta.get("durationSec")),
                "fps": dash(meta.get("fps")),
                "resolution": dash(meta.get("resolution")),
                "isRepresentative": bool(row.get("isRepresentative"))
                or (representative is not None and episode_index == int(representative)),
            }

    if not indexed and aggregate.get("videoPath"):
        indexed[0] = {
            "episodeIndex": "-",
            "videoFilename": Path(str(aggregate.get("videoPath"))).name,
            "videoPath": str(aggregate.get("videoPath")),
            "durationSec": "-",
            "fps": "-",
            "resolution": "-",
            "isRepresentative": True,
        }

    return [indexed[key] for key in sorted(indexed.keys(), key=lambda x: (x == -1, x))]


def _read_log_tail(job_root: Path, lines: int = 40) -> str:
    for rel in ("logs/eval.log", "logs/run.log", "logs/worker.log"):
        path = job_root / rel
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(content[-lines:])
        except OSError:
            continue
    return ""


def _build_diagnostics(job_root: Path, status: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    job_status = str(status.get("status") or "").lower()
    message = _pick_str(status.get("message"), status.get("error"), aggregate.get("message"))
    log_tail = _read_log_tail(job_root)
    missing_files = [
        row["relativePath"]
        for row in _build_runtime_files(job_root)
        if not row.get("exists") and row.get("relativePath") in {
            "results/aggregate_result.json",
            "results/per_episode_results.json",
        }
    ]
    runtime_health = status.get("runtimeHealth") if isinstance(status.get("runtimeHealth"), dict) else {}
    timed_out = bool(status.get("timedOut") or runtime_health.get("timedOut"))
    abnormal_exit = job_status in {"failed", "error", "cancelled"} or bool(status.get("error"))

    if job_status in {"completed", "success", "succeeded"} and not message:
        summary = "无明显异常"
    elif message:
        summary = message
    else:
        summary = "任务状态异常，请查看日志"

    return {
        "summary": summary,
        "failureReason": message or "-",
        "errorMessage": dash(status.get("error")),
        "logTail": log_tail or "-",
        "runtimeHealth": runtime_health or {},
        "timedOut": timed_out,
        "abnormalExit": abnormal_exit,
        "missingFiles": missing_files,
    }


def _build_runtime_files(job_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_row(filename: str, relative_path: str, description: str) -> None:
        if relative_path in seen:
            return
        seen.add(relative_path)
        path = job_root / relative_path
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        rows.append(
            {
                "filename": filename,
                "relativePath": relative_path,
                "exists": exists,
                "sizeBytes": size,
                "sizeLabel": format_file_size(size),
                "description": description,
            }
        )

    for filename, relative_path, description in RUNTIME_FILE_SPECS:
        add_row(filename, relative_path, description)

    step_root = job_root / "results" / "step_metrics"
    if step_root.is_dir():
        for episode_dir in sorted(step_root.iterdir()):
            if not episode_dir.is_dir():
                continue
            rel = f"results/step_metrics/{episode_dir.name}/summary.json"
            add_row(f"{episode_dir.name}_summary.json", rel, "逐步指标摘要")

    videos_dir = job_root / "videos"
    if videos_dir.is_dir():
        for video in sorted(videos_dir.glob("*.mp4")):
            rel = f"videos/{video.name}"
            add_row(video.name, rel, "episode 视频")

    return rows


def build_evaluation_report_data(job_id: str) -> EvaluationReportData:
    validated = validate_eval_job_id(job_id)
    job_root = eval_job_dir(validated)
    if not job_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "ok": False,
                "error": "未找到评测任务",
                "jobId": validated,
                "hint": "请确认导出使用的是真实 job_id（eval_* / ct_eval_* / isaac_eval_*），而不是任务名称",
            },
        )

    from app.services.evaluation.evaluation_service import get_evaluation_result, get_evaluation_status

    status_payload = get_evaluation_status(validated)
    result_payload: dict[str, Any] = {}
    try:
        result_payload = get_evaluation_result(validated)
    except HTTPException:
        logger.debug("evaluation result unavailable for report export job=%s", validated)

    aggregate = _load_aggregate(job_root, status_payload, result_payload)
    task_type = _pick_str(status_payload.get("taskType"), result_payload.get("taskType"), aggregate.get("taskType"))

    selected_metric_ids = _resolve_selected_metric_ids(
        status=status_payload,
        result=result_payload,
        aggregate=aggregate,
        job_root=job_root,
    )
    metric_results = _resolve_metric_results(
        status=status_payload,
        result=result_payload,
        aggregate=aggregate,
    )
    run_metrics = _resolve_run_metrics(
        status=status_payload,
        result=result_payload,
        aggregate=aggregate,
    )
    run_metrics = enrich_run_metrics_sim_time(run_metrics)

    raw_selected_metric_ids = list(selected_metric_ids)
    legacy_notice: str | None = None
    try:
        finalized = finalize_selected_evaluation_metrics(
            aggregate or {},
            job_root,
            selected_metric_ids or None,
            task_type=task_type or "cable_threading",
            persist=False,
            legacy_fallback=True,
        )
        selected_metric_ids = finalized.get("selectedMetricIds") or selected_metric_ids
        metric_results = finalized.get("metricResults") or metric_results
        run_metrics = finalized.get("runMetrics") or run_metrics
        run_metrics = enrich_run_metrics_sim_time(run_metrics)
        aggregate = finalized.get("aggregate") or aggregate
    except Exception as exc:
        logger.warning("failed to finalize metrics for report job=%s: %s", validated, exc)

    if not raw_selected_metric_ids and not _read_json(job_root / "metadata" / "evaluation_context.json").get(
        "selectedMetricIds"
    ):
        legacy_notice = "该任务为旧格式，部分运行指标缺少原始数据。"

    partition_selected_ids = list(
        dict.fromkeys(raw_selected_metric_ids + (selected_metric_ids or []))
    )
    body_metric_results, deprecated_metrics, unknown_metrics = _filter_body_metric_results(
        partition_selected_ids,
        metric_results,
    )
    selected_metric_ids = [mid for mid in selected_metric_ids if is_report_body_metric(mid)]

    workbench = status_payload.get("workbenchBasicInfo")
    if not isinstance(workbench, dict):
        workbench = build_evaluation_workbench_basic_info(validated, job_root, status_payload)

    basic_info = {
        "taskName": dash(workbench.get("taskName") or status_payload.get("taskName")),
        "jobId": validated,
        "evaluationTypeLabel": dash(workbench.get("evaluationTypeLabel")),
        "associatedTaskName": dash(workbench.get("associatedTaskName")),
        "simulationPlatform": dash(workbench.get("simulationPlatform")),
        "statusLabel": dash(workbench.get("statusLabel") or status_payload.get("status")),
        "evaluationObjectLabel": dash(workbench.get("evaluationObjectLabel")),
        "modelAssetName": dash(workbench.get("modelAssetName")),
        "robotType": dash(workbench.get("robotType")),
        "createdAt": dash(_iso_time(status_payload.get("startedAt") or status_payload.get("createdAt"))),
        "finishedAt": dash(_iso_time(status_payload.get("finishedAt") or status_payload.get("updatedAt"))),
    }

    evaluation_config = _build_evaluation_config(
        status=status_payload,
        result=result_payload,
        aggregate=aggregate,
        job_root=job_root,
        selected_metric_ids=selected_metric_ids,
    )

    evaluation_object = {
        "evaluationType": dash(workbench.get("evaluationType")),
        "evaluationObject": dash(workbench.get("evaluationObject")),
        "evaluationMode": dash(status_payload.get("evaluationMode")),
        "taskType": dash(task_type),
    }

    episode_results = _build_episode_results(
        job_root,
        status_payload,
        aggregate,
        result_payload,
        run_metrics,
        task_type=task_type,
    )
    video_info = _build_video_info(job_root, status_payload, aggregate)
    diagnostics = _build_diagnostics(job_root, status_payload, aggregate)
    runtime_files = _build_runtime_files(job_root)

    return EvaluationReportData(
        reportMeta={
            "reportTitle": "评测报告",
            "generatedAt": utc_now_iso(),
            "exportVersion": "v1",
            "jobId": validated,
            "taskName": basic_info["taskName"],
        },
        basicInfo=basic_info,
        evaluationConfig=evaluation_config,
        evaluationObject=evaluation_object,
        selectedMetricIds=selected_metric_ids,
        metricResults=body_metric_results,
        runMetrics=run_metrics,
        rawRunMetrics=dict(run_metrics),
        deprecatedMetrics=deprecated_metrics,
        unknownMetrics=unknown_metrics,
        episodeResults=episode_results,
        videoInfo=video_info,
        diagnostics=diagnostics,
        runtimeFiles=runtime_files,
        rawSources={
            "jobRoot": str(job_root),
            "statusPath": str(job_root / "status.json"),
            "aggregatePath": str(job_root / "results" / "aggregate_result.json"),
        },
        legacyNotice=legacy_notice,
    )
