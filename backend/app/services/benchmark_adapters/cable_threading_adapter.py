from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services import cable_threading_service as ct_svc
from app.services.benchmark_adapters.base import (
    BenchmarkCapabilities,
    BenchmarkTaskAdapter,
    is_ct_eval_job_id,
)
from app.services.benchmark_adapters.response import build_evaluate_async_response
from app.services.evaluation.base import utc_now_iso
from app.services.evaluation.display_name import build_evaluation_display_name
from app.services.evaluation.evaluation_type import enrich_evaluation_request_payload
from app.services.evaluation.evaluation_request_resolver import (
    NormalizedEvaluateRequest,
    normalize_evaluate_request,
)
from app.services.cable_threading_eval_params import (
    DEFAULT_CABLE_EVAL_DISPLAY_CAMERA,
    resolve_cable_eval_episodes,
    resolve_cable_eval_horizon,
    resolve_cable_eval_seed,
    resolve_cable_record_video,
    resolve_cable_eval_display_camera,
    resolve_cable_allow_camera_fallback,
)
from app.services.workspace_job_service import sync_workspace_job_from_runtime


def _resolve_runtime_updated_at(job_root: Path, live: dict[str, Any]) -> Optional[str]:
    updated = live.get("updatedAt") or live.get("updated_at")
    if updated:
        return str(updated)
    status_path = job_root / "live" / "status.json"
    if status_path.is_file():
        try:
            mtime = status_path.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
        except OSError:
            pass
    return None


class CableThreadingTaskAdapter(BenchmarkTaskAdapter):
    task_template_id = "cable_threading_single_arm"
    task_type = "cable_threading"
    task_family = "cable_manipulation"
    simulator_type = "mujoco"
    supported_evaluation_modes = ["expert_policy_evaluation", "trained_model_evaluation"]

    def get_capabilities(self) -> BenchmarkCapabilities:
        return BenchmarkCapabilities(
            task_template_id=self.task_template_id,
            task_type=self.task_type,
            task_family=self.task_family,
            simulator_type=self.simulator_type,
            supported_evaluation_modes=list(self.supported_evaluation_modes),
            supported_policy_types=["scripted", "robomimic", "diffusion_policy", "act", "pi0"],
            supports_checkpoint=True,
            supports_policy_evaluation=True,
            supports_episode_stability=False,
            supports_train_model_evaluation=True,
            supports_video=True,
            result_artifact="eval.results.json",
            description="线缆穿杆：支持专家策略评测与训练模型评测。",
        )

    def normalize_request(self, request: EvaluateAsyncRequest) -> NormalizedEvaluateRequest:
        normalized = normalize_evaluate_request(request)
        if normalized.task_type != self.task_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"adapter {self.task_template_id} does not support taskType={normalized.task_type}",
            )
        return normalized

    def start_evaluation(
        self,
        request: EvaluateAsyncRequest,
        normalized: NormalizedEvaluateRequest,
    ) -> dict[str, Any]:
        params = request.cableThreading or {}
        model_name = (
            str(
                params.get("modelName")
                or params.get("taskName")
                or request.modelName
                or request.taskName
                or ""
            ).strip()
            or None
        )
        result = ct_svc.start_evaluate_async(
            episodes=resolve_cable_eval_episodes(request),
            robot=str(normalized.robot or params.get("robot") or "Panda"),
            cable_model=str(params.get("cableModel") or "composite_cable"),
            difficulty=str(params.get("difficulty") or "easy"),
            horizon=int(params.get("horizon") or resolve_cable_eval_horizon(request)),
            seed=resolve_cable_eval_seed(request),
            policy=normalized.policy or "scripted",
            checkpoint=normalized.checkpoint_path,
            device=str(params.get("device") or ""),
            task_config_id=request.taskConfigId,
            model_name=model_name,
            record_video=resolve_cable_record_video(request),
            eval_display_camera=resolve_cable_eval_display_camera(request),
            allow_camera_fallback=resolve_cable_allow_camera_fallback(request),
            eval_executor=normalized.eval_executor,
            controller_type=normalized.controller_type,
            action_mode=normalized.action_mode,
            train_config_path=normalized.train_config_path,
            task_instruction=normalized.task_instruction,
            model_asset_id=normalized.model_asset_id,
            source_train_job_id=normalized.source_train_job_id,
            state_dim=normalized.state_dim,
            action_dim=normalized.action_dim,
        )
        eval_job_id = str(result["evalJobId"])
        job_root = ct_svc._job_dir(eval_job_id)
        self._write_evaluation_context(job_root, normalized, request)
        result_path = str(job_root / "results" / "eval.results.json")
        return build_evaluate_async_response(
            eval_job_id=eval_job_id,
            task_type=self.task_type,
            task_template_id=normalized.task_template_id,
            evaluation_mode=normalized.public_evaluation_mode,
            status=str(result.get("status") or "running"),
            runtime_path=str(job_root),
            result_path=result_path,
        )

    def get_status(self, eval_job_id: str) -> dict[str, Any]:
        if not is_ct_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cable threading eval job ID format",
            )
        sync_workspace_job_from_runtime(eval_job_id)
        ct_status = ct_svc.get_job_status(eval_job_id)
        job_root = ct_svc._job_dir(eval_job_id)
        context = self._read_evaluation_context(job_root)

        live = ct_status.get("live") or {}
        metrics = ct_status.get("metrics") or {}
        total = int(live.get("episodes") or metrics.get("episodes") or metrics.get("numEpisodes") or 0)
        completed = int(live.get("completedEpisodes") or 0)
        progress_raw = live.get("progressPercent")
        progress = float(progress_raw) / 100.0 if progress_raw is not None else None
        if progress is None and total > 0:
            progress = min(1.0, completed / total)

        status_value = str(ct_status.get("status") or "running")
        phase = "evaluating" if status_value in {"queued", "running"} else status_value
        eval_request = context.get("evaluationRequest") if isinstance(context.get("evaluationRequest"), dict) else {}
        evaluation_mode = str(
            context.get("evaluationMode")
            or context.get("productEvaluationMode")
            or eval_request.get("evaluationMode")
            or eval_request.get("productEvaluationMode")
            or "expert_policy_evaluation"
        )
        if evaluation_mode == "scripted":
            evaluation_mode = "expert_policy_evaluation"
        elif evaluation_mode in {"robomimic", "diffusion_policy", "act", "pi0"}:
            evaluation_mode = "trained_model_evaluation"

        updated_at = _resolve_runtime_updated_at(job_root, live)
        default_messages = {
            "completed": "策略评测已完成",
            "failed": "策略评测失败",
            "canceled": "策略评测已取消",
            "cancelled": "策略评测已取消",
        }
        live_message = live.get("message")
        if status_value not in {"queued", "running"} and live_message == "策略评测运行中":
            live_message = None
        message = str(
            live.get("error")
            or live_message
            or default_messages.get(status_value, "策略评测运行中")
        )
        if status_value == "failed" and live.get("error"):
            message = str(live.get("error"))

        artifacts = {
            "evalCsv": (ct_status.get("paths") or {}).get("evalCsv"),
            "resultsJson": (ct_status.get("paths") or {}).get("resultsJson"),
            "evalVideo": {
                "exists": ct_status.get("evalVideoExists"),
                "path": ct_status.get("evalVideoPath"),
            },
            "runtimePath": str(job_root),
        }

        payload = {
            "evalJobId": eval_job_id,
            "taskType": self.task_type,
            "evaluationMode": evaluation_mode,
            "status": status_value,
            "phase": phase,
            "progress": progress,
            "currentEpisode": completed if completed else None,
            "totalEpisodes": total if total else None,
            "message": message,
            "metrics": metrics,
            "artifacts": artifacts,
            "updatedAt": updated_at,
            "checkedAt": utc_now_iso(),
        }
        from app.services.evaluation_workbench_basic_info import attach_workbench_basic_info

        return attach_workbench_basic_info(payload, eval_job_id=eval_job_id, job_root=job_root)

    def get_result(self, eval_job_id: str) -> dict[str, Any]:
        from app.services.eval_job_db_service import get_evaluation_result_from_db
        from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

        if is_imported_workspace_eval_job_id(eval_job_id):
            db_result = get_evaluation_result_from_db(eval_job_id)
            if db_result:
                return db_result
        if not is_ct_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cable threading eval job ID format",
            )
        return ct_svc.get_eval_job_result(eval_job_id)

    def get_log(self, eval_job_id: str) -> str:
        from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id, resolve_eval_job_root

        if is_imported_workspace_eval_job_id(eval_job_id):
            job_root = resolve_eval_job_root(eval_job_id)
            if job_root is not None:
                for log_path in (
                    job_root.parent / "eval.log",
                    job_root / "logs" / "run.log",
                    job_root / "run.log",
                ):
                    if log_path.is_file():
                        try:
                            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                            return "\n".join(lines[-80:])
                        except OSError:
                            pass
            return ""
        if not is_ct_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cable threading eval job ID format",
            )
        return ct_svc.read_job_log_tail(eval_job_id)

    def get_video(self, eval_job_id: str, episode_id: Optional[int] = None) -> Optional[Path]:
        from app.services.evaluation_replay_info import resolve_episode_video_path
        from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id, resolve_eval_job_root

        if is_imported_workspace_eval_job_id(eval_job_id):
            job_root = resolve_eval_job_root(eval_job_id)
            if job_root is None:
                return None
            path = resolve_episode_video_path(job_root, episode_id)
            if path is not None and path.name.endswith(".mp4"):
                browser = path.with_name(f"{path.stem}.browser{path.suffix}")
                if browser.is_file() and browser.stat().st_size > 0:
                    return browser
            return path
        if not is_ct_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cable threading eval job ID format",
            )
        if episode_id is not None and episode_id != 0:
            return None
        return ct_svc.resolve_job_video_path(eval_job_id)

    def recognizes_eval_job_id(self, eval_job_id: str) -> bool:
        from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

        return is_ct_eval_job_id(eval_job_id) or is_imported_workspace_eval_job_id(eval_job_id)

    def _write_evaluation_context(
        self,
        job_root: Path,
        normalized: NormalizedEvaluateRequest,
        request: EvaluateAsyncRequest,
    ) -> None:
        meta_dir = job_root / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "taskTemplateId": normalized.task_template_id,
            "evaluationMode": normalized.public_evaluation_mode,
            "datasetId": normalized.dataset_id,
            "modelAssetId": normalized.model_asset_id,
            "taskConfigId": request.taskConfigId,
            "submittedAt": utc_now_iso(),
        }
        if request.config:
            payload["config"] = request.config
        record_camera = resolve_cable_eval_display_camera(request)
        payload["recordCamera"] = record_camera
        payload["evalDisplayCamera"] = record_camera
        payload["cameraFallbackUsed"] = False
        payload["allowCameraFallback"] = resolve_cable_allow_camera_fallback(request)
        if request.metrics:
            payload["metrics"] = request.metrics
        payload["evaluationRequest"] = enrich_evaluation_request_payload(
            request.model_dump(mode="json", by_alias=True)
        )
        from app.services.evaluation.selected_evaluation_metrics import normalize_selected_metric_ids

        raw_selected = []
        if isinstance(payload.get("metrics"), list):
            raw_selected.extend(str(item) for item in payload["metrics"] if str(item).strip())
        config_block = payload.get("config")
        if isinstance(config_block, dict):
            config_metrics = config_block.get("metrics")
            if isinstance(config_metrics, list):
                raw_selected.extend(str(item) for item in config_metrics if str(item).strip())
        eval_req = payload.get("evaluationRequest")
        if isinstance(eval_req, dict):
            for key in ("metrics", "selectedMetricIds", "selectedMetricKeys"):
                value = eval_req.get(key)
                if isinstance(value, list):
                    raw_selected.extend(str(item) for item in value if str(item).strip())
        selected_metric_ids = normalize_selected_metric_ids(raw_selected, "cable_threading")
        if selected_metric_ids:
            payload["selectedMetricIds"] = selected_metric_ids
            payload["metrics"] = selected_metric_ids
            if isinstance(payload["evaluationRequest"], dict):
                payload["evaluationRequest"]["selectedMetricIds"] = selected_metric_ids
                payload["evaluationRequest"]["metrics"] = selected_metric_ids
        for key in ("evaluationObject", "productEvaluationMode", "evaluationType", "evaluationTypeLabel"):
            if payload["evaluationRequest"].get(key) is not None:
                payload[key] = payload["evaluationRequest"][key]
        cable_params = request.cableThreading or {}
        if isinstance(cable_params, dict):
            for key in ("robot", "cableModel", "difficulty", "horizon", "device"):
                if cable_params.get(key) is not None:
                    payload[key] = cable_params.get(key)
        model_name = str(
            (request.cableThreading or {}).get("modelName")
            or (request.cableThreading or {}).get("taskName")
            or request.modelName
            or request.taskName
            or ""
        ).strip()
        if model_name:
            payload["modelName"] = model_name
        if normalized.policy == "pi0":
            payload["modelType"] = normalized.model_type or "pi0"
            payload["policyRuntime"] = normalized.policy_runtime or "pi0"
            payload["policyType"] = "pi0"
            payload["datasetFormat"] = "lerobot"
            if normalized.robot:
                payload["robot"] = normalized.robot
            if normalized.eval_executor:
                payload["evalExecutor"] = normalized.eval_executor
            if normalized.controller_type:
                payload["controllerType"] = normalized.controller_type
            if normalized.action_mode:
                payload["actionMode"] = normalized.action_mode
            if normalized.state_dim is not None:
                payload["stateDim"] = normalized.state_dim
            if normalized.action_dim is not None:
                payload["actionDim"] = normalized.action_dim
            if normalized.task_instruction:
                payload["taskInstruction"] = normalized.task_instruction
            if normalized.train_config_path:
                payload["trainConfigPath"] = normalized.train_config_path
            if normalized.source_train_job_id:
                payload["sourceTrainJobId"] = normalized.source_train_job_id
        generated_display_name = build_evaluation_display_name(
            normalized.task_type, normalized.public_evaluation_mode
        )
        payload["templateDisplayName"] = generated_display_name
        payload["displayName"] = generated_display_name
        (meta_dir / "evaluation_context.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _read_evaluation_context(self, job_root: Path) -> dict[str, Any]:
        path = job_root / "metadata" / "evaluation_context.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
