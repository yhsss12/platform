from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.benchmark_adapters.base import (
    BenchmarkCapabilities,
    BenchmarkTaskAdapter,
    is_platform_eval_job_id,
)
from app.services.benchmark_adapters.response import build_evaluate_async_response
from app.services.evaluation.base import EvaluationJob, utc_now_iso
from app.services.evaluation.dual_arm_cable_adapter import DualArmCableEvaluationAdapter
from app.services.evaluation.evaluation_request_resolver import (
    NormalizedEvaluateRequest,
    normalize_evaluate_request,
    to_internal_adapter_request,
)
from app.services.evaluation.job_paths import (
    EVAL_OUTPUT_ROOT,
    eval_job_dir,
    prepare_eval_job_root,
    validate_eval_job_id,
)
from app.services.task_config_metadata import build_job_resource_metadata
from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)
from app.services.evaluation.display_name import (
    build_evaluation_display_name,
    resolve_evaluation_task_name,
    resolve_evaluation_type_label,
    resolve_task_display_name,
)
from app.services.evaluation.evaluation_type import enrich_evaluation_request_payload

_INTERNAL_ADAPTER = DualArmCableEvaluationAdapter()


def _make_eval_job_id() -> str:
    from app.services.evaluation.job_paths import make_eval_job_id

    return make_eval_job_id()


def _write_evaluation_request(job_root: Path, request: EvaluateAsyncRequest) -> None:
    payload = enrich_evaluation_request_payload(request.model_dump(mode="json", by_alias=True))
    payload["submittedAt"] = utc_now_iso()
    path = job_root / "metadata" / "evaluation_request.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_evaluation_context(
    job_root: Path,
    request: EvaluateAsyncRequest,
    normalized: NormalizedEvaluateRequest,
) -> None:
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    enriched = enrich_evaluation_request_payload(request.model_dump(mode="json", by_alias=True))
    payload = {
        "taskTemplateId": normalized.task_template_id,
        "taskType": normalized.task_type,
        "evaluationMode": normalized.public_evaluation_mode,
        "evaluationRequest": enriched,
        "evaluationObject": enriched.get("evaluationObject"),
        "productEvaluationMode": enriched.get("productEvaluationMode"),
        "evaluationType": enriched.get("evaluationType"),
        "evaluationTypeLabel": enriched.get("evaluationTypeLabel"),
    }
    from app.services.evaluation.selected_evaluation_metrics import normalize_selected_metric_ids

    raw_selected: list[str] = []
    if isinstance(enriched.get("metrics"), list):
        raw_selected.extend(str(item) for item in enriched["metrics"] if str(item).strip())
    if isinstance(request.metrics, list):
        raw_selected.extend(str(item) for item in request.metrics if str(item).strip())
    if isinstance(request.config, dict):
        config_metrics = request.config.get("metrics")
        if isinstance(config_metrics, list):
            raw_selected.extend(str(item) for item in config_metrics if str(item).strip())
    for key in ("selectedMetricIds", "selectedMetricKeys"):
        value = enriched.get(key)
        if isinstance(value, list):
            raw_selected.extend(str(item) for item in value if str(item).strip())
    selected_metric_ids = normalize_selected_metric_ids(raw_selected, normalized.task_type)
    if selected_metric_ids:
        payload["selectedMetricIds"] = selected_metric_ids
        payload["metrics"] = selected_metric_ids
        if isinstance(payload["evaluationRequest"], dict):
            payload["evaluationRequest"]["selectedMetricIds"] = selected_metric_ids
    (meta_dir / "evaluation_context.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _attach_dual_arm_metric_results(job_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    aggregate_path = job_root / "results" / "aggregate_result.json"
    if not aggregate_path.is_file():
        return payload
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if not isinstance(aggregate, dict):
        return payload

    from app.services.evaluation.selected_evaluation_metrics import finalize_selected_evaluation_metrics

    status_value = str(payload.get("status") or "").lower()
    finalized = finalize_selected_evaluation_metrics(
        aggregate,
        job_root,
        payload.get("selectedMetricIds") if isinstance(payload.get("selectedMetricIds"), list) else None,
        task_type="dual_arm_cable_manipulation",
        persist=status_value in {"completed", "failed"},
        legacy_fallback=True,
    )
    payload["selectedMetricIds"] = finalized["selectedMetricIds"]
    payload["metricResults"] = finalized["metricResults"]
    if isinstance(finalized.get("runMetrics"), dict):
        payload["runMetrics"] = finalized["runMetrics"]
    return payload


def _load_request_or_404(job_root: Path) -> dict[str, Any]:
    if not job_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation job not found",
        )
    request_path = job_root / "metadata" / "evaluation_request.json"
    if not request_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation job not found",
        )
    return json.loads(request_path.read_text(encoding="utf-8"))


class DualArmCableTaskAdapter(BenchmarkTaskAdapter):
    task_template_id = "dual_arm_cable_manipulation"
    task_type = "dual_arm_cable_manipulation"
    task_family = "cable_manipulation"
    simulator_type = "mujoco"
    supported_evaluation_modes = ["episode_stability", "trained_model_evaluation"]

    def get_capabilities(self) -> BenchmarkCapabilities:
        return BenchmarkCapabilities(
            task_template_id=self.task_template_id,
            task_type=self.task_type,
            task_family=self.task_family,
            simulator_type=self.simulator_type,
            supported_evaluation_modes=list(self.supported_evaluation_modes),
            supported_policy_types=["torch_bc"],
            supports_checkpoint=True,
            supports_policy_evaluation=True,
            supports_episode_stability=True,
            supports_train_model_evaluation=True,
            supports_video=True,
            result_artifact="aggregate_result.json",
            description=(
                "线缆整理：支持 episode 稳定性评测与 torch_bc 训练模型 MuJoCo rollout 评测。"
            ),
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
        internal_request = to_internal_adapter_request(request, normalized)
        if not _INTERNAL_ADAPTER.supports_mode(normalized.internal_evaluation_mode):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"evaluationMode={normalized.internal_evaluation_mode} not supported for "
                    f"taskType={normalized.task_type}"
                ),
            )
        _INTERNAL_ADAPTER.validate_request(internal_request)

        eval_job_id = _make_eval_job_id()
        job_root = prepare_eval_job_root(eval_job_id)
        enriched_request = enrich_evaluation_request_payload(
            internal_request.model_dump(mode="json", by_alias=True)
        )
        _write_evaluation_request(job_root, internal_request)
        _write_evaluation_context(job_root, internal_request, normalized)

        log_path = job_root / "logs" / "eval.log"
        log_path.write_text(
            f"[evaluation_service] evalJobId={eval_job_id} taskType={normalized.task_type} "
            f"mode={normalized.internal_evaluation_mode} started_at={utc_now_iso()}\n",
            encoding="utf-8",
        )

        try:
            job: EvaluationJob = _INTERNAL_ADAPTER.start_async(
                internal_request,
                eval_job_id=eval_job_id,
                job_root=job_root,
            )
        except HTTPException as exc:
            if exc.status_code == status.HTTP_501_NOT_IMPLEMENTED:
                import shutil

                shutil.rmtree(job_root, ignore_errors=True)
            raise

        task_name, generated_display_name = resolve_evaluation_task_name(
            request,
            normalized.task_type,
            normalized.public_evaluation_mode,
        )

        record_workspace_job_start(
            job_id=eval_job_id,
            job_type="evaluation",
            task_type=normalized.task_type,
            runtime_path=str(job_root),
            runner="dual_arm_cable_eval_worker.py",
            status=job.status,
            task_name=task_name,
            metadata={
                **build_job_resource_metadata(
                    task_type=normalized.task_type,
                    task_config_id=request.taskConfigId,
                    extra={
                        "taskTemplateId": normalized.task_template_id,
                        "evaluationMode": normalized.public_evaluation_mode,
                        "evaluationObject": enriched_request.get("evaluationObject"),
                        "productEvaluationMode": enriched_request.get("productEvaluationMode"),
                        "evaluationType": enriched_request.get("evaluationType"),
                        "evaluationTypeLabel": enriched_request.get("evaluationTypeLabel"),
                        "evaluationRequest": enriched_request,
                        "displayName": generated_display_name,
                        "templateDisplayName": generated_display_name,
                        "taskDisplayName": resolve_task_display_name(normalized.task_type),
                    "originalName": str(
                        (request.dualArmCable or {}).get("modelName")
                        or (request.dualArmCable or {}).get("taskName")
                        or (request.cableThreading or {}).get("modelName")
                        or (request.cableThreading or {}).get("taskName")
                        or request.modelName
                        or request.taskName
                        or ""
                    ).strip()
                    or None,
                    "numEpisodes": request.numEpisodes,
                    "datasetId": normalized.dataset_id,
                    "modelAssetId": normalized.model_asset_id,
                    **(
                        {
                            "modelName": str(
                                (request.dualArmCable or {}).get("modelName")
                                or (request.dualArmCable or {}).get("taskName")
                                or (request.cableThreading or {}).get("modelName")
                                or (request.cableThreading or {}).get("taskName")
                                or request.modelName
                                or request.taskName
                                or ""
                            ).strip()
                        }
                        if str(
                            (request.dualArmCable or {}).get("modelName")
                            or (request.dualArmCable or {}).get("taskName")
                            or (request.cableThreading or {}).get("modelName")
                            or (request.cableThreading or {}).get("taskName")
                            or request.modelName
                            or request.taskName
                            or ""
                        ).strip()
                        else {}
                    ),
                },
            ),
            },
        )

        result_path = str(job_root / "results" / "aggregate_result.json")
        return build_evaluate_async_response(
            eval_job_id=job.eval_job_id,
            task_type=job.task_type,
            task_template_id=normalized.task_template_id,
            evaluation_mode=normalized.public_evaluation_mode,
            status=job.status,
            runtime_path=str(job_root),
            result_path=result_path,
        )

    def get_status(self, eval_job_id: str) -> dict[str, Any]:
        validate_eval_job_id(eval_job_id)
        sync_workspace_job_from_runtime(eval_job_id)
        job_root = eval_job_dir(eval_job_id)
        request_data = _load_request_or_404(job_root)
        if str(request_data.get("taskType")) != self.task_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="evaluation job not found for dual_arm adapter",
            )
        payload = _INTERNAL_ADAPTER.get_status(eval_job_id, job_root).to_dict()
        return _attach_dual_arm_metric_results(job_root, payload)

    def get_result(self, eval_job_id: str) -> dict[str, Any]:
        validate_eval_job_id(eval_job_id)
        job_root = eval_job_dir(eval_job_id)
        request_data = _load_request_or_404(job_root)
        if str(request_data.get("taskType")) != self.task_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="evaluation job not found for dual_arm adapter",
            )
        return _INTERNAL_ADAPTER.get_result(eval_job_id, job_root)

    def get_log(self, eval_job_id: str) -> str:
        validate_eval_job_id(eval_job_id)
        job_root = eval_job_dir(eval_job_id)
        request_data = _load_request_or_404(job_root)
        if str(request_data.get("taskType")) != self.task_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="evaluation job not found for dual_arm adapter",
            )
        return _INTERNAL_ADAPTER.get_log(eval_job_id, job_root)

    def get_video(self, eval_job_id: str, episode_id: Optional[int] = None) -> Optional[Path]:
        validate_eval_job_id(eval_job_id)
        job_root = eval_job_dir(eval_job_id)
        request_data = _load_request_or_404(job_root)
        if str(request_data.get("taskType")) != self.task_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="evaluation job not found for dual_arm adapter",
            )
        return _INTERNAL_ADAPTER.get_video_path(eval_job_id, job_root, episode=episode_id)

    def recognizes_eval_job_id(self, eval_job_id: str) -> bool:
        if not is_platform_eval_job_id(eval_job_id):
            return False
        job_root = eval_job_dir(eval_job_id)
        if not job_root.is_dir():
            return False
        request_path = job_root / "metadata" / "evaluation_request.json"
        if not request_path.is_file():
            return False
        try:
            data = json.loads(request_path.read_text(encoding="utf-8"))
            return str(data.get("taskType")) == self.task_type
        except (OSError, json.JSONDecodeError):
            return False
