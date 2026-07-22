from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.benchmark_adapters.base import BenchmarkCapabilities, BenchmarkTaskAdapter
from app.services.benchmark_adapters.response import build_evaluate_async_response
from app.services.evaluation.evaluation_request_resolver import (
    NormalizedEvaluateRequest,
    normalize_evaluate_request,
)
from app.services.evaluation.job_paths import is_isaac_eval_job_id, validate_eval_job_id
from app.services.isaac_lab.eval_service import (
    read_isaac_eval_log,
    read_isaac_eval_result,
    read_isaac_eval_status,
    resolve_isaac_eval_video_path,
    start_isaac_evaluation_async,
)
from app.services.isaac_lab.isaac_runtime_service import check_runtime, RUNTIME_NOT_CONFIGURED_MSG

ISAAC_BLOCK_STACKING_TEMPLATE_ID = "isaac_block_stacking"
ISAAC_BLOCK_STACKING_TASK_TYPE = "block_stacking"
ISAAC_RUNTIME_UNAVAILABLE_MSG = RUNTIME_NOT_CONFIGURED_MSG


class IsaacLabBlockStackingAdapter(BenchmarkTaskAdapter):
    """Isaac Lab 物块堆叠：Isaac Robomimic BC 训练模型 rollout 评测。"""

    task_template_id = ISAAC_BLOCK_STACKING_TEMPLATE_ID
    task_type = ISAAC_BLOCK_STACKING_TASK_TYPE
    task_family = "manipulation_core"
    simulator_type = "isaac"
    supported_evaluation_modes: list[str] = ["trained_model_evaluation"]

    @property
    def template_ids(self) -> list[str]:
        return [ISAAC_BLOCK_STACKING_TEMPLATE_ID]

    def check_runtime(self) -> dict[str, Any]:
        return check_runtime()

    def get_capabilities(self) -> BenchmarkCapabilities:
        runtime = self.check_runtime()
        return BenchmarkCapabilities(
            task_template_id=self.task_template_id,
            task_type=self.task_type,
            task_family=self.task_family,
            simulator_type=self.simulator_type,
            supported_evaluation_modes=list(self.supported_evaluation_modes),
            supported_policy_types=["isaac_robomimic_bc"],
            supports_checkpoint=True,
            supports_policy_evaluation=True,
            supports_episode_stability=False,
            supports_train_model_evaluation=True,
            supports_video=True,
            result_artifact="aggregate_result.json",
            description=(
                "Isaac Lab 物块堆叠：支持 Isaac Robomimic BC 训练模型 rollout 评测，"
                "输出成功率、aggregate_result.json 与 episode 视频。"
            ),
            extensions={
                "simulatorBackend": "isaac_lab",
                "physicsBackend": "physx",
                "requiresExternalRuntime": True,
                "defaultEnv": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
                "supportsDatasetGeneration": True,
                "supportsTraining": True,
                "supportsEvaluation": True,
                "supportsReplay": True,
                "scriptedExpertAvailable": False,
                "datasetFormats": ["hdf5"],
                "trainingBackends": ["isaac_robomimic_bc"],
                "evaluationBackends": ["isaac_robomimic_bc"],
                "adapterStatus": "ready" if runtime.get("ok") else "runtime_unavailable",
            },
        )

    def normalize_request(self, request: EvaluateAsyncRequest) -> NormalizedEvaluateRequest:
        normalized = normalize_evaluate_request(request)
        if (
            normalized.task_template_id != self.task_template_id
            and normalized.task_type != self.task_type
        ):
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
        runtime = self.check_runtime()
        if not runtime.get("ok"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=runtime.get("message") or ISAAC_RUNTIME_UNAVAILABLE_MSG,
            )

        result = start_isaac_evaluation_async(request, normalized)
        return build_evaluate_async_response(
            eval_job_id=str(result["evalJobId"]),
            task_type=self.task_type,
            task_template_id=normalized.task_template_id,
            evaluation_mode=normalized.public_evaluation_mode,
            status=str(result.get("status") or "running"),
            runtime_path=str(result.get("runtimePath") or ""),
            result_path=str(result.get("resultPath") or ""),
        )

    def get_status(self, eval_job_id: str) -> dict[str, Any]:
        if not is_isaac_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Isaac Lab eval job ID format",
            )
        return read_isaac_eval_status(eval_job_id)

    def get_result(self, eval_job_id: str) -> dict[str, Any]:
        if not is_isaac_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Isaac Lab eval job ID format",
            )
        return read_isaac_eval_result(eval_job_id)

    def get_log(self, eval_job_id: str) -> str:
        if not is_isaac_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Isaac Lab eval job ID format",
            )
        return read_isaac_eval_log(eval_job_id)

    def get_video(self, eval_job_id: str, episode_id: Optional[int] = None) -> Optional[Path]:
        if not is_isaac_eval_job_id(eval_job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Isaac Lab eval job ID format",
            )
        return resolve_isaac_eval_video_path(eval_job_id, episode_id=episode_id)

    def recognizes_eval_job_id(self, eval_job_id: str) -> bool:
        if not is_isaac_eval_job_id(eval_job_id):
            return False
        from app.services.isaac_lab.eval_service import load_isaac_eval_job_root

        job_root = load_isaac_eval_job_root(eval_job_id)
        request_path = job_root / "metadata" / "evaluation_request.json"
        if not request_path.is_file():
            return job_root.is_dir()
        try:
            data = json.loads(request_path.read_text(encoding="utf-8"))
            return str(data.get("taskType")) in {self.task_type, "isaac_block_stacking"}
        except (OSError, json.JSONDecodeError):
            return False
