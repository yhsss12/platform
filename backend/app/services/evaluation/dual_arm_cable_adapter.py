from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.base import (
    BaseEvaluationAdapter,
    EvaluationJob,
    EvaluationStatus,
    utc_now_iso,
)
from app.services.evaluation.dual_arm_cable_eval_worker import (
    resolve_eval_params,
    resolve_policy_eval_params,
    spawn_evaluation_worker,
)
from app.services.workspace_model_asset_service import get_model_asset_by_id
from app.services.training_backend_canonical import resolve_asset_training_backend

_POLICY_TYPES = frozenset({"robomimic", "robomimic_bc", "scripted", "random", "act", "dt", "diffusion"})
_DUAL_ARM_TEMPLATE_IDS = frozenset(
    {"dual_arm_cable_manipulation", "task_dual_arm_cable_manipulation_v1"}
)
_MAX_EVALUATION_EPISODES = 100


def _load_model_manifest(asset: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(asset.get("manifestPath") or ""))
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _validate_torch_bc_model_asset(model_asset_id: str, checkpoint_path: str) -> dict[str, Any]:
    asset = get_model_asset_by_id(model_asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAssetId not found: {model_asset_id}",
        )
    backend_type = resolve_asset_training_backend(asset)
    if backend_type != "torch_bc":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset backendType must be torch_bc, got {backend_type!r}",
        )
    task_template_id = str(asset.get("taskTemplateId") or "")
    if task_template_id and task_template_id not in _DUAL_ARM_TEMPLATE_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset taskTemplateId must be dual_arm_cable_manipulation, got {task_template_id!r}",
        )
    manifest = _load_model_manifest(asset)
    action_dim = manifest.get("actionDim")
    if action_dim is not None and int(action_dim) != 14:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset actionDim must be 14, got {action_dim}",
        )
    obs_schema = str(manifest.get("observationSchema") or "")
    if obs_schema and obs_schema != "dual_arm_cable_il_v1":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"modelAsset observationSchema must be dual_arm_cable_il_v1, got {obs_schema!r}",
        )
    ckpt = Path(checkpoint_path).expanduser()
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"checkpointPath not found or empty: {checkpoint_path}",
        )
    return asset


class DualArmCableEvaluationAdapter(BaseEvaluationAdapter):
    """Dual-arm episode 稳定性与 torch_bc 训练模型 rollout 评测适配器。"""

    task_type = "dual_arm_cable_manipulation"
    supported_modes = ["episode_stability", "trained_model_evaluation"]

    def validate_request(self, request: EvaluateAsyncRequest) -> None:
        mode = str(request.evaluationMode or "")
        if mode not in self.supported_modes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"dual_arm_cable_manipulation supports {self.supported_modes}, got {mode!r}"
                ),
            )
        if request.numEpisodes < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="numEpisodes must be >= 1",
            )
        if request.numEpisodes > _MAX_EVALUATION_EPISODES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Episodes 超出上限，当前最大支持 {_MAX_EVALUATION_EPISODES}。",
            )

        if mode == "episode_stability":
            self._validate_episode_stability_request(request)
            return
        self._validate_trained_model_request(request)

    def _validate_episode_stability_request(self, request: EvaluateAsyncRequest) -> None:
        params = request.dualArmCable or {}
        policy = params.get("policyType") or request.policyType
        if policy and str(policy).lower() in _POLICY_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="episode_stability does not support policy evaluation fields",
            )
        if params.get("policyType") or request.policyType or request.checkpointId or request.checkpointPath:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="episode_stability does not support checkpoint or policy evaluation",
            )
        if request.modelAssetId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="episode_stability does not require modelAssetId",
            )

        seeds = list(request.seeds or params.get("seeds") or [])
        if not seeds:
            if request.seed is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="seeds required for episode_stability evaluation (or provide seed)",
                )
            seeds = [int(request.seed) + i for i in range(request.numEpisodes)]
        if request.numEpisodes != len(seeds):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="numEpisodes must match len(seeds)",
            )

    def _validate_trained_model_request(self, request: EvaluateAsyncRequest) -> None:
        params = request.dualArmCable or {}
        model_asset_id = str(params.get("modelAssetId") or request.modelAssetId or "").strip()
        checkpoint_path = str(
            params.get("checkpointPath")
            or request.checkpointPath
            or request.checkpointId
            or ""
        ).strip()
        if not model_asset_id and not checkpoint_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="modelAssetId or checkpointPath required for trained_model_evaluation",
            )
        if not checkpoint_path and model_asset_id:
            asset = get_model_asset_by_id(model_asset_id)
            if asset:
                checkpoint_path = str(asset.get("checkpointPath") or "").strip()
        if not checkpoint_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="checkpointPath unavailable for trained_model_evaluation",
            )
        if model_asset_id:
            _validate_torch_bc_model_asset(model_asset_id, checkpoint_path)
        else:
            ckpt = Path(checkpoint_path).expanduser()
            if not ckpt.is_file() or ckpt.stat().st_size <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"checkpointPath not found or empty: {checkpoint_path}",
                )

        try:
            policy_params = resolve_policy_eval_params(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        seeds = policy_params["seeds"]
        if request.numEpisodes != len(seeds):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="numEpisodes must match len(seeds)",
            )

    def start_async(
        self,
        request: EvaluateAsyncRequest,
        *,
        eval_job_id: str,
        job_root: Path,
    ) -> EvaluationJob:
        mode = str(request.evaluationMode or "episode_stability")
        if mode == "trained_model_evaluation":
            params = resolve_policy_eval_params(request)
            total = len(params["seeds"])
        else:
            params = resolve_eval_params(request)
            total = len(params["seeds"])

        payload = {
            "evalJobId": eval_job_id,
            "taskType": self.task_type,
            "evaluationMode": request.evaluationMode,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "currentEpisode": 0,
            "totalEpisodes": total,
            "message": "评测任务已创建",
            "metrics": {},
            "artifacts": {},
            "startedAt": utc_now_iso(),
            "updatedAt": utc_now_iso(),
        }
        (job_root / "status.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        spawn_evaluation_worker(eval_job_id, job_root, request)

        return EvaluationJob(
            eval_job_id=eval_job_id,
            task_type=self.task_type,
            evaluation_mode=request.evaluationMode,
            job_root=job_root,
            status="queued",
        )

    def get_status(self, eval_job_id: str, job_root: Path) -> EvaluationStatus:
        return self._read_platform_status(eval_job_id, job_root)

    def get_result(self, eval_job_id: str, job_root: Path) -> dict[str, Any]:
        result_path = job_root / "results" / "aggregate_result.json"
        if result_path.is_file():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        status_obj = self._read_platform_status(eval_job_id, job_root)
        return {
            "evalJobId": eval_job_id,
            "taskType": self.task_type,
            "evaluationMode": status_obj.evaluation_mode,
            "status": status_obj.status,
            "message": status_obj.message,
        }

    def get_log(self, eval_job_id: str, job_root: Path) -> str:
        log_path = job_root / "logs" / "eval.log"
        if not log_path.is_file():
            return ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-80:])
        except OSError:
            return ""

    def get_video_path(
        self,
        eval_job_id: str,
        job_root: Path,
        *,
        episode: Optional[int] = None,
    ) -> Optional[Path]:
        from app.services.evaluation_replay_info import resolve_episode_video_path

        return resolve_episode_video_path(job_root, episode=episode)

    def _read_platform_status(self, eval_job_id: str, job_root: Path) -> EvaluationStatus:
        status_path = job_root / "status.json"
        if not status_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"evaluation job not found: {eval_job_id}",
            )
        raw = json.loads(status_path.read_text(encoding="utf-8"))
        return EvaluationStatus(
            eval_job_id=str(raw.get("evalJobId") or eval_job_id),
            task_type=str(raw.get("taskType") or self.task_type),
            evaluation_mode=str(raw.get("evaluationMode") or "episode_stability"),
            status=str(raw.get("status") or "queued"),
            phase=raw.get("phase"),
            progress=raw.get("progress"),
            current_episode=raw.get("currentEpisode"),
            total_episodes=raw.get("totalEpisodes"),
            message=str(raw.get("message") or ""),
            metrics=dict(raw.get("metrics") or {}),
            artifacts=dict(raw.get("artifacts") or {}),
            updated_at=raw.get("updatedAt"),
        )
