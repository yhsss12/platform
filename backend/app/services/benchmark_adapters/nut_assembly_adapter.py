from __future__ import annotations

import json
import os
import subprocess
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
from app.services.evaluation.base import utc_now_iso
from app.services.evaluation.evaluation_request_resolver import (
    NormalizedEvaluateRequest,
    normalize_evaluate_request,
)
from app.services.evaluation.job_paths import eval_job_dir, make_eval_job_id, prepare_eval_job_root
from app.services.task_config_metadata import build_job_resource_metadata
from app.services.workspace_job_service import record_workspace_job_start, sync_workspace_job_from_runtime

_PYTHON = Path("/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python")
_WORKER = Path(__file__).resolve().parents[1] / "evaluation" / "nut_assembly_eval_worker.py"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class NutAssemblyTaskAdapter(BenchmarkTaskAdapter):
    task_template_id = "nut_assembly_single_arm"
    task_type = "nut_assembly"
    task_family = "rigid_body_assembly"
    simulator_type = "mujoco"
    supported_evaluation_modes = ["trained_model_evaluation"]

    def get_capabilities(self) -> BenchmarkCapabilities:
        return BenchmarkCapabilities(
            task_template_id=self.task_template_id,
            task_type=self.task_type,
            task_family=self.task_family,
            simulator_type=self.simulator_type,
            supported_evaluation_modes=list(self.supported_evaluation_modes),
            supported_policy_types=["robomimic_bc"],
            supports_checkpoint=True,
            supports_policy_evaluation=True,
            supports_train_model_evaluation=True,
            supports_video=True,
            result_artifact="aggregate_result.json",
            description="螺母装配：支持 Robomimic BC 训练模型的 MuJoCo rollout 评测。",
        )

    def normalize_request(self, request: EvaluateAsyncRequest) -> NormalizedEvaluateRequest:
        normalized = normalize_evaluate_request(request)
        if normalized.task_type != self.task_type:
            raise HTTPException(status_code=400, detail="评测任务类型不是 nut_assembly")
        return normalized

    def start_evaluation(self, request: EvaluateAsyncRequest, normalized: NormalizedEvaluateRequest) -> dict[str, Any]:
        if not normalized.checkpoint_path:
            raise HTTPException(status_code=400, detail="螺母装配模型评测需要 checkpointPath 或 modelAssetId")
        if not _PYTHON.is_file() or not _WORKER.is_file():
            raise HTTPException(status_code=500, detail="螺母装配评测运行环境不完整")

        eval_job_id = make_eval_job_id()
        job_root = prepare_eval_job_root(eval_job_id)
        request_payload = request.model_dump(mode="json", by_alias=True)
        request_payload.update(
            {
                "taskType": self.task_type,
                "taskTemplateId": self.task_template_id,
                "evaluationMode": normalized.public_evaluation_mode,
                "checkpointPath": normalized.checkpoint_path,
                "submittedAt": utc_now_iso(),
            }
        )
        (job_root / "metadata" / "evaluation_request.json").write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        initial = {
            "evalJobId": eval_job_id,
            "taskType": self.task_type,
            "evaluationMode": normalized.public_evaluation_mode,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "currentEpisode": 0,
            "totalEpisodes": request.numEpisodes,
            "message": "螺母装配模型评测已创建",
            "metrics": {"modelAssetId": normalized.model_asset_id},
            "artifacts": {},
            "startedAt": utc_now_iso(),
            "updatedAt": utc_now_iso(),
        }
        (job_root / "status.json").write_text(json.dumps(initial, indent=2, ensure_ascii=False), encoding="utf-8")

        cmd = [
            str(_PYTHON), str(_WORKER), "--job-root", str(job_root), "--eval-job-id", eval_job_id,
            "--checkpoint", normalized.checkpoint_path, "--episodes", str(request.numEpisodes),
            "--horizon", str(request.horizon or 600), "--seed", str(request.seed or 0),
            "--model-asset-id", normalized.model_asset_id or "",
        ]
        if request.record:
            cmd.append("--record")
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env.setdefault("MPLCONFIGDIR", "/tmp/eai-nut-eval-matplotlib")
        project_root = Path(__file__).resolve().parents[4]
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root / "integrations" / "CableThreadingMVP"), str(project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        log_file = open(job_root / "logs" / "eval.log", "w", encoding="utf-8")
        subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_file, stderr=subprocess.STDOUT)

        task_name = (request.taskName or request.evaluationTaskName or "螺母装配模型评测").strip()
        record_workspace_job_start(
            job_id=eval_job_id,
            job_type="evaluation",
            task_type=self.task_type,
            runtime_path=str(job_root),
            runner="nut_assembly_eval_worker.py",
            status="queued",
            task_name=task_name,
            metadata=build_job_resource_metadata(
                task_type=self.task_type,
                task_config_id=request.taskConfigId,
                extra={
                    "taskTemplateId": self.task_template_id,
                    "evaluationMode": normalized.public_evaluation_mode,
                    "modelAssetId": normalized.model_asset_id,
                    "numEpisodes": request.numEpisodes,
                },
            ),
        )
        return build_evaluate_async_response(
            eval_job_id=eval_job_id,
            task_type=self.task_type,
            task_template_id=self.task_template_id,
            evaluation_mode=normalized.public_evaluation_mode,
            status="queued",
            runtime_path=str(job_root),
            result_path=str(job_root / "results" / "aggregate_result.json"),
        )

    def _root(self, eval_job_id: str) -> Path:
        root = eval_job_dir(eval_job_id)
        request = _read_json(root / "metadata" / "evaluation_request.json")
        if request.get("taskType") != self.task_type:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="螺母装配评测任务不存在")
        return root

    def get_status(self, eval_job_id: str) -> dict[str, Any]:
        root = self._root(eval_job_id)
        sync_workspace_job_from_runtime(eval_job_id)
        return _read_json(root / "status.json")

    def get_result(self, eval_job_id: str) -> dict[str, Any]:
        root = self._root(eval_job_id)
        result = _read_json(root / "results" / "aggregate_result.json")
        return result or self.get_status(eval_job_id)

    def get_log(self, eval_job_id: str) -> str:
        path = self._root(eval_job_id) / "logs" / "eval.log"
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]) if path.is_file() else ""

    def get_video(self, eval_job_id: str, episode_id: Optional[int] = None) -> Optional[Path]:
        root = self._root(eval_job_id)
        index = 0 if episode_id is None else episode_id
        path = root / "videos" / f"episode_{index:02d}.mp4"
        return path if path.is_file() else None

    def recognizes_eval_job_id(self, eval_job_id: str) -> bool:
        if not is_platform_eval_job_id(eval_job_id):
            return False
        root = eval_job_dir(eval_job_id)
        request = _read_json(root / "metadata" / "evaluation_request.json")
        return request.get("taskType") == self.task_type
