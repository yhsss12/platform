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
from app.services import cable_threading_service as ct_svc

class CableThreadingEvaluationAdapter(BaseEvaluationAdapter):
    task_type = "cable_threading"
    supported_modes = ["policy_evaluation"]

    def validate_request(self, request: EvaluateAsyncRequest) -> None:
        if request.evaluationMode != "policy_evaluation":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"cable_threading only supports evaluationMode=policy_evaluation, got {request.evaluationMode}",
            )
        params = request.cableThreading or {}
        policy = params.get("policyType") or request.policyType or "scripted"
        if policy in {"robomimic", "diffusion_policy", "act", "pi0"} and not (params.get("checkpointId") or request.checkpointId):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="checkpointId required for trained model policy evaluation",
            )

    def start_async(
        self,
        request: EvaluateAsyncRequest,
        *,
        eval_job_id: str,
        job_root: Path,
    ) -> EvaluationJob:
        params = request.cableThreading or {}
        policy = str(params.get("policyType") or request.policyType or "scripted")
        checkpoint = params.get("checkpointId") or request.checkpointId

        result = ct_svc.start_evaluate_async(
            episodes=request.numEpisodes,
            robot=str(params.get("robot") or "Panda"),
            cable_model=str(params.get("cableModel") or "composite_cable"),
            difficulty=str(params.get("difficulty") or "easy"),
            horizon=int(params.get("horizon") or 600),
            seed=request.seed if request.seed is not None else 0,
            policy=policy,
            checkpoint=checkpoint,
            device=str(params.get("device") or ""),
        )

        source_job_id = str(result["evalJobId"])
        source_root = ct_svc._job_dir(source_job_id)

        self._write_source_jobs(job_root, source_job_id, source_root)
        self._write_platform_status(
            job_root,
            eval_job_id=eval_job_id,
            request=request,
            status="running",
            phase="evaluating",
            progress=0.0,
            current_episode=0,
            total_episodes=request.numEpisodes,
            message="已启动 cable_threading 策略评测",
            source_job_id=source_job_id,
        )

        return EvaluationJob(
            eval_job_id=eval_job_id,
            task_type=self.task_type,
            evaluation_mode=request.evaluationMode,
            job_root=job_root,
            status="running",
            source_job_ids=[source_job_id],
        )

    def get_status(self, eval_job_id: str, job_root: Path) -> EvaluationStatus:
        source = self._load_source_job(job_root)
        ct_status = ct_svc.get_job_status(source["evalJobId"])

        live = ct_status.get("live") or {}
        metrics = ct_status.get("metrics") or {}
        total = int(live.get("episodes") or metrics.get("episodes") or 0)
        completed = int(live.get("completedEpisodes") or 0)
        progress_raw = live.get("progressPercent")
        progress = float(progress_raw) / 100.0 if progress_raw is not None else None
        if progress is None and total > 0:
            progress = min(1.0, completed / total)

        status_value = str(ct_status.get("status") or "running")
        phase = "evaluating" if status_value in {"queued", "running"} else status_value

        artifacts = {
            "evalCsv": (ct_status.get("paths") or {}).get("evalCsv"),
            "resultsJson": (ct_status.get("paths") or {}).get("resultsJson"),
            "evalVideo": {
                "exists": ct_status.get("evalVideoExists"),
                "path": ct_status.get("evalVideoPath"),
            },
            "sourceJobId": source["evalJobId"],
            "sourceJobRoot": source.get("jobRoot"),
        }

        return EvaluationStatus(
            eval_job_id=eval_job_id,
            task_type=self.task_type,
            evaluation_mode=self._read_evaluation_mode(job_root),
            status=status_value,
            phase=phase,
            progress=progress,
            current_episode=completed if completed else None,
            total_episodes=total if total else None,
            message=str(
                live.get("error")
                or {
                    "completed": "策略评测已完成",
                    "failed": "策略评测失败",
                    "canceled": "策略评测已取消",
                    "cancelled": "策略评测已取消",
                }.get(status_value, "策略评测运行中")
            ),
            metrics=metrics,
            artifacts=artifacts,
            updated_at=utc_now_iso(),
        )

    def get_result(self, eval_job_id: str, job_root: Path) -> dict[str, Any]:
        source = self._load_source_job(job_root)
        source_payload = ct_svc.get_eval_job_result(source["evalJobId"])
        payload = dict(source_payload)
        payload["evalJobId"] = eval_job_id
        payload["sourceJobId"] = source["evalJobId"]
        payload["sourceJobRoot"] = source.get("jobRoot")
        ct_svc._write_eval_report_artifacts(job_root, payload)
        status_path = job_root / "metadata" / "status.json"
        if not status_path.is_file():
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(
                    {
                        "status": payload.get("status"),
                        "progress": 100 if payload.get("status") == "completed" else 0,
                        "success_rate": payload.get("successRate"),
                        "total_episodes": (payload.get("aggregate") or {}).get("total_episodes"),
                        "success_episodes": (payload.get("aggregate") or {}).get("success_episodes"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return payload

    def get_log(self, eval_job_id: str, job_root: Path) -> str:
        source = self._load_source_job(job_root)
        source_tail = ct_svc.read_job_log_tail(source["evalJobId"])
        if source_tail.strip():
            return source_tail
        platform_log = job_root / "logs" / "eval.log"
        if platform_log.is_file():
            try:
                lines = platform_log.read_text(encoding="utf-8", errors="replace").splitlines()
                return "\n".join(lines[-80:])
            except OSError:
                pass
        return ""

    def get_video_path(
        self,
        eval_job_id: str,
        job_root: Path,
        *,
        episode: Optional[int] = None,
    ) -> Optional[Path]:
        if episode is not None and episode != 0:
            return None
        source = self._load_source_job(job_root)
        return ct_svc.resolve_job_video_path(source["evalJobId"])

    def _write_source_jobs(self, job_root: Path, source_job_id: str, source_root: Path) -> None:
        meta_dir = job_root / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cable_threading": {
                "evalJobId": source_job_id,
                "jobRoot": str(source_root),
            }
        }
        (meta_dir / "source_jobs.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_source_job(self, job_root: Path) -> dict[str, Any]:
        path = job_root / "metadata" / "source_jobs.json"
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="evaluation source job mapping not found",
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get("cable_threading")
        if not entry or not entry.get("evalJobId"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="cable_threading source job not linked",
            )
        return entry

    def _read_evaluation_mode(self, job_root: Path) -> str:
        req_path = job_root / "metadata" / "evaluation_request.json"
        if req_path.is_file():
            try:
                return str(json.loads(req_path.read_text(encoding="utf-8")).get("evaluationMode") or "policy_evaluation")
            except (OSError, json.JSONDecodeError):
                pass
        return "policy_evaluation"

    def _write_platform_status(
        self,
        job_root: Path,
        *,
        eval_job_id: str,
        request: EvaluateAsyncRequest,
        status: str,
        phase: str,
        progress: float,
        current_episode: int,
        total_episodes: int,
        message: str,
        source_job_id: str,
    ) -> None:
        payload = EvaluationStatus(
            eval_job_id=eval_job_id,
            task_type=self.task_type,
            evaluation_mode=request.evaluationMode,
            status=status,
            phase=phase,
            progress=progress,
            current_episode=current_episode,
            total_episodes=total_episodes,
            message=message,
            artifacts={"sourceJobId": source_job_id},
            updated_at=utc_now_iso(),
        ).to_dict()
        status_path = job_root / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
