from __future__ import annotations

from typing import Any, Optional

from app.services.evaluation.base import utc_now_iso


def build_unified_urls(eval_job_id: str) -> dict[str, str]:
    base = f"/api/workspace/evaluation/jobs/{eval_job_id}"
    return {
        "statusUrl": f"{base}/status",
        "logUrl": f"{base}/log",
        "resultUrl": f"{base}/result",
    }


def build_evaluate_async_response(
    *,
    eval_job_id: str,
    task_type: str,
    task_template_id: str,
    evaluation_mode: str,
    status: str,
    runtime_path: str,
    result_path: Optional[str] = None,
) -> dict[str, Any]:
    created_at = utc_now_iso()
    urls = build_unified_urls(eval_job_id)
    return {
        "evalJobId": eval_job_id,
        "taskType": task_type,
        "taskTemplateId": task_template_id,
        "evaluationMode": evaluation_mode,
        "status": status if status in {"queued", "running"} else "running",
        "runtimePath": runtime_path,
        "resultPath": result_path,
        "createdAt": created_at,
        **urls,
    }
