"""Training task display name helpers (ACT vs DP joint-space)."""
from __future__ import annotations

from typing import Any


def _normalize_backend(model_type: str | None, training_backend: str | None) -> tuple[bool, bool]:
    backend = str(training_backend or "").strip().lower()
    model = str(model_type or "").strip().lower()
    is_act = backend == "act" or model == "act"
    is_dp = backend == "diffusion_policy" or "diffusion" in model
    return is_act, is_dp


def normalize_joint_space_training_display_name(
    name: str,
    *,
    model_type: str | None = None,
    training_backend: str | None = None,
) -> str:
    """Replace misleading Joint-Space DP prefix when the job is ACT joint-space."""
    value = str(name or "").strip()
    if not value:
        return value
    is_act, is_dp = _normalize_backend(model_type, training_backend)
    if is_act and not is_dp and value.startswith("Joint-Space DP"):
        return value.replace("Joint-Space DP", "ACT Joint-Space", 1)
    return value


def build_joint_space_training_task_name(
    *,
    dataset_name: str | None,
    model_type: str | None = None,
    training_backend: str | None = None,
    suffix: str | None = None,
) -> str:
    is_act, is_dp = _normalize_backend(model_type, training_backend)
    prefix = "ACT Joint-Space" if is_act and not is_dp else "Joint-Space DP" if is_dp else ""
    dataset = str(dataset_name or "").strip()
    if dataset.startswith("Joint-Space DP ·"):
        dataset = dataset.split("·", 1)[-1].strip()
    elif dataset.startswith("ACT Joint-Space ·"):
        dataset = dataset.split("·", 1)[-1].strip()
    body = suffix or dataset or "训练任务"
    if prefix:
        return f"{prefix} · {body}"
    return body


def resolve_training_task_display_name(
    *,
    task_name: str | None = None,
    dataset_name: str | None = None,
    model_type: str | None = None,
    training_backend: str | None = None,
    job_id: str | None = None,
) -> str:
    for candidate in (task_name, dataset_name):
        value = normalize_joint_space_training_display_name(
            str(candidate or "").strip(),
            model_type=model_type,
            training_backend=training_backend,
        )
        if value and value.lower() not in {"unknown", "未知任务"}:
            return value
    job = str(job_id or "").strip()
    return job or "未命名任务"


def apply_training_task_name_backfill(
    status_data: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    """Backfill taskName in status payload when missing or misleading for ACT jobs."""
    model_type = str(
        train_config.get("downstreamModelType")
        or status_data.get("downstreamModelType")
        or ""
    )
    training_backend = str(
        train_config.get("trainingBackend")
        or status_data.get("trainingBackendResolved")
        or status_data.get("trainingBackend")
        or ""
    )
    current = str(status_data.get("taskName") or train_config.get("taskName") or "").strip()
    dataset_name = str(status_data.get("datasetName") or train_config.get("datasetName") or "").strip()
    resolved = resolve_training_task_display_name(
        task_name=current or None,
        dataset_name=dataset_name or None,
        model_type=model_type,
        training_backend=training_backend,
        job_id=str(status_data.get("jobId") or ""),
    )
    if resolved and resolved != current:
        status_data["taskName"] = resolved
        train_config["taskName"] = resolved
    return status_data
