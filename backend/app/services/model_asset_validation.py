"""模型资产：文件系统校验、评测兼容性、列表 enrichment。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.platform_paths import platform_paths, resolve_runtime_reference
from app.services.model_asset_db_service import TASK_TYPE_TO_TEMPLATE

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root

EVALUATION_MODEL_BACKEND_COMPATIBILITY: dict[str, frozenset[str]] = {
    "cable_threading": frozenset(
        {
            "expert_policy",
            "robomimic_bc",
            "robomimic",
            "bc",
            "act",
            "pi0",
            "diffusion_policy",
        }
    ),
    "dual_arm_cable_manipulation": frozenset({"torch_bc", "act", "diffusion_policy", "bc"}),
    "block_stacking": frozenset({"isaac_robomimic_bc"}),
    "isaac_block_stacking": frozenset({"isaac_robomimic_bc"}),
    "nut_assembly": frozenset({"robomimic_bc", "robomimic", "bc"}),
}

BACKEND_ALIASES: dict[str, str] = {
    "act": "act",
    "bc": "bc",
    "pi0": "pi0",
    "robomimic": "robomimic_bc",
    "robomimic_bc": "robomimic_bc",
    "robomimic bc": "robomimic_bc",
    "isaac_robomimic_bc": "isaac_robomimic_bc",
    "isaac robomimic bc": "isaac_robomimic_bc",
    "torch_bc": "torch_bc",
    "diffusion_policy": "diffusion_policy",
    "diffusion policy": "diffusion_policy",
    "diffusion": "diffusion_policy",
    "expert_policy": "expert_policy",
}


@dataclass(frozen=True)
class ModelAssetValidationResult:
    ok: bool
    reason: str
    model_asset_id: str
    artifact_path: str
    backend_type: str
    source_task_type: str
    file_exists: bool
    file_size_bytes: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "modelAssetId": self.model_asset_id,
            "artifactPath": self.artifact_path,
            "backendType": self.backend_type,
            "sourceTaskType": self.source_task_type,
            "fileExists": self.file_exists,
            "fileSizeBytes": self.file_size_bytes,
            "status": self.status,
        }


def normalize_backend_type(value: Optional[str]) -> str:
    from app.services.training_backend_canonical import canonicalize_training_backend

    return canonicalize_training_backend(value)


def resolve_model_asset_backend_type(asset: dict[str, Any]) -> str:
    from app.services.training_backend_canonical import resolve_asset_training_backend

    return resolve_asset_training_backend(asset)


def resolve_source_task_type(asset: dict[str, Any]) -> str:
    task_type = str(asset.get("taskType") or "").strip()
    if task_type:
        return task_type
    template_id = str(asset.get("taskTemplateId") or "").strip()
    for key, value in TASK_TYPE_TO_TEMPLATE.items():
        if value == template_id:
            return key
    if template_id == "cable_threading_single_arm":
        return "cable_threading"
    if template_id == "isaac_block_stacking":
        return "block_stacking"
    return ""


def resolve_checkpoint_path_on_disk(path_str: Optional[str]) -> Optional[Path]:
    text = str(path_str or "").strip()
    if not text:
        return None
    if text.startswith("file://"):
        text = text[len("file://") :]
    if text.startswith("minio://") or text.startswith("s3://"):
        return None
    path = Path(text)
    if not path.is_absolute():
        if path.parts and path.parts[0] == "runs":
            path = resolve_runtime_reference(text)
        else:
            path = PROJECT_ROOT / path
    return path


def check_checkpoint_file(path_str: Optional[str]) -> tuple[bool, int]:
    path = resolve_checkpoint_path_on_disk(path_str)
    if path is None:
        return False, 0
    try:
        if path.is_file():
            size = path.stat().st_size
            return size > 0, int(size)
    except OSError:
        pass
    return False, 0


def _training_job_exists(train_job_id: Optional[str]) -> bool:
    candidate = str(train_job_id or "").strip()
    if not candidate:
        return False
    try:
        from app.core.database import SessionLocal
        from app.models.workspace_job import WorkspaceJob

        with SessionLocal() as db:
            row = (
                db.query(WorkspaceJob)
                .filter(
                    WorkspaceJob.job_id == candidate,
                    WorkspaceJob.job_type == "training",
                    WorkspaceJob.status != "deleted",
                )
                .one_or_none()
            )
            return row is not None
    except Exception:
        return False


def resolve_effective_asset_status(
    *,
    db_status: str,
    file_exists: bool,
    train_job_exists: bool,
    is_placeholder: bool = False,
) -> str:
    status = str(db_status or "").strip().lower()
    if status == "deleted":
        return "deleted"
    if is_placeholder:
        return "invalid"
    if not train_job_exists:
        return "invalid"
    if not file_exists:
        if status == "generating":
            return "generating"
        return "missing"
    if status in {"ready", "available", "active", "completed"}:
        return "available"
    if status == "generating":
        return "generating"
    if status in {"superseded"}:
        return "superseded"
    return "invalid"


def is_model_asset_compatible_with_evaluation(
    asset: dict[str, Any],
    *,
    evaluation_task_type: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    task_type = str(evaluation_task_type or "").strip()
    if not task_type:
        return True, None
    if task_type == "isaac_block_stacking":
        task_type = "block_stacking"

    allowed = EVALUATION_MODEL_BACKEND_COMPATIBILITY.get(task_type)
    if not allowed:
        return True, None

    backend_type = resolve_model_asset_backend_type(asset)
    if not backend_type:
        return False, "模型资产缺少 backendType"

    source_task_type = resolve_source_task_type(asset)
    if task_type == "cable_threading" and source_task_type in {
        "dual_arm_cable_manipulation",
        "block_stacking",
        "isaac_block_stacking",
    }:
        return False, f"模型资产任务类型 {source_task_type} 与线缆穿杆评测不兼容"
    if task_type == "dual_arm_cable_manipulation" and source_task_type == "cable_threading":
        return False, "模型资产任务类型与线缆整理评测不兼容"
    if task_type == "block_stacking" and source_task_type not in {
        "block_stacking",
        "isaac_block_stacking",
        "",
    }:
        if source_task_type in {"cable_threading", "dual_arm_cable_manipulation"}:
            return False, "模型资产任务类型与物块堆叠评测不兼容"

    if backend_type not in allowed:
        expected = ", ".join(sorted(allowed))
        return False, f"backendType={backend_type} 与评测任务 {task_type} 不兼容，期望 {expected}"

    return True, None


def enrich_model_asset(asset: dict[str, Any], *, for_list: bool = False) -> dict[str, Any]:
    backend_type = resolve_model_asset_backend_type(asset)
    source_task_type = resolve_source_task_type(asset)
    checkpoint_path = str(asset.get("checkpointPath") or asset.get("artifactPath") or "").strip()
    artifact_kind = str(asset.get("checkpointKind") or asset.get("artifactKind") or asset.get("asset_type") or "")

    if for_list:
        db_status = str(asset.get("status") or "")
        enriched = dict(asset)
        enriched.update(
            {
                "artifactPath": checkpoint_path,
                "checkpointPath": checkpoint_path,
                "localCheckpointPath": checkpoint_path,
                "fileName": Path(checkpoint_path).name if checkpoint_path else "",
                "fileExists": asset.get("fileExists", db_status in {"available", "ready"}),
                "fileSizeBytes": int(asset.get("fileSizeBytes") or 0),
                "status": db_status,
                "backendType": backend_type or asset.get("backendType"),
                "sourceTaskType": source_task_type or None,
                "artifactKind": artifact_kind or None,
                "associatedTask": enriched.get("datasetDisplayName") or enriched.get("name"),
                "canEvaluate": db_status in {"available", "ready"},
            }
        )
        return enriched

    from app.services.model_asset_checkpoint_resolver import resolve_local_checkpoint_path

    local_path = resolve_local_checkpoint_path(
        asset=asset,
        path_hint=checkpoint_path,
        model_asset_id=str(asset.get("id") or asset.get("modelAssetId") or ""),
    )
    if local_path:
        checkpoint_path = local_path
    file_exists, file_size = check_checkpoint_file(checkpoint_path)
    train_job_id = str(asset.get("sourceTrainingJobId") or "").strip()
    train_job_exists = _training_job_exists(train_job_id) if train_job_id else False
    db_status = str(asset.get("status") or "")
    is_placeholder = bool(asset.get("isPlaceholder"))
    effective_status = resolve_effective_asset_status(
        db_status=db_status,
        file_exists=file_exists,
        train_job_exists=train_job_exists,
        is_placeholder=is_placeholder,
    )
    backend_type = resolve_model_asset_backend_type(asset)
    source_task_type = resolve_source_task_type(asset)
    artifact_kind = str(asset.get("checkpointKind") or asset.get("artifactKind") or asset.get("asset_type") or "")

    enriched = dict(asset)
    enriched.update(
        {
            "artifactPath": checkpoint_path,
            "checkpointPath": checkpoint_path,
            "localCheckpointPath": local_path or checkpoint_path,
            "fileName": Path(checkpoint_path).name if checkpoint_path else "",
            "fileExists": file_exists,
            "fileSizeBytes": file_size,
            "status": effective_status,
            "backendType": backend_type or asset.get("backendType"),
            "sourceTaskType": source_task_type or None,
            "artifactKind": artifact_kind or None,
            "associatedTask": enriched.get("datasetDisplayName") or enriched.get("name"),
            "canEvaluate": effective_status == "available" and file_exists,
        }
    )
    if backend_type == "diffusion_policy" or resolve_model_asset_backend_type(enriched) == "diffusion_policy":
        from app.services.dp_init_weight_compat import enrich_asset_dp_init_schema

        try:
            enriched = enrich_asset_dp_init_schema(enriched)
        except Exception as exc:
            logger.warning(
                "DP schema enrichment skipped for model asset %s: %s",
                enriched.get("id") or enriched.get("modelAssetId"),
                exc,
            )
    model_type = str(enriched.get("modelType") or enriched.get("policyType") or "").lower()
    if backend_type == "pi0" or model_type in {"pi0", "openpi"}:
        from app.services.policy_schema_resolver import resolve_pi0_model_asset_eval_fields

        try:
            enriched.update(
                resolve_pi0_model_asset_eval_fields(enriched, checkpoint_path=checkpoint_path)
            )
        except Exception as exc:
            logger.warning(
                "pi0 eval field enrichment skipped for model asset %s: %s",
                enriched.get("id") or enriched.get("modelAssetId"),
                exc,
            )
    return enriched


def is_evaluable_model_asset(asset: dict[str, Any]) -> bool:
    enriched = enrich_model_asset(asset) if "fileExists" not in asset else asset
    return bool(enriched.get("fileExists")) and str(enriched.get("status") or "") == "available"


def validate_model_asset(
    model_asset_id: str,
    *,
    evaluation_task_type: Optional[str] = None,
    require_file: bool = True,
) -> ModelAssetValidationResult:
    from app.services.workspace_model_asset_service import get_model_asset_by_id

    candidate = str(model_asset_id or "").strip()
    if not candidate:
        return ModelAssetValidationResult(
            ok=False,
            reason="modelAssetId 为空",
            model_asset_id="",
            artifact_path="",
            backend_type="",
            source_task_type="",
            file_exists=False,
            file_size_bytes=0,
            status="invalid",
        )

    asset = get_model_asset_by_id(candidate)
    if not asset:
        return ModelAssetValidationResult(
            ok=False,
            reason="模型资产记录不存在",
            model_asset_id=candidate,
            artifact_path="",
            backend_type="",
            source_task_type="",
            file_exists=False,
            file_size_bytes=0,
            status="missing",
        )

    enriched = enrich_model_asset(asset)
    artifact_path = str(enriched.get("checkpointPath") or enriched.get("artifactPath") or "")
    file_exists = bool(enriched.get("fileExists"))
    file_size = int(enriched.get("fileSizeBytes") or 0)
    status = str(enriched.get("status") or "")
    backend_type = str(enriched.get("backendType") or "")
    source_task_type = str(enriched.get("sourceTaskType") or "")

    if status == "deleted":
        return ModelAssetValidationResult(
            ok=False,
            reason="模型资产已删除",
            model_asset_id=candidate,
            artifact_path=artifact_path,
            backend_type=backend_type,
            source_task_type=source_task_type,
            file_exists=file_exists,
            file_size_bytes=file_size,
            status=status,
        )

    if require_file and not file_exists:
        reason = "模型 checkpoint 文件不存在或大小为 0"
        if artifact_path.startswith("minio://") or artifact_path.startswith("s3://"):
            reason = "模型文件仅存在于远程存储，本地不可用"
        return ModelAssetValidationResult(
            ok=False,
            reason=reason,
            model_asset_id=candidate,
            artifact_path=artifact_path,
            backend_type=backend_type,
            source_task_type=source_task_type,
            file_exists=False,
            file_size_bytes=0,
            status="missing",
        )

    if status not in {"available"}:
        return ModelAssetValidationResult(
            ok=False,
            reason=f"模型资产状态不可用: {status}",
            model_asset_id=candidate,
            artifact_path=artifact_path,
            backend_type=backend_type,
            source_task_type=source_task_type,
            file_exists=file_exists,
            file_size_bytes=file_size,
            status=status,
        )

    if evaluation_task_type:
        compatible, compat_reason = is_model_asset_compatible_with_evaluation(
            enriched,
            evaluation_task_type=evaluation_task_type,
        )
        if not compatible:
            return ModelAssetValidationResult(
                ok=False,
                reason=compat_reason or "模型资产与评测任务不兼容",
                model_asset_id=candidate,
                artifact_path=artifact_path,
                backend_type=backend_type,
                source_task_type=source_task_type,
                file_exists=file_exists,
                file_size_bytes=file_size,
                status=status,
            )

    backend_lower = backend_type.lower()
    model_type_lower = str(enriched.get("modelType") or enriched.get("policyType") or "").lower()
    if backend_lower == "pi0" or model_type_lower in {"pi0", "openpi"}:
        from app.services.policy_schema_resolver import pi0_eval_creation_allowed

        allowed, pi0_reason = pi0_eval_creation_allowed(enriched, checkpoint_path=artifact_path)
        if not allowed:
            return ModelAssetValidationResult(
                ok=False,
                reason=pi0_reason or "pi0 eval adapter not ready",
                model_asset_id=candidate,
                artifact_path=artifact_path,
                backend_type=backend_type,
                source_task_type=source_task_type,
                file_exists=file_exists,
                file_size_bytes=file_size,
                status=status,
            )

    return ModelAssetValidationResult(
        ok=True,
        reason="",
        model_asset_id=candidate,
        artifact_path=artifact_path,
        backend_type=backend_type,
        source_task_type=source_task_type,
        file_exists=file_exists,
        file_size_bytes=file_size,
        status=status,
    )


def filter_evaluable_model_assets(
    assets: list[dict[str, Any]],
    *,
    evaluation_task_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for asset in assets:
        enriched = enrich_model_asset(asset)
        if not is_evaluable_model_asset(enriched):
            continue
        if evaluation_task_type:
            compatible, _ = is_model_asset_compatible_with_evaluation(
                enriched,
                evaluation_task_type=evaluation_task_type,
            )
            if not compatible:
                continue
        result.append(enriched)
    return result
