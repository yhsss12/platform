"""Import external pretrained checkpoints into platform model assets."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.platform_paths import platform_paths
from app.models.workspace_index import ModelAsset as ModelAssetRow
from app.models.workspace_job import WorkspaceJob
from app.services.adapter_layer.adapter_service import resolve_manifest_by_dataset_id
from app.services.training_dataset_compat import extract_dataset_structure_signature

PROJECT_ROOT = platform_paths.project_root
IMPORTED_ROOT = platform_paths.models / "imported"
IMPORT_HUB_JOB_ID = "model_asset_import_hub"
MAX_CHECKPOINT_BYTES = 500 * 1024 * 1024
ALLOWED_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
ALLOWED_METADATA_SUFFIXES = {".json", ".yaml", ".yml"}

MODEL_TYPE_SPECS: dict[str, dict[str, str]] = {
    "diffusion_policy": {
        "backend": "diffusion_policy",
        "framework": "diffusion_policy",
        "label": "Diffusion Policy",
        "model_type": "Diffusion Policy",
    },
    "robomimic_bc": {
        "backend": "robomimic_bc",
        "framework": "robomimic_bc",
        "label": "Robomimic BC",
        "model_type": "Robomimic BC",
    },
    "act": {
        "backend": "act",
        "framework": "act",
        "label": "ACT",
        "model_type": "ACT",
    },
}

TASK_TYPE_OPTIONS: dict[str, str] = {
    "cable_threading": "线缆穿杆",
    "dual_arm_cable_manipulation": "双臂线缆协作",
    "isaac_block_stacking": "物块堆叠",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_model_asset_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"model_import_{stamp}_{suffix}"


def _resolved_low_dim_dim(signature: dict[str, Any]) -> int:
    keys = signature.get("lowDimKeys") or ()
    if not keys:
        return 9
    return sum(3 if "quat" in str(key) else 1 for key in keys)


def _build_dp_config_from_signature(signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_dim": signature.get("actionDim"),
        "image_keys": list(signature.get("imageKeys") or ()),
        "low_dim_keys": list(signature.get("lowDimKeys") or ()),
        "image_size": signature.get("imageSize"),
        "low_dim_dim": _resolved_low_dim_dim(signature),
        "vision_encoder": "resnet18",
    }


def _infer_checkpoint_backend(payload: dict[str, Any]) -> str:
    train_cfg = payload.get("train_config") if isinstance(payload.get("train_config"), dict) else {}
    backend = str(train_cfg.get("backend") or payload.get("backend") or "").strip().lower()
    if backend:
        return backend
    if "normalizer" in payload and "shape_meta" in payload:
        return "diffusion_policy"
    if isinstance(payload.get("model"), dict) or "algo_name" in payload:
        return "robomimic_bc"
    return ""


def _validate_checkpoint_payload(
    *,
    payload: dict[str, Any],
    model_type: str,
    dataset_signature: dict[str, Any],
    dataset_manifest: dict[str, Any],
    task_type: str,
) -> dict[str, Any]:
    if "state_dict" not in payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="checkpoint 缺少 state_dict，无法导入")

    spec = MODEL_TYPE_SPECS.get(model_type)
    if spec is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的模型类型: {model_type}")

    manifest_task = str(dataset_manifest.get("taskType") or "").strip()
    if manifest_task and task_type and manifest_task != task_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"适用任务与参考数据集不一致（数据集={manifest_task!r}, 选择={task_type!r}）",
        )

    ckpt_backend = _infer_checkpoint_backend(payload)
    if ckpt_backend and ckpt_backend != spec["backend"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"模型类型与 checkpoint 结构不匹配（期望 {spec['label']}，checkpoint backend={ckpt_backend!r}）",
        )

    validation: dict[str, Any] = {
        "hasStateDict": True,
        "hasNormalizer": "normalizer" in payload,
        "hasTrainConfig": isinstance(payload.get("train_config"), dict),
        "checkpointBackend": ckpt_backend or None,
        "matchedModelType": spec["label"],
    }

    if model_type == "diffusion_policy":
        if "normalizer" not in payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Diffusion Policy checkpoint 必须包含 normalizer",
            )
        train_cfg = payload.get("train_config")
        if not isinstance(train_cfg, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Diffusion Policy checkpoint 必须包含 train_config 或可解析配置",
            )
        dp_config = _build_dp_config_from_signature(dataset_signature)
        ckpt_cfg = train_cfg
        checks = [
            ("action_dim", ckpt_cfg.get("action_dim"), dp_config.get("action_dim")),
            ("image_keys", tuple(sorted(str(k) for k in (ckpt_cfg.get("image_keys") or []))), tuple(dp_config.get("image_keys") or ())),
            ("low_dim_keys", tuple(sorted(str(k) for k in (ckpt_cfg.get("low_dim_keys") or []))), tuple(dp_config.get("low_dim_keys") or ())),
            ("image_size", ckpt_cfg.get("image_size"), dp_config.get("image_size")),
            ("low_dim_dim", ckpt_cfg.get("low_dim_dim") or _resolved_low_dim_dim(ckpt_cfg), dp_config.get("low_dim_dim")),
        ]
        for field, left, right in checks:
            if left is None or right is None:
                continue
            if left != right:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"checkpoint 与参考数据集结构不匹配：{field} 不一致（checkpoint={left!r}, 数据集={right!r}）",
                )
        validation["structureMatched"] = True

    ckpt_action_dim = None
    train_cfg = payload.get("train_config") if isinstance(payload.get("train_config"), dict) else {}
    if train_cfg.get("action_dim") is not None:
        ckpt_action_dim = int(train_cfg["action_dim"])
    shape_meta = payload.get("shape_meta") if isinstance(payload.get("shape_meta"), dict) else {}
    if ckpt_action_dim is None and shape_meta.get("action_dim") is not None:
        ckpt_action_dim = int(shape_meta["action_dim"])
    target_action_dim = dataset_signature.get("actionDim")
    if ckpt_action_dim is not None and target_action_dim is not None and ckpt_action_dim != target_action_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"checkpoint action_dim={ckpt_action_dim} 与参考数据集 action_dim={target_action_dim} 不一致",
        )

    validation["actionDim"] = ckpt_action_dim or target_action_dim
    return validation


def _ensure_import_hub_job(db: Session) -> None:
    row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == IMPORT_HUB_JOB_ID).one_or_none()
    if row is not None:
        return
    runtime_path = str((platform_paths.runs_root / "model_assets").resolve())
    db.add(
        WorkspaceJob(
            job_id=IMPORT_HUB_JOB_ID,
            job_type="model_asset_import",
            task_type="platform",
            task_name="外部导入模型资产",
            status="completed",
            source="real",
            runner="model_asset_import",
            runtime_path=runtime_path,
            metadata_json={"kind": "imported_model_asset_hub"},
        )
    )
    db.commit()


def _build_structure_config(signature: dict[str, Any], dp_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = dp_config or _build_dp_config_from_signature(signature)
    return {
        "taskType": signature.get("taskType"),
        "input": {
            "image_keys": cfg.get("image_keys"),
            "low_dim_keys": cfg.get("low_dim_keys"),
            "image_size": cfg.get("image_size"),
        },
        "output": {
            "action_dim": cfg.get("action_dim"),
        },
    }


async def import_pretrained_model_asset(
    *,
    model_name: str,
    model_type: str,
    task_type: str,
    dataset_id: str,
    checkpoint_file: UploadFile,
    metadata_file: Optional[UploadFile] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    name = (model_name or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="模型名称不能为空")

    model_type_key = (model_type or "").strip().lower()
    if model_type_key not in MODEL_TYPE_SPECS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择有效的模型类型")

    task_key = (task_type or "").strip()
    dataset_key = (dataset_id or "").strip()
    if not dataset_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择参考数据集")

    dataset_manifest = resolve_manifest_by_dataset_id(dataset_key)
    if dataset_manifest is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"参考数据集不存在: {dataset_key}")

    from app.services import training_service as ts

    hdf5_path = ts._resolve_hdf5_path(dataset_manifest)
    dataset_signature = extract_dataset_structure_signature(dataset_manifest, hdf5_path)
    if task_key and dataset_signature.get("taskType") and dataset_signature["taskType"] != task_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="适用任务与参考数据集 taskType 不一致",
        )

    checkpoint_suffix = Path(checkpoint_file.filename or "").suffix.lower()
    if checkpoint_suffix not in ALLOWED_CHECKPOINT_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="checkpoint 文件仅支持 .pt / .pth / .ckpt",
        )

    checkpoint_bytes = await checkpoint_file.read()
    if not checkpoint_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="checkpoint 文件为空")
    if len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="checkpoint 文件过大（上限 500MB）")

    metadata_bytes: Optional[bytes] = None
    metadata_name: Optional[str] = None
    if metadata_file is not None and metadata_file.filename:
        metadata_suffix = Path(metadata_file.filename).suffix.lower()
        if metadata_suffix not in ALLOWED_METADATA_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="metadata 文件仅支持 .json / .yaml / .yml",
            )
        metadata_bytes = await metadata_file.read()
        metadata_name = metadata_file.filename

    try:
        payload = torch.load(__import__("io").BytesIO(checkpoint_bytes), map_location="cpu")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"checkpoint 无法读取: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="checkpoint 格式无效，应为 dict")

    validation = _validate_checkpoint_payload(
        payload=payload,
        model_type=model_type_key,
        dataset_signature=dataset_signature,
        dataset_manifest=dataset_manifest,
        task_type=task_key,
    )

    spec = MODEL_TYPE_SPECS[model_type_key]
    dp_config = _build_dp_config_from_signature(dataset_signature) if model_type_key == "diffusion_policy" else None
    structure_config = _build_structure_config(dataset_signature, dp_config)

    model_asset_id = _make_model_asset_id()
    asset_dir = IMPORTED_ROOT / model_asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = asset_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    manifest_path = asset_dir / "manifest.json"
    validation_path = asset_dir / "validation_report.json"

    if metadata_bytes:
        meta_path = asset_dir / (metadata_name or "metadata.json")
        meta_path.write_bytes(metadata_bytes)

    created_at = _utc_now_iso()
    manifest_doc = {
        "modelAssetId": model_asset_id,
        "name": name,
        "displayName": name,
        "assetSource": "imported",
        "importMetadata": {
            "note": note or "",
            "referenceDatasetId": dataset_key,
            "referenceDatasetName": dataset_manifest.get("datasetName"),
            "taskType": task_key or dataset_signature.get("taskType"),
            "taskLabel": TASK_TYPE_OPTIONS.get(task_key or "", task_key or ""),
            "importedAt": created_at,
            "checkpointFilename": checkpoint_file.filename,
            "metadataFilename": metadata_name,
        },
        "framework": spec["framework"],
        "trainingBackend": spec["backend"],
        "modelType": spec["model_type"],
        "taskType": dataset_signature.get("taskType") or task_key,
        "taskTemplateId": dataset_manifest.get("taskTemplateId") or dataset_signature.get("taskTemplateId"),
        "sourceDatasetId": dataset_key,
        "datasetDisplayName": dataset_manifest.get("datasetName"),
        "checkpointPath": str(checkpoint_path),
        "manifestPath": str(manifest_path),
        "structureConfig": structure_config,
        "resolvedModelParams": {
            "action_dim": dataset_signature.get("actionDim"),
            "image_keys": list(dataset_signature.get("imageKeys") or ()),
            "low_dim_keys": list(dataset_signature.get("lowDimKeys") or ()),
            "image_size": dataset_signature.get("imageSize"),
            "low_dim_dim": _resolved_low_dim_dim(dataset_signature),
        },
        "validationResult": validation,
        "createdAt": created_at,
    }
    _write_json(manifest_path, manifest_doc)
    _write_json(validation_path, {"validationResult": validation, "datasetSignature": dataset_signature})

    with SessionLocal() as db:
        _ensure_import_hub_job(db)
        row = ModelAssetRow(
            model_asset_id=model_asset_id,
            train_job_id=IMPORT_HUB_JOB_ID,
            dataset_id=dataset_key,
            model_name=name,
            model_type=spec["model_type"],
            asset_type="imported",
            epoch=None,
            storage_uri=f"file://{checkpoint_path}",
            manifest_json=manifest_doc,
            metrics_json=None,
            sha256=_sha256_bytes(checkpoint_bytes),
            size_bytes=len(checkpoint_bytes),
            status="available",
        )
        db.add(row)
        db.commit()

    return {
        "modelAssetId": model_asset_id,
        "modelName": name,
        "modelType": spec["model_type"],
        "taskName": TASK_TYPE_OPTIONS.get(task_key, task_key) or dataset_manifest.get("taskName"),
        "datasetName": dataset_manifest.get("datasetName"),
        "structureConfig": structure_config,
        "checkpointPath": str(checkpoint_path),
        "createdAt": created_at,
        "validationResult": validation,
        "assetSource": "imported",
    }
