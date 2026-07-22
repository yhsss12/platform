"""模型类型定义 CRUD、默认 seed 与结构校验。"""

from __future__ import annotations

import logging
import re
import secrets
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.model_type_definition import ModelTypeDefinition

logger = logging.getLogger(__name__)

MODEL_TYPE_TABLE_MISSING_MESSAGE = (
    "数据库迁移 022_model_type_definitions 未执行，请在 backend 目录运行: alembic upgrade head"
)

BASE_ALGORITHM_ADAPTER_MAP: dict[str, str] = {
    "robomimic_bc": "robomimic_bc_adapter",
    "act": "act_adapter",
    "diffusion_policy": "diffusion_policy_adapter",
    "pi0": "pi0_adapter",
}

ADAPTER_BACKEND_MAP: dict[str, str] = {
    "robomimic_bc_adapter": "robomimic_bc",
    "act_adapter": "act",
    "diffusion_policy_adapter": "diffusion_policy",
    "pi0_adapter": "pi0",
}

ADAPTER_DOWNSTREAM_MAP: dict[str, str] = {
    "robomimic_bc_adapter": "Robomimic",
    "act_adapter": "ACT",
    "diffusion_policy_adapter": "Diffusion Policy",
    "pi0_adapter": "pi0",
}

PI0_RUNNER_UNAVAILABLE_MESSAGE = "openpi 环境未配置，无法训练 pi0（请设置 PI0_RUNNER_ENABLED、OPENPI_ROOT、OPENPI_PYTHON）"
PI0_PROBE_PENDING_MESSAGE = "正在检测 runner"

READINESS_TTL_SEC = 120.0
_READINESS_PROBE_LOCK = threading.Lock()
_READINESS_PROBE_IN_PROGRESS = False

LEGACY_MODEL_TYPE_ID_MAP: dict[str, str] = {
    "robomimic": "robomimic-bc",
    "robomimic_bc": "robomimic-bc",
    "Robomimic": "robomimic-bc",
    "Robomimic BC": "robomimic-bc",
    "act": "act",
    "ACT": "act",
    "diffusion_policy": "diffusion-policy",
    "Diffusion Policy": "diffusion-policy",
    "pi0": "pi0",
}

ROBOMIMIC_BC_STRUCTURE_DEFAULTS: dict[str, Any] = {
    "actor_hidden_dims": "512,512",
    "l2_regularization": 0.0,
    "encoder_type": "low_dim",
    "activation": "relu",
}

ACT_STRUCTURE_DEFAULTS: dict[str, Any] = {
    "hidden_dim": 512,
    "dim_feedforward": 2048,
    "chunk_size": 100,
    "n_action_steps": 100,
    "kl_weight": 10.0,
    "latent_dim": 32,
    "enc_layers": 4,
    "dec_layers": 4,
    "nheads": 8,
    "dropout": 0.1,
}

DIFFUSION_POLICY_STRUCTURE_DEFAULTS: dict[str, Any] = {
    "horizon": 16,
    "n_obs_steps": 2,
    "n_action_steps": 8,
    "num_inference_steps": 20,
    "weight_decay": 1e-4,
    "vision_encoder": "resnet18",
    "noise_scheduler": "ddpm",
}

PI0_STRUCTURE_DEFAULTS: dict[str, Any] = {
    "context_window": 256,
    "action_horizon": 16,
    "vision_encoder": "siglip",
    "language_conditioning": True,
    "action_head": "flow_matching",
    "tokenizer_or_processor": "default",
}

TRAINING_DEFAULTS_TEMPLATE: dict[str, Any] = {
    "default_epochs": 5,
    "default_batch_size": 16,
    "default_learning_rate": 0.0001,
    "default_seed_strategy": "random",
}

DEFAULT_MODEL_TYPES: list[dict[str, Any]] = [
    {
        "model_type_id": "robomimic-bc",
        "name": "Robomimic BC",
        "base_algorithm": "robomimic_bc",
        "adapter_key": "robomimic_bc_adapter",
        "simulator": "mujoco",
        "robot_type": "panda",
        "tags": ["BC", "Robomimic", "内置"],
        "description": "单臂低维行为克隆（MuJoCo / Robomimic HDF5），对应线缆穿杆等任务默认 BC 模型。",
        "structure_config": dict(ROBOMIMIC_BC_STRUCTURE_DEFAULTS),
        "training_defaults": dict(TRAINING_DEFAULTS_TEMPLATE),
        "status": "available",
        "is_builtin": True,
    },
    {
        "model_type_id": "act",
        "name": "ACT",
        "base_algorithm": "act",
        "adapter_key": "act_adapter",
        "simulator": "general",
        "robot_type": "general",
        "tags": ["ACT", "模仿学习", "内置"],
        "description": "Action Chunking Transformer，适用于图像 + proprio 观测。",
        "structure_config": dict(ACT_STRUCTURE_DEFAULTS),
        "training_defaults": {
            **TRAINING_DEFAULTS_TEMPLATE,
            "default_batch_size": 8,
        },
        "status": "available",
        "is_builtin": True,
    },
    {
        "model_type_id": "diffusion-policy",
        "name": "Diffusion Policy",
        "base_algorithm": "diffusion_policy",
        "adapter_key": "diffusion_policy_adapter",
        "simulator": "general",
        "robot_type": "general",
        "tags": ["Diffusion Policy", "扩散模型", "内置"],
        "description": "扩散策略训练，支持低维与图像观测。",
        "structure_config": dict(DIFFUSION_POLICY_STRUCTURE_DEFAULTS),
        "training_defaults": {
            **TRAINING_DEFAULTS_TEMPLATE,
            "default_batch_size": 8,
        },
        "status": "available",
        "is_builtin": True,
    },
    {
        "model_type_id": "pi0",
        "name": "pi0",
        "base_algorithm": "pi0",
        "adapter_key": "pi0_adapter",
        "simulator": "general",
        "robot_type": "general",
        "tags": ["pi0", "VLA", "generalist", "robot-policy"],
        "description": "通用视觉-语言-动作机器人策略模型类型，可通过标准适配层适配不同数据集与任务。",
        "structure_config": dict(PI0_STRUCTURE_DEFAULTS),
        "training_defaults": {
            **TRAINING_DEFAULTS_TEMPLATE,
            "default_batch_size": 8,
        },
        "status": "available",
        "is_builtin": True,
    },
]

REQUIRED_STRUCTURE_FIELDS: dict[str, list[str]] = {
    "robomimic_bc": ["actor_hidden_dims"],
    "act": ["hidden_dim", "chunk_size", "n_action_steps"],
    "diffusion_policy": ["horizon", "n_obs_steps", "n_action_steps", "num_inference_steps"],
    "pi0": ["context_window", "action_horizon", "vision_encoder"],
}

_MODEL_TYPE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,126}$")


def _iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def resolve_training_readiness_for_adapter(
    adapter_key: str,
    *,
    supported_backends: Optional[set[str]] = None,
    pi0_capability: Optional[dict[str, Any]] = None,
) -> tuple[bool, Optional[str]]:
    """根据 probe 结果判断模型类型是否可创建训练任务（仅用于 probe 持久化与训练任务创建）。"""
    backend = ADAPTER_BACKEND_MAP.get(adapter_key)
    if not backend:
        return False, f"适配器 {adapter_key} 未注册训练 backend"

    if supported_backends is None:
        from app.services.training_service import probe_training_capabilities

        capabilities = probe_training_capabilities()
        supported_backends = set(capabilities.get("supportedTrainingBackends") or [])
        if pi0_capability is None:
            pi0_capability = capabilities.get("pi0Capability")

    if backend in supported_backends:
        return True, None

    if backend == "pi0":
        if isinstance(pi0_capability, dict) and pi0_capability.get("pending"):
            reason = pi0_capability.get("reason") or PI0_PROBE_PENDING_MESSAGE
            return False, str(reason)
        reason = (pi0_capability or {}).get("reason") if isinstance(pi0_capability, dict) else None
        if isinstance(reason, str) and reason.strip():
            return False, reason.strip()
        return False, PI0_RUNNER_UNAVAILABLE_MESSAGE

    return False, f"当前 {backend} runner 尚未接入，无法创建训练任务"


def _readiness_status_from_probe(
    *,
    adapter_key: str,
    training_ready: bool,
    pi0_capability: Optional[dict[str, Any]] = None,
) -> str:
    if training_ready:
        return "ready"
    if adapter_key == "pi0_adapter":
        if isinstance(pi0_capability, dict) and pi0_capability.get("pending"):
            return "pending"
        return "unavailable"
    return "unavailable"


def _collect_fast_supported_backends() -> tuple[set[str], list[str]]:
    from app.services.training_service import _collect_script_supported_backends

    evidence: list[str] = []
    supported = set(_collect_script_supported_backends(evidence=evidence))
    return supported, evidence


def _initial_readiness_for_algorithm(
    base_algorithm: str,
    *,
    supported_backends: set[str],
    evidence: list[str],
) -> tuple[bool, str, Optional[str], Optional[list[str]], Optional[datetime]]:
    adapter_key = BASE_ALGORITHM_ADAPTER_MAP.get(base_algorithm, "")
    backend = ADAPTER_BACKEND_MAP.get(adapter_key, "")
    now = datetime.now(timezone.utc)
    if backend == "pi0":
        return False, "pending", PI0_PROBE_PENDING_MESSAGE, None, None
    if backend and backend in supported_backends:
        return True, "ready", None, list(evidence), now
    reason = f"当前 {backend or base_algorithm} runner 尚未接入，无法创建训练任务"
    return False, "unavailable", reason, list(evidence), now


def _apply_capabilities_to_rows(db: Session, capabilities: dict[str, Any]) -> None:
    supported = set(capabilities.get("supportedTrainingBackends") or [])
    pi0_cap = capabilities.get("pi0Capability") if isinstance(capabilities.get("pi0Capability"), dict) else {}
    evidence = capabilities.get("evidence") if isinstance(capabilities.get("evidence"), list) else []
    now = datetime.now(timezone.utc)
    rows = db.query(ModelTypeDefinition).filter(ModelTypeDefinition.status != "deleted").all()
    for row in rows:
        training_ready, disabled_reason = resolve_training_readiness_for_adapter(
            row.adapter_key,
            supported_backends=supported,
            pi0_capability=pi0_cap,
        )
        row.training_ready = training_ready
        row.training_readiness_status = _readiness_status_from_probe(
            adapter_key=row.adapter_key,
            training_ready=training_ready,
            pi0_capability=pi0_cap,
        )
        row.disabled_reason = disabled_reason
        row.capability_checked_at = now
        if row.adapter_key == "pi0_adapter":
            row.capability_evidence = list(pi0_cap.get("evidence") or [])
        else:
            row.capability_evidence = list(evidence)
    db.commit()


def refresh_model_type_readiness(*, force: bool = False) -> None:
    """Full probe (may block on pi0 subprocess) and persist readiness to database."""
    from app.services.training_service import (
        _probe_training_capabilities_uncached,
        _store_capabilities_cache,
        invalidate_training_capabilities_cache,
    )

    if force:
        invalidate_training_capabilities_cache()
    capabilities = _probe_training_capabilities_uncached()
    _store_capabilities_cache(capabilities)
    db = SessionLocal()
    try:
        ensure_default_model_types(db)
        _apply_capabilities_to_rows(db, capabilities)
    finally:
        db.close()


def _run_background_readiness_refresh(*, force: bool = False) -> None:
    global _READINESS_PROBE_IN_PROGRESS
    try:
        refresh_model_type_readiness(force=force)
    except Exception as exc:
        logger.warning("background model type readiness refresh failed: %s", exc)
    finally:
        with _READINESS_PROBE_LOCK:
            _READINESS_PROBE_IN_PROGRESS = False


def schedule_model_type_readiness_refresh(*, force: bool = False) -> None:
    """Kick off full readiness probe in a daemon thread; never blocks callers."""
    global _READINESS_PROBE_IN_PROGRESS
    with _READINESS_PROBE_LOCK:
        if _READINESS_PROBE_IN_PROGRESS:
            return
        _READINESS_PROBE_IN_PROGRESS = True
    thread = threading.Thread(
        target=_run_background_readiness_refresh,
        kwargs={"force": force},
        name="model-type-readiness-refresh",
        daemon=True,
    )
    thread.start()


def _needs_readiness_refresh(db: Session) -> bool:
    rows = (
        db.query(ModelTypeDefinition)
        .filter(ModelTypeDefinition.status != "deleted")
        .all()
    )
    if not rows:
        return False
    now = datetime.now(timezone.utc)
    for row in rows:
        if row.training_readiness_status is None or row.capability_checked_at is None:
            return True
        if row.training_readiness_status in {"pending", "unknown"}:
            return True
        checked_at = row.capability_checked_at
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        if (now - checked_at).total_seconds() > READINESS_TTL_SEC:
            return True
    return False


def _row_to_dict(row: ModelTypeDefinition) -> dict[str, Any]:
    tags = row.tags if isinstance(row.tags, list) else []
    structure = row.structure_config if isinstance(row.structure_config, dict) else {}
    defaults = row.training_defaults if isinstance(row.training_defaults, dict) else {}
    training_ready = bool(row.training_ready) if row.training_ready is not None else False
    training_readiness_status = row.training_readiness_status or "unknown"
    disabled_reason = row.disabled_reason
    return {
        "modelTypeId": row.model_type_id,
        "name": row.name,
        "baseAlgorithm": row.base_algorithm,
        "adapterKey": row.adapter_key,
        "simulator": row.simulator,
        "robotType": row.robot_type,
        "tags": tags,
        "description": row.description,
        "structureConfig": structure,
        "trainingDefaults": defaults,
        "status": row.status,
        "trainingReady": training_ready,
        "trainingReadinessStatus": training_readiness_status,
        "disabledReason": disabled_reason,
        "isBuiltin": bool(row.is_builtin),
        "createdAt": _iso(row.created_at),
        "updatedAt": _iso(row.updated_at),
    }


def _normalize_model_type_id(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:128]


def generate_model_type_id(name: str) -> str:
    base = _normalize_model_type_id(name) or "model-type"
    if not _MODEL_TYPE_ID_PATTERN.match(base):
        base = f"model-{base}"[:120]
    suffix = secrets.token_hex(2)
    candidate = f"{base}-{suffix}"[:128]
    return candidate


def validate_structure_config(base_algorithm: str, structure_config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    algo = (base_algorithm or "").strip()
    if algo not in BASE_ALGORITHM_ADAPTER_MAP:
        errors.append(f"不支持的基础算法: {base_algorithm}")
        return errors

    config = dict(structure_config or {})
    required = REQUIRED_STRUCTURE_FIELDS.get(algo, [])
    for key in required:
        value = config.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"缺少必要结构参数: {key}")

    if algo == "robomimic_bc":
        dims = str(config.get("actor_hidden_dims") or config.get("hidden_dims") or "").strip()
        if dims and not re.match(r"^[\d,\s]+$", dims):
            errors.append("actor_hidden_dims 格式无效，应为逗号分隔整数")

    if algo == "act":
        for key in ("hidden_dim", "chunk_size", "n_action_steps", "enc_layers", "dec_layers", "nheads"):
            value = config.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                errors.append(f"{key} 必须为正数")

    if algo == "diffusion_policy":
        for key in ("horizon", "n_obs_steps", "n_action_steps", "num_inference_steps"):
            value = config.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                errors.append(f"{key} 必须为正整数")

    if algo == "pi0":
        for key in ("context_window", "action_horizon"):
            value = config.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                errors.append(f"{key} 必须为正整数")
        if not str(config.get("vision_encoder") or "").strip():
            errors.append("vision_encoder 不能为空")

    return errors


def merge_structure_defaults(base_algorithm: str, structure_config: dict[str, Any]) -> dict[str, Any]:
    algo = (base_algorithm or "").strip()
    defaults: dict[str, Any]
    if algo == "robomimic_bc":
        defaults = dict(ROBOMIMIC_BC_STRUCTURE_DEFAULTS)
    elif algo == "act":
        defaults = dict(ACT_STRUCTURE_DEFAULTS)
    elif algo == "diffusion_policy":
        defaults = dict(DIFFUSION_POLICY_STRUCTURE_DEFAULTS)
    elif algo == "pi0":
        defaults = dict(PI0_STRUCTURE_DEFAULTS)
    else:
        defaults = {}
    merged = {**defaults, **(structure_config or {})}
    return merged


def _assert_model_type_table(db: Session) -> None:
    bind = db.get_bind()
    table_names = set(inspect(bind).get_table_names())
    if "model_type_definitions" not in table_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_TYPE_TABLE_MISSING_MESSAGE,
        )


def ensure_default_model_types(db: Session) -> None:
    _assert_model_type_table(db)
    supported_backends, evidence = _collect_fast_supported_backends()
    for item in DEFAULT_MODEL_TYPES:
        existing = (
            db.query(ModelTypeDefinition)
            .filter(ModelTypeDefinition.model_type_id == item["model_type_id"])
            .first()
        )
        if existing:
            continue
        ready, status, reason, row_evidence, checked_at = _initial_readiness_for_algorithm(
            item["base_algorithm"],
            supported_backends=supported_backends,
            evidence=evidence,
        )
        db.add(
            ModelTypeDefinition(
                model_type_id=item["model_type_id"],
                name=item["name"],
                base_algorithm=item["base_algorithm"],
                adapter_key=item["adapter_key"],
                simulator=item.get("simulator"),
                robot_type=item.get("robot_type"),
                tags=item.get("tags") or [],
                description=item.get("description"),
                structure_config=item.get("structure_config") or {},
                training_defaults=item.get("training_defaults") or {},
                status=item.get("status") or "available",
                is_builtin=bool(item.get("is_builtin")),
                training_ready=ready,
                training_readiness_status=status,
                disabled_reason=reason,
                capability_checked_at=checked_at,
                capability_evidence=row_evidence,
            )
        )
    db.commit()


def list_model_types(*, status: Optional[str] = None) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        ensure_default_model_types(db)
        if _needs_readiness_refresh(db):
            schedule_model_type_readiness_refresh()
        query = db.query(ModelTypeDefinition).filter(ModelTypeDefinition.status != "deleted")
        if status:
            query = query.filter(ModelTypeDefinition.status == status)
        rows = query.order_by(ModelTypeDefinition.is_builtin.desc(), ModelTypeDefinition.updated_at.desc()).all()
        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                results.append(_row_to_dict(row))
            except Exception as exc:
                logger.warning(
                    "skip model type %s during list: %s",
                    getattr(row, "model_type_id", "?"),
                    exc,
                )
        return results
    finally:
        db.close()


def get_model_type(model_type_id: str) -> Optional[dict[str, Any]]:
    db = SessionLocal()
    try:
        ensure_default_model_types(db)
        row = (
            db.query(ModelTypeDefinition)
            .filter(
                ModelTypeDefinition.model_type_id == model_type_id,
                ModelTypeDefinition.status != "deleted",
            )
            .first()
        )
        return _row_to_dict(row) if row else None
    finally:
        db.close()


def get_available_model_type(model_type_id: str) -> dict[str, Any]:
    row = get_model_type(model_type_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"模型类型不存在: {model_type_id}")
    if row["status"] != "available":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"模型类型 {model_type_id} 当前状态为 {row['status']}，不可用于训练",
        )
    errors = validate_structure_config(row["baseAlgorithm"], row["structureConfig"])
    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=errors[0])
    return row


def resolve_legacy_model_type_id(
    *,
    downstream_model_type: Optional[str] = None,
    training_backend: Optional[str] = None,
) -> Optional[str]:
    for candidate in (downstream_model_type, training_backend):
        if not candidate:
            continue
        mapped = LEGACY_MODEL_TYPE_ID_MAP.get(candidate.strip())
        if mapped:
            return mapped
    return None


def create_model_type(payload: dict[str, Any]) -> dict[str, Any]:
    db = SessionLocal()
    try:
        ensure_default_model_types(db)
        base_algorithm = str(payload.get("baseAlgorithm") or "").strip()
        if base_algorithm not in BASE_ALGORITHM_ADAPTER_MAP:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的基础算法: {base_algorithm}")

        model_type_id = str(payload.get("modelTypeId") or "").strip()
        if not model_type_id:
            model_type_id = generate_model_type_id(str(payload.get("name") or base_algorithm))
        else:
            model_type_id = _normalize_model_type_id(model_type_id)
        if not _MODEL_TYPE_ID_PATTERN.match(model_type_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="modelTypeId 格式无效")

        existing = (
            db.query(ModelTypeDefinition)
            .filter(ModelTypeDefinition.model_type_id == model_type_id)
            .first()
        )
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"modelTypeId 已存在: {model_type_id}")

        structure_config = merge_structure_defaults(base_algorithm, payload.get("structureConfig") or {})
        errors = validate_structure_config(base_algorithm, structure_config)
        if errors:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=errors[0])

        training_defaults = {
            **TRAINING_DEFAULTS_TEMPLATE,
            **(payload.get("trainingDefaults") or {}),
        }
        status_value = str(payload.get("status") or "draft").strip()
        if status_value not in {"available", "draft", "disabled"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无效状态: {status_value}")

        supported_backends, evidence = _collect_fast_supported_backends()
        ready, readiness_status, reason, row_evidence, checked_at = _initial_readiness_for_algorithm(
            base_algorithm,
            supported_backends=supported_backends,
            evidence=evidence,
        )

        row = ModelTypeDefinition(
            model_type_id=model_type_id,
            name=str(payload.get("name") or model_type_id).strip(),
            base_algorithm=base_algorithm,
            adapter_key=BASE_ALGORITHM_ADAPTER_MAP[base_algorithm],
            simulator=payload.get("simulator"),
            robot_type=payload.get("robotType"),
            tags=payload.get("tags") or [],
            description=payload.get("description"),
            structure_config=structure_config,
            training_defaults=training_defaults,
            status=status_value,
            is_builtin=False,
            training_ready=ready,
            training_readiness_status=readiness_status,
            disabled_reason=reason,
            capability_checked_at=checked_at,
            capability_evidence=row_evidence,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _row_to_dict(row)
    finally:
        db.close()


def update_model_type(model_type_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = (
            db.query(ModelTypeDefinition)
            .filter(
                ModelTypeDefinition.model_type_id == model_type_id,
                ModelTypeDefinition.status != "deleted",
            )
            .first()
        )
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="模型类型不存在")
        if row.is_builtin and payload.get("status") == "deleted":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="内置模型类型不可删除")

        if payload.get("name") is not None:
            row.name = str(payload["name"]).strip()
        if payload.get("simulator") is not None:
            row.simulator = payload.get("simulator")
        if payload.get("robotType") is not None:
            row.robot_type = payload.get("robotType")
        if payload.get("tags") is not None:
            row.tags = payload.get("tags") or []
        if payload.get("description") is not None:
            row.description = payload.get("description")
        if payload.get("structureConfig") is not None:
            merged = merge_structure_defaults(row.base_algorithm, payload["structureConfig"])
            errors = validate_structure_config(row.base_algorithm, merged)
            if errors:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=errors[0])
            row.structure_config = merged
        if payload.get("trainingDefaults") is not None:
            row.training_defaults = {
                **TRAINING_DEFAULTS_TEMPLATE,
                **(row.training_defaults or {}),
                **(payload.get("trainingDefaults") or {}),
            }
        if payload.get("status") is not None:
            next_status = str(payload["status"]).strip()
            if next_status not in {"available", "draft", "disabled", "deleted"}:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无效状态: {next_status}")
            row.status = next_status

        db.commit()
        db.refresh(row)
        return _row_to_dict(row)
    finally:
        db.close()


def delete_model_type(model_type_id: str) -> dict[str, Any]:
    db = SessionLocal()
    try:
        row = (
            db.query(ModelTypeDefinition)
            .filter(
                ModelTypeDefinition.model_type_id == model_type_id,
                ModelTypeDefinition.status != "deleted",
            )
            .first()
        )
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="模型类型不存在")
        if row.is_builtin:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="内置模型类型不可删除，请使用禁用")
        row.status = "deleted"
        db.commit()
        return {"modelTypeId": model_type_id, "deleted": True}
    finally:
        db.close()


def validate_model_type_definition(model_type_id: str) -> dict[str, Any]:
    row = get_model_type(model_type_id)
    if not row:
        return {"valid": False, "errors": [f"模型类型不存在: {model_type_id}"]}
    errors = validate_structure_config(row["baseAlgorithm"], row["structureConfig"])
    if row["status"] != "available":
        errors.append(f"状态为 {row['status']}，不可用于训练")
    return {"valid": not errors, "errors": errors}
