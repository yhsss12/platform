from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, status

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.workspace_model_asset_service import get_model_asset_by_id

TASK_TEMPLATE_TO_TASK_TYPE: dict[str, str] = {
    "cable_threading_single_arm": "cable_threading",
    "dual_arm_cable_manipulation": "dual_arm_cable_manipulation",
    "isaac_block_stacking": "block_stacking",
    "isaaclab_franka_stack_cube": "block_stacking",
    "nut_assembly_single_arm": "nut_assembly",
}

TASK_TYPE_ALIASES: dict[str, str] = {
    "cable_threading_single_arm": "cable_threading",
    "stacking": "block_stacking",
    "isaaclab_franka_stack_cube": "block_stacking",
    "nut_assembly_single_arm": "nut_assembly",
}

TASK_TYPE_TO_TEMPLATE: dict[str, str] = {
    "cable_threading": "cable_threading_single_arm",
    "dual_arm_cable_manipulation": "dual_arm_cable_manipulation",
    "block_stacking": "isaac_block_stacking",
    "nut_assembly": "nut_assembly_single_arm",
}

PUBLIC_EVALUATION_MODES = frozenset(
    {
        "policy_evaluation",
        "episode_stability",
        "expert_policy_evaluation",
        "trained_model_evaluation",
    }
)


@dataclass(frozen=True)
class NormalizedEvaluateRequest:
    task_type: str
    task_template_id: str
    internal_evaluation_mode: str
    public_evaluation_mode: str
    policy: Optional[str] = None
    checkpoint_path: Optional[str] = None
    dataset_id: Optional[str] = None
    model_asset_id: Optional[str] = None
    eval_executor: Optional[str] = None
    controller_type: Optional[str] = None
    action_mode: Optional[str] = None
    side_channel_mode: Optional[str] = None
    robot: Optional[str] = None
    train_config_path: Optional[str] = None
    task_instruction: Optional[str] = None
    source_train_job_id: Optional[str] = None
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    model_type: Optional[str] = None
    policy_runtime: Optional[str] = None


def _resolve_task_type(request: EvaluateAsyncRequest) -> str:
    template_id = (request.taskTemplateId or "").strip()
    raw_type = (request.taskType or "").strip()

    if template_id:
        mapped = TASK_TEMPLATE_TO_TASK_TYPE.get(template_id)
        if mapped:
            return mapped
        if template_id in TASK_TYPE_ALIASES:
            return TASK_TYPE_ALIASES[template_id]

    if raw_type:
        if raw_type in TASK_TYPE_ALIASES:
            return TASK_TYPE_ALIASES[raw_type]
        if raw_type in TASK_TEMPLATE_TO_TASK_TYPE:
            return TASK_TEMPLATE_TO_TASK_TYPE[raw_type]
        if raw_type in {"cable_threading", "dual_arm_cable_manipulation", "block_stacking", "nut_assembly"}:
            return raw_type

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="taskTemplateId or taskType is required",
    )


def _resolve_task_template_id(task_type: str, request: EvaluateAsyncRequest) -> str:
    template_id = (request.taskTemplateId or "").strip()
    if template_id == "isaaclab_franka_stack_cube":
        return "isaac_block_stacking"
    if template_id and template_id in TASK_TEMPLATE_TO_TASK_TYPE:
        return template_id if template_id != "isaaclab_franka_stack_cube" else "isaac_block_stacking"
    return TASK_TYPE_TO_TEMPLATE.get(task_type, task_type)


def _resolve_checkpoint(request: EvaluateAsyncRequest, *, task_type: Optional[str] = None) -> Optional[str]:
    from app.services.model_asset_checkpoint_resolver import resolve_eval_checkpoint_path

    model_asset_id = (request.modelAssetId or "").strip() or None
    checkpoint_hint = (request.checkpointPath or request.checkpointId or "").strip() or None
    asset = get_model_asset_by_id(model_asset_id) if model_asset_id else None

    resolved_path, file_exists = resolve_eval_checkpoint_path(
        asset=asset,
        path_hint=checkpoint_hint,
        model_asset_id=model_asset_id,
    )
    if resolved_path and file_exists:
        return resolved_path

    if checkpoint_hint and not checkpoint_hint.startswith(("minio://", "s3://")):
        on_disk = resolve_eval_checkpoint_path(asset=asset, path_hint=checkpoint_hint)
        if on_disk[0] and on_disk[1]:
            return on_disk[0]

    if not model_asset_id:
        return resolved_path

    from app.services.model_asset_validation import (
        EVALUATION_MODEL_BACKEND_COMPATIBILITY,
        validate_model_asset,
    )

    validation = validate_model_asset(
        model_asset_id,
        evaluation_task_type=task_type,
        require_file=True,
    )
    if not validation.ok:
        code = "MODEL_ASSET_NOT_AVAILABLE"
        if validation.file_exists and task_type:
            from app.services.model_asset_validation import is_model_asset_compatible_with_evaluation

            asset = get_model_asset_by_id(model_asset_id) or {}
            compatible, _ = is_model_asset_compatible_with_evaluation(
                asset,
                evaluation_task_type=task_type,
            )
            if validation.status == "available" and not compatible:
                code = "MODEL_ASSET_INCOMPATIBLE"

        expected = list(EVALUATION_MODEL_BACKEND_COMPATIBILITY.get(task_type or "", [])) if task_type else None
        if task_type == "block_stacking":
            expected = list(EVALUATION_MODEL_BACKEND_COMPATIBILITY.get("block_stacking", []))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": code,
                "message": validation.reason
                or "所选模型资产不存在或模型文件已丢失，请重新选择可用模型资产。",
                "modelAssetId": model_asset_id,
                "reason": validation.reason,
                "expectedBackendTypes": expected,
                "actualBackendType": validation.backend_type or None,
            },
        )
    return validation.artifact_path


def _resolve_trained_policy_for_model_asset(
    model_asset_id: Optional[str],
    *,
    checkpoint_path: Optional[str] = None,
) -> str:
    from app.services.model_asset_checkpoint_resolver import infer_trained_policy_type

    candidate = (model_asset_id or "").strip()
    asset = get_model_asset_by_id(candidate) if candidate else None
    return infer_trained_policy_type(model_asset=asset, checkpoint_path=checkpoint_path)


def normalize_evaluate_request(request: EvaluateAsyncRequest) -> NormalizedEvaluateRequest:
    mode = str(request.evaluationMode or "").strip()
    if mode not in PUBLIC_EVALUATION_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported evaluationMode: {mode}",
        )

    task_type = _resolve_task_type(request)
    task_template_id = _resolve_task_template_id(task_type, request)
    dataset_id = (request.datasetId or "").strip() or None
    model_asset_id = (request.modelAssetId or "").strip() or None

    if task_type == "cable_threading":
        if mode == "episode_stability":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cable_threading does not support episode_stability evaluation",
            )
        eval_executor = None
        controller_type = None
        action_mode = None
        side_channel_mode = None
        robot: Optional[str] = "Panda"
        asset: dict[str, Any] = {}
        checkpoint_path: Optional[str] = None
        pi0_job_fields: dict[str, Any] = {}
        if mode in {"expert_policy_evaluation", "policy_evaluation"}:
            public_mode = "expert_policy_evaluation"
            internal_mode = "policy_evaluation"
            policy = "scripted"
        elif mode == "trained_model_evaluation":
            public_mode = "trained_model_evaluation"
            internal_mode = "policy_evaluation"
            checkpoint_path = _resolve_checkpoint(request, task_type=task_type)
            if not checkpoint_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="checkpointPath or modelAssetId required for trained_model_evaluation",
                )
            policy = _resolve_trained_policy_for_model_asset(
                model_asset_id,
                checkpoint_path=checkpoint_path,
            )
            side_channel_mode = "policy"
            asset = get_model_asset_by_id(model_asset_id) if model_asset_id else {}
            pi0_paths: dict[str, Any] = {}
            if policy == "diffusion_policy":
                from app.services.dp_schema_resolver import resolve_dp_eval_executor

                spec = resolve_dp_eval_executor(
                    policy=policy,
                    model_asset=asset or {},
                    checkpoint_path=checkpoint_path,
                )
                eval_executor = spec.eval_executor
                controller_type = spec.controller_type
                action_mode = spec.action_mode
                side_channel_mode = spec.side_channel_mode
            elif policy == "act":
                from app.services.policy_schema_resolver import resolve_act_eval_executor

                spec = resolve_act_eval_executor(
                    policy=policy,
                    model_asset=asset or {},
                    checkpoint_path=checkpoint_path,
                )
                eval_executor = spec.eval_executor
                controller_type = spec.controller_type
                action_mode = spec.action_mode
                side_channel_mode = spec.side_channel_mode
            elif policy == "pi0":
                from app.services.policy_schema_resolver import (
                    resolve_pi0_eval_executor,
                    resolve_pi0_eval_job_paths,
                )

                spec = resolve_pi0_eval_executor(
                    policy=policy,
                    model_asset=asset or {},
                    checkpoint_path=checkpoint_path,
                )
                eval_executor = spec.eval_executor
                controller_type = spec.controller_type
                action_mode = spec.action_mode
                side_channel_mode = spec.side_channel_mode
                pi0_paths = resolve_pi0_eval_job_paths(
                    asset or {},
                    checkpoint_path=checkpoint_path,
                )
            from app.services.policy_schema_resolver import resolve_eval_robot_for_policy

            robot, _robot_warnings = resolve_eval_robot_for_policy(
                policy=policy or "",
                model_asset=asset or {},
                checkpoint_path=checkpoint_path,
                eval_executor=eval_executor,
                controller_type=controller_type,
                action_mode=action_mode,
            )
            if policy == "pi0":
                pi0_job_fields = {
                    "train_config_path": pi0_paths.get("trainConfigPath"),
                    "task_instruction": pi0_paths.get("taskInstruction"),
                    "source_train_job_id": pi0_paths.get("sourceTrainJobId"),
                    "state_dim": pi0_paths.get("stateDim"),
                    "action_dim": pi0_paths.get("actionDim"),
                    "model_type": pi0_paths.get("modelType"),
                    "policy_runtime": pi0_paths.get("policyRuntime"),
                }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"cable_threading unsupported evaluationMode: {mode}",
            )
        return NormalizedEvaluateRequest(
            task_type=task_type,
            task_template_id=task_template_id,
            internal_evaluation_mode=internal_mode,
            public_evaluation_mode=public_mode,
            policy=policy,
            checkpoint_path=checkpoint_path,
            dataset_id=dataset_id,
            model_asset_id=model_asset_id,
            eval_executor=eval_executor,
            controller_type=controller_type,
            action_mode=action_mode,
            side_channel_mode=side_channel_mode,
            robot=robot,
            train_config_path=pi0_job_fields.get("train_config_path"),
            task_instruction=pi0_job_fields.get("task_instruction"),
            source_train_job_id=pi0_job_fields.get("source_train_job_id"),
            state_dim=pi0_job_fields.get("state_dim"),
            action_dim=pi0_job_fields.get("action_dim"),
            model_type=pi0_job_fields.get("model_type"),
            policy_runtime=pi0_job_fields.get("policy_runtime"),
        )

    if task_type == "dual_arm_cable_manipulation":
        if mode == "episode_stability":
            return NormalizedEvaluateRequest(
                task_type=task_type,
                task_template_id=task_template_id,
                internal_evaluation_mode="episode_stability",
                public_evaluation_mode="episode_stability",
                dataset_id=dataset_id,
                model_asset_id=model_asset_id,
            )
        if mode == "trained_model_evaluation":
            checkpoint_path = _resolve_checkpoint(request, task_type=task_type)
            if not checkpoint_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="checkpointPath or modelAssetId required for trained_model_evaluation",
                )
            return NormalizedEvaluateRequest(
                task_type=task_type,
                task_template_id=task_template_id,
                internal_evaluation_mode="trained_model_evaluation",
                public_evaluation_mode="trained_model_evaluation",
                policy="torch_bc",
                checkpoint_path=checkpoint_path,
                dataset_id=dataset_id,
                model_asset_id=model_asset_id,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dual_arm_cable_manipulation unsupported evaluationMode: {mode}",
        )

    if task_type == "block_stacking":
        if mode != "trained_model_evaluation":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="block_stacking only supports trained_model_evaluation (expert policy not available)",
            )
        checkpoint_path = _resolve_checkpoint(request, task_type=task_type)
        if not checkpoint_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="checkpointPath or modelAssetId required for trained_model_evaluation",
            )
        return NormalizedEvaluateRequest(
            task_type=task_type,
            task_template_id=task_template_id,
            internal_evaluation_mode="trained_model_evaluation",
            public_evaluation_mode="trained_model_evaluation",
            policy="isaac_robomimic_bc",
            checkpoint_path=checkpoint_path,
            dataset_id=dataset_id,
            model_asset_id=model_asset_id,
        )

    if task_type == "nut_assembly":
        if mode != "trained_model_evaluation":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="nut_assembly only supports trained_model_evaluation",
            )
        checkpoint_path = _resolve_checkpoint(request, task_type=task_type)
        if not checkpoint_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="checkpointPath or modelAssetId required for trained_model_evaluation",
            )
        return NormalizedEvaluateRequest(
            task_type=task_type,
            task_template_id=task_template_id,
            internal_evaluation_mode="trained_model_evaluation",
            public_evaluation_mode="trained_model_evaluation",
            policy="robomimic_bc",
            checkpoint_path=checkpoint_path,
            dataset_id=dataset_id,
            model_asset_id=model_asset_id,
            robot="Panda",
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported taskType for evaluation: {task_type}",
    )


def to_internal_adapter_request(
    request: EvaluateAsyncRequest,
    normalized: NormalizedEvaluateRequest,
) -> EvaluateAsyncRequest:
    """构造适配器层使用的内部请求（episode_stability / policy_evaluation）。"""
    payload: dict[str, Any] = request.model_dump(mode="json", by_alias=True)
    payload["taskType"] = normalized.task_type
    payload["taskTemplateId"] = normalized.task_template_id
    payload["evaluationMode"] = normalized.internal_evaluation_mode
    if normalized.task_type == "cable_threading":
        cable = dict(payload.get("cableThreading") or {})
        if normalized.policy:
            cable["policyType"] = normalized.policy
        if normalized.checkpoint_path:
            cable["checkpointId"] = normalized.checkpoint_path
        if normalized.robot:
            cable["robot"] = normalized.robot
        if normalized.eval_executor:
            cable["evalExecutor"] = normalized.eval_executor
        if normalized.controller_type:
            cable["controllerType"] = normalized.controller_type
        if normalized.action_mode:
            cable["actionMode"] = normalized.action_mode
        if normalized.train_config_path:
            cable["trainConfigPath"] = normalized.train_config_path
        if normalized.task_instruction:
            cable["taskInstruction"] = normalized.task_instruction
        payload["cableThreading"] = cable
        payload["policyType"] = normalized.policy
        payload["checkpointId"] = normalized.checkpoint_path
    if normalized.task_type == "dual_arm_cable_manipulation":
        dac = dict(payload.get("dualArmCable") or {})
        if normalized.policy:
            dac["policyType"] = normalized.policy
        if normalized.checkpoint_path:
            dac["checkpointPath"] = normalized.checkpoint_path
        if normalized.model_asset_id:
            dac["modelAssetId"] = normalized.model_asset_id
        payload["dualArmCable"] = dac
        if normalized.checkpoint_path:
            payload["checkpointPath"] = normalized.checkpoint_path
        if normalized.model_asset_id:
            payload["modelAssetId"] = normalized.model_asset_id
    return EvaluateAsyncRequest(**payload)
