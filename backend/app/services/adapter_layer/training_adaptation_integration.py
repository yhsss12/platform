from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from app.services.adapter_layer.training_adaptation_service import build_training_adaptation_plan


def resolve_adapter_model_type(payload: dict[str, Any]) -> str:
    """从 create_training_job 请求解析适配层 modelType。"""
    backend = str(payload.get("trainingBackend") or "").strip().lower()
    if backend and backend not in {"auto", "robomimic"}:
        if backend == "robomimic":
            return "robomimic_bc"
        return backend

    downstream = str(payload.get("downstreamModelType") or "").strip()
    mapping = {
        "Diffusion Policy": "diffusion_policy",
        "Robomimic": "robomimic_bc",
        "ACT": "act",
        "DT": "dt",
        "pi0": "pi0",
    }
    if downstream in mapping:
        return mapping[downstream]
    if backend == "robomimic":
        return "robomimic_bc"
    return backend or "robomimic_bc"


def build_dp_config_dict(
    profile: dict[str, Any],
    adaptation: dict[str, Any],
    *,
    dp_schema: Any | None = None,
) -> dict[str, Any]:
    """生成 Diffusion Policy 运行时 YAML 配置（替代 cable_threading 硬编码）。"""
    from app.services.adapter_layer.hdf5_inspector import sum_low_dim_key_dims
    from app.services.dp_schema_resolver import DpTrainingSchemaSpec, resolve_dp_training_schema

    arch = adaptation.get("architectureConfig") or {}
    input_cfg = adaptation.get("inputConfig") or {}
    training = adaptation.get("trainingConfig") or {}
    advanced = adaptation.get("advancedConfig") or {}

    task_name = (
        str(profile.get("taskName") or profile.get("taskType") or profile.get("datasetId") or "adapted_task")
        .strip()
        .lower()
        .replace(" ", "_")
    )

    artifacts = profile.get("artifacts") if isinstance(profile.get("artifacts"), dict) else {}
    hdf5_path = (
        profile.get("storageUri")
        or artifacts.get("hdf5")
        or profile.get("hdf5")
        or profile.get("hdf5Path")
        or profile.get("datasetPath")
    )

    manifest_source = dict(profile)
    if dp_schema is None or not isinstance(dp_schema, DpTrainingSchemaSpec):
        dp_schema = resolve_dp_training_schema(
            manifest_source,
            profile=profile,
            hdf5_path=hdf5_path,
        )

    camera_keys = list(dp_schema.image_keys or input_cfg.get("camera_keys") or profile.get("cameraKeys") or [])
    low_dim_keys = list(input_cfg.get("low_dim_keys") or dp_schema.low_dim_keys or [])
    if not low_dim_keys:
        obs_keys = list(input_cfg.get("obs_keys") or profile.get("observationKeys") or [])
        low_dim_keys = [k for k in obs_keys if k not in camera_keys and k != "attachment_enabled"]

    low_dim_dim: Optional[int] = None
    if low_dim_keys:
        if hdf5_path:
            low_dim_dim = sum_low_dim_key_dims(str(hdf5_path), low_dim_keys)
        if low_dim_dim is None and all(
            key in {"robot0_joint_pos", "robot0_gripper_qpos"} for key in low_dim_keys
        ):
            dim_map = {"robot0_joint_pos": 7, "robot0_gripper_qpos": 2}
            low_dim_dim = sum(dim_map[key] for key in low_dim_keys if key in dim_map)
    else:
        state_dim = int(profile.get("stateDim") or 0)
        low_dim_dim = state_dim if state_dim > 0 else None

    config = {
        "task_name": task_name,
        "action_dim": int(dp_schema.action_dim),
        "action_key": dp_schema.action_key,
        "action_mode": dp_schema.action_mode,
        "controller_type": dp_schema.controller_type,
        "eval_executor": dp_schema.eval_executor,
        "trained_action_mode": dp_schema.trained_action_mode,
        "gripper_action_key": dp_schema.gripper_action_key,
        "low_dim_keys": low_dim_keys,
        "image_keys": camera_keys,
        "low_dim_dim": low_dim_dim,
        "n_obs_steps": int(advanced.get("n_obs_steps") or arch.get("obs_horizon") or 2),
        "horizon": int(advanced.get("horizon") or arch.get("pred_horizon") or 16),
        "n_action_steps": int(advanced.get("n_action_steps") or arch.get("action_horizon") or 8),
        "num_diffusion_steps": int(advanced.get("num_diffusion_steps") or 20),
        "num_inference_steps": int(advanced.get("num_inference_steps") or 16),
        "batch_size": int(training.get("batchSize") or 16),
        "num_epochs": int(training.get("epochs") or 5),
        "learning_rate": float(training.get("learningRate") or 1e-4),
        "weight_decay": float(advanced.get("weight_decay") or 1e-4),
        "seed": int(training.get("seed") or 1),
        "image_size": int(advanced.get("image_size") or 128),
        "vision_encoder": str(
            (arch.get("image_encoder") or {}).get("type")
            or advanced.get("vision_encoder")
            or ("resnet18" if camera_keys else "tiny_cnn")
        ),
        "use_ema": bool(advanced.get("use_ema", True)),
        "ema_decay": float(advanced.get("ema_decay") or 0.999),
        "observation_schema": dp_schema.observation_schema,
        "action_schema": dp_schema.action_schema,
        "controller_schema": dp_schema.controller_schema,
        "side_channel_schema": dp_schema.side_channel_schema,
        "preferred_policy_schema_id": dp_schema.policy_schema_id,
    }
    return config


def build_act_config_dict(
    profile: dict[str, Any],
    adaptation: dict[str, Any],
    *,
    act_schema: Any | None = None,
) -> dict[str, Any]:
    """生成 ACT 运行时 YAML 配置。"""
    arch = adaptation.get("architectureConfig") or {}
    input_cfg = adaptation.get("inputConfig") or {}
    output_cfg = adaptation.get("outputConfig") or {}
    training = adaptation.get("trainingConfig") or {}
    advanced = adaptation.get("advancedConfig") or {}
    loader = adaptation.get("dataLoaderConfig") or {}

    camera_keys = list(
        input_cfg.get("image_keys")
        or input_cfg.get("camera_keys")
        or input_cfg.get("camera_names")
        or profile.get("cameraKeys")
        or []
    )
    low_dim_keys = list(input_cfg.get("low_dim_keys") or loader.get("state_keys") or [])
    if not low_dim_keys:
        obs_keys = list(input_cfg.get("obs_keys") or profile.get("observationKeys") or [])
        low_dim_keys = [k for k in obs_keys if k not in camera_keys]

    task_name = (
        str(profile.get("taskName") or profile.get("taskType") or profile.get("datasetId") or "adapted_task")
        .strip()
        .lower()
        .replace(" ", "_")
    )

    low_dim_dim: Optional[int] = None
    if act_schema is not None:
        from app.services.policy_schema_resolver import to_act_config_fields

        schema_fields = to_act_config_fields(act_schema)
        if schema_fields.get("image_keys"):
            camera_keys = list(schema_fields["image_keys"])
        if schema_fields.get("low_dim_keys"):
            low_dim_keys = list(schema_fields["low_dim_keys"])
    if low_dim_keys and low_dim_dim is None:
        from app.services.adapter_layer.hdf5_inspector import sum_low_dim_key_dims

        hdf5_guess = str(profile.get("storageUri") or profile.get("hdf5") or "")
        if hdf5_guess:
            low_dim_dim = sum_low_dim_key_dims(hdf5_guess, low_dim_keys)
        elif all(key in {"robot0_joint_pos", "robot0_gripper_qpos"} for key in low_dim_keys):
            low_dim_dim = 9

    action_dim = int(
        (act_schema.action_dim if act_schema is not None else None)
        or output_cfg.get("action_dim")
        or profile.get("actionDim")
        or 7
    )

    config: dict[str, Any] = {
        "task_name": task_name,
        "action_dim": action_dim,
        "action_key": str(output_cfg.get("action_key") or (act_schema.action_key if act_schema else None) or "actions"),
        "action_mode": str(
            output_cfg.get("action_mode") or (act_schema.action_mode if act_schema else None) or "osc_pose_delta_eef"
        ),
        "controller_type": str(
            output_cfg.get("controller_type") or (act_schema.controller_type if act_schema else None) or "OSC_POSE"
        ),
        "eval_executor": str(
            output_cfg.get("eval_executor") or (act_schema.eval_executor if act_schema else None) or "osc_pose"
        ),
        "trained_action_mode": str(
            output_cfg.get("trained_action_mode")
            or (act_schema.trained_action_mode if act_schema else None)
            or output_cfg.get("action_mode")
            or "osc_pose_delta_eef"
        ),
        "gripper_action_key": output_cfg.get("gripper_action_key") or (
            act_schema.gripper_action_key if act_schema else None
        ),
        "state_dim": int(low_dim_dim or profile.get("stateDim") or input_cfg.get("state_dim") or 0) or None,
        "low_dim_dim": low_dim_dim,
        "chunk_size": int(advanced.get("chunk_size") or arch.get("chunk_size") or loader.get("chunk_size") or 20),
        "image_keys": camera_keys,
        "low_dim_keys": low_dim_keys,
        "hidden_dim": int(advanced.get("hidden_dim") or arch.get("hidden_dim") or 512),
        "dim_feedforward": int(advanced.get("dim_feedforward") or arch.get("dim_feedforward") or 2048),
        "enc_layers": int(advanced.get("enc_layers") or arch.get("enc_layers") or 4),
        "dec_layers": int(advanced.get("dec_layers") or arch.get("dec_layers") or 4),
        "nheads": int(advanced.get("nheads") or arch.get("nheads") or 8),
        "dropout": float(advanced.get("dropout") or arch.get("dropout") or 0.1),
        "kl_weight": float(advanced.get("kl_weight") or arch.get("kl_weight") or 10.0),
        "latent_dim": int(advanced.get("latent_dim") or arch.get("latent_dim") or 32),
        "backbone": str(advanced.get("backbone") or arch.get("backbone") or "tiny_cnn"),
        "batch_size": int(training.get("batchSize") or 8),
        "num_epochs": int(training.get("epochs") or 5),
        "learning_rate": float(training.get("learningRate") or 1e-4),
        "weight_decay": float(advanced.get("weight_decay") or 1e-4),
        "seed": int(training.get("seed") or 1),
        "image_size": int(advanced.get("image_size") or 128),
        "val_ratio": float(loader.get("val_ratio") or 0.1),
        "act_variant": str(input_cfg.get("act_variant") or "image_proprio"),
    }
    if act_schema is not None:
        config["preferred_policy_schema_id"] = act_schema.policy_schema_id
        config["observation_schema"] = act_schema.observation_schema
        config["action_schema"] = act_schema.action_schema
        config["controller_schema"] = act_schema.controller_schema
        config["side_channel_schema"] = act_schema.side_channel_schema
    return config


def write_act_config_yaml(train_job_dir: Path, act_config: dict[str, Any]) -> Path:
    config_path = train_job_dir / "config" / "act_adapted.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v for k, v in act_config.items() if v is not None}
    config_path.write_text(yaml.safe_dump(cleaned, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def write_dp_config_yaml(train_job_dir: Path, dp_config: dict[str, Any]) -> Path:
    config_path = train_job_dir / "config" / "dp_adapted.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v for k, v in dp_config.items() if v is not None}
    config_path.write_text(yaml.safe_dump(cleaned, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def apply_training_adaptation(
    *,
    manifest: dict[str, Any],
    payload: dict[str, Any],
    train_job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    在 create_training_job 内部调用：生成适配方案并合并到训练配置。

    返回 (merged_fields, adaptation_snapshot)。
    用户提交的基础超参（epochs 等）优先于适配默认值。
    """
    model_type = resolve_adapter_model_type(payload)
    overrides = payload.get("adaptationOverrides")
    if not isinstance(overrides, dict):
        overrides = {}
    if payload.get("modelTypeId") and isinstance(payload.get("modelParams"), dict):
        overrides = {**overrides, **payload["modelParams"]}

    adaptation_result = build_training_adaptation_plan(
        dataset_id=str(manifest.get("datasetId") or payload.get("datasetId") or ""),
        raw_manifest=manifest,
        model_type=model_type,
        overrides=overrides,
    )

    patch = dict(adaptation_result.get("configPatch") or {})
    validation = adaptation_result.get("validation") or {}
    model_adaptation = adaptation_result.get("modelAdaptation") or {}
    dataset_profile = adaptation_result.get("datasetProfile") or {}

    merged: dict[str, Any] = {
        "downstreamModelType": patch.get("downstreamModelType") or payload.get("downstreamModelType"),
        "trainingBackend": patch.get("trainingBackend") or payload.get("trainingBackend"),
        "architectureConfig": patch.get("architectureConfig"),
        "dataLoaderConfig": patch.get("dataLoaderConfig"),
        "normalizationConfig": patch.get("normalizationConfig"),
        "inputConfig": patch.get("inputConfig"),
        "outputConfig": patch.get("outputConfig"),
    }

    if not payload.get("advancedEnabled") and patch.get("modelParams"):
        merged["advancedEnabled"] = True
        merged["modelParams"] = patch.get("modelParams")
    elif payload.get("advancedEnabled") and isinstance(payload.get("modelParams"), dict):
        base_params = dict(patch.get("modelParams") or {})
        base_params.update(payload["modelParams"])
        merged["modelParams"] = base_params
        merged["advancedEnabled"] = True
    elif patch.get("modelParams"):
        merged["modelParams"] = patch.get("modelParams")
        merged["advancedEnabled"] = bool(patch.get("advancedEnabled", True))

    for key in ("saveFinal", "saveBest", "checkpointIntervalEpochs"):
        if patch.get(key) is not None and payload.get(key) is None:
            merged[key] = patch[key]

    backend = str(merged.get("trainingBackend") or "").lower()
    if backend == "diffusion_policy":
        from app.services.dp_schema_resolver import resolve_dp_training_schema

        hdf5_guess = str((manifest.get("artifacts") or {}).get("hdf5") or manifest.get("hdf5") or "")
        if not hdf5_guess:
            hdf5_guess = str((manifest.get("storageUri") or "")).strip()
        dp_schema = resolve_dp_training_schema(
            manifest,
            hdf5_path=Path(hdf5_guess) if hdf5_guess else None,
            profile=dataset_profile,
        )
        dp_config = build_dp_config_dict(dataset_profile, model_adaptation, dp_schema=dp_schema)
        dp_path = write_dp_config_yaml(train_job_dir, dp_config)
        merged["dpConfigPath"] = str(dp_path)
        merged["dpConfig"] = dp_config
        merged["trainedActionMode"] = dp_schema.trained_action_mode
        merged["evalExecutor"] = dp_schema.eval_executor
        merged["controllerType"] = dp_schema.controller_type
        merged["actionMode"] = dp_schema.action_mode
        merged["actionSchema"] = dp_schema.action_schema
        merged["observationSchema"] = dp_schema.observation_schema
        merged["controllerSchema"] = dp_schema.controller_schema
        merged["sideChannelSchema"] = dp_schema.side_channel_schema
    elif backend == "act":
        from app.services.policy_schema_resolver import resolve_act_training_schema

        hdf5_guess = str((manifest.get("artifacts") or {}).get("hdf5") or manifest.get("hdf5") or "")
        if not hdf5_guess:
            hdf5_guess = str((manifest.get("storageUri") or "")).strip()
        act_schema = resolve_act_training_schema(
            manifest,
            hdf5_path=Path(hdf5_guess) if hdf5_guess else None,
            profile=dataset_profile,
        )
        act_config = build_act_config_dict(dataset_profile, model_adaptation, act_schema=act_schema)
        act_path = write_act_config_yaml(train_job_dir, act_config)
        merged["actConfigPath"] = str(act_path)
        merged["actConfig"] = act_config
        merged["trainedActionMode"] = act_schema.trained_action_mode
        merged["evalExecutor"] = act_schema.eval_executor
        merged["controllerType"] = act_schema.controller_type
        merged["actionMode"] = act_schema.action_mode
        merged["actionSchema"] = act_schema.action_schema
        merged["observationSchema"] = act_schema.observation_schema
        merged["controllerSchema"] = act_schema.controller_schema
        merged["sideChannelSchema"] = act_schema.side_channel_schema
    elif backend == "pi0":
        from app.services.pi0_training_runner import build_pi0_config_dict, write_pi0_config_yaml

        pi0_stub_index = train_job_dir / "artifacts" / "pi0_dataset_index.json"
        pi0_stub_index.parent.mkdir(parents=True, exist_ok=True)
        if not pi0_stub_index.is_file():
            pi0_stub_index.write_text(json.dumps({"pending": True}), encoding="utf-8")
        hdf5_guess = str((manifest.get("artifacts") or {}).get("hdf5") or "")
        pi0_config = build_pi0_config_dict(
            train_job_dir=train_job_dir,
            manifest=manifest,
            train_config={**payload, **merged},
            model_adaptation=model_adaptation,
            dataset_index_path=pi0_stub_index,
            hdf5_path=Path(hdf5_guess) if hdf5_guess else train_job_dir / "artifacts" / "dataset.hdf5",
        )
        pi0_path = write_pi0_config_yaml(train_job_dir, pi0_config)
        merged["pi0ConfigPath"] = str(pi0_path)
        merged["openpiPlatformConfigPath"] = str(pi0_path)
        merged["pi0Config"] = {
            "camera_keys": pi0_config.get("dataset", {}).get("camera_keys") or [],
            "low_dim_keys": pi0_config.get("dataset", {}).get("low_dim_keys") or [],
            "structure": pi0_config.get("structure") or {},
        }

    snapshot = {
        "datasetProfile": dataset_profile,
        "modelType": model_type,
        "modelAdaptation": model_adaptation,
        "configPatch": patch,
        "validation": validation,
        "explanation": adaptation_result.get("explanation") or [],
        "adapterLayerVersion": adaptation_result.get("adapterLayerVersion") or "2.0",
    }
    if payload.get("adaptationSnapshot"):
        snapshot["clientProvided"] = payload.get("adaptationSnapshot")

    merged["adaptationSnapshot"] = snapshot
    return merged, snapshot


def write_adaptation_artifact(train_job_dir: Path, snapshot: dict[str, Any]) -> None:
    path = train_job_dir / "artifacts" / "training_adaptation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
