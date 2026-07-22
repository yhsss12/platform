from __future__ import annotations

import json
from pathlib import Path

from app.services.evaluation_workbench_basic_info import build_evaluation_workbench_basic_info


def test_build_workbench_basic_info_model_eval(tmp_path: Path) -> None:
    job_id = "ct_eval_20260625_090114_3c3a"
    job_root = tmp_path / "jobs" / job_id
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True)

    context = {
        "taskTemplateId": "cable_threading_single_arm",
        "evaluationMode": "trained_model_evaluation",
        "evaluationObject": "trained_model",
        "evaluationType": "model",
        "evaluationTypeLabel": "模型评测",
        "config": {
            "simulationPlatform": "mujoco",
            "robotType": "Panda",
            "taskType": "cable_threading",
        },
        "evaluationRequest": {
            "taskName": "线缆穿杆评测_20260625_814",
            "modelAssetId": "model__105826_57c8_final",
            "modelName": "线缆穿杆评测_20260625_814",
            "evaluationObject": "trained_model",
            "evaluationTypeLabel": "模型评测",
            "checkpointPath": "/home/ubuntu/project/eai-idev2.1/runs/training/jobs/train_20260624_105826_57c8/checkpoints/model_final.pth",
        },
    }
    (meta_dir / "evaluation_context.json").write_text(
        json.dumps(context, ensure_ascii=False),
        encoding="utf-8",
    )

    info = build_evaluation_workbench_basic_info(
        job_id,
        job_root,
        {"status": "completed", "taskType": "cable_threading"},
    )

    assert info["taskName"] == "线缆穿杆评测_20260625_814"
    assert info["evaluationTypeLabel"] == "模型评测"
    assert info["evaluationObjectLabel"] == "已训练模型"
    assert info["simulationPlatform"] == "MuJoCo"
    assert info["statusLabel"] == "已完成"
    assert info["robotType"] == "Panda"
    assert info["associatedTaskName"] == "线缆穿杆"
    assert info["modelAssetName"] == "线缆穿杆 · Final"


def test_build_workbench_basic_info_expert_dual_arm(tmp_path: Path) -> None:
    job_id = "eval_20260624_170036_ee2e"
    job_root = tmp_path / "jobs" / job_id
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True)

    request = {
        "taskType": "dual_arm_cable_manipulation",
        "evaluationMode": "episode_stability",
        "evaluationObject": "expert_policy",
        "evaluationTypeLabel": "专家策略评测",
        "taskName": "线缆整理稳定性评测",
        "config": {
            "simulationPlatform": "mujoco",
            "robotType": "dual_fr3",
        },
    }
    (meta_dir / "evaluation_request.json").write_text(
        json.dumps(request, ensure_ascii=False),
        encoding="utf-8",
    )

    info = build_evaluation_workbench_basic_info(
        job_id,
        job_root,
        {"status": "completed", "taskType": "dual_arm_cable_manipulation", "evaluationMode": "episode_stability"},
    )

    assert info["taskName"] == "线缆整理稳定性评测"
    assert info["evaluationTypeLabel"] == "专家策略评测"
    assert info["evaluationObjectLabel"] == "专家策略"
    assert info["associatedTaskName"] == "线缆整理"
    assert info["simulationPlatform"] == "MuJoCo"


def test_cable_threading_status_response_preserves_workbench_basic_info() -> None:
    from app.schemas.cable_threading import CableThreadingJobStatusResponse

    raw = {
        "jobId": "ct_eval_20260625_090114_3c3a",
        "taskType": "cable_threading",
        "status": "completed",
        "live": {},
        "paths": {},
        "metrics": {},
        "taskName": "线缆穿杆评测_20260625_814",
        "evaluationTypeLabel": "模型评测",
        "workbenchBasicInfo": {
            "taskName": "线缆穿杆评测_20260625_814",
            "evaluationTypeLabel": "模型评测",
            "evaluationObjectLabel": "已训练模型",
            "simulationPlatform": "MuJoCo",
            "statusLabel": "已完成",
            "robotType": "Panda",
            "modelAssetName": "线缆穿杆评测_20260625_814",
            "associatedTaskName": "线缆穿杆",
            "evaluationType": "model",
            "evaluationObject": "trained_model",
        },
    }
    parsed = CableThreadingJobStatusResponse(**raw)
    assert parsed.workbenchBasicInfo is not None
    assert parsed.workbenchBasicInfo.evaluationTypeLabel == "模型评测"
    dumped = parsed.model_dump()
    assert dumped["workbenchBasicInfo"]["taskName"] == "线缆穿杆评测_20260625_814"
    assert dumped["taskName"] == "线缆穿杆评测_20260625_814"


def test_trained_model_not_overridden_by_policy_name(tmp_path: Path) -> None:
    job_id = "ct_eval_policy_name_smoke"
    job_root = tmp_path / "jobs" / job_id
    meta_dir = job_root / "metadata"
    meta_dir.mkdir(parents=True)
    context = {
        "evaluationMode": "trained_model_evaluation",
        "evaluationObject": "trained_model",
        "evaluationTypeLabel": "模型评测",
        "evaluationRequest": {
            "taskName": "线缆穿杆评测_20260625_814",
            "modelAssetId": "model_xxx",
            "modelAssetName": "线缆穿杆 · Final",
            "evaluationObject": "trained_model",
        },
        "config": {"simulationPlatform": "mujoco", "robotType": "Panda"},
    }
    (meta_dir / "evaluation_context.json").write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
    info = build_evaluation_workbench_basic_info(
        job_id,
        job_root,
        {"status": "completed", "metrics": {"policy": "scripted"}},
    )
    assert info["evaluationTypeLabel"] == "模型评测"
    assert info["evaluationObjectLabel"] == "已训练模型"
