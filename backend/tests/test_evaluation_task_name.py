from __future__ import annotations

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.display_name import (
    extract_user_evaluation_task_name,
    resolve_evaluation_task_name,
)


def test_extract_user_task_name_from_top_level():
    request = EvaluateAsyncRequest(
        taskType="cable_threading",
        evaluationMode="expert_policy_evaluation",
        numEpisodes=1,
        taskName="FLOW_TEST_custom_task_name_cable_01",
    )
    assert extract_user_evaluation_task_name(request) == "FLOW_TEST_custom_task_name_cable_01"


def test_extract_user_task_name_from_nested_block():
    request = EvaluateAsyncRequest(
        taskType="dual_arm_cable_manipulation",
        evaluationMode="episode_stability",
        numEpisodes=1,
        dualArmCable={"taskName": "FLOW_TEST_custom_task_name_dual_01"},
    )
    assert extract_user_evaluation_task_name(request) == "FLOW_TEST_custom_task_name_dual_01"


def test_resolve_evaluation_task_name_prefers_user_input():
    request = EvaluateAsyncRequest(
        taskType="isaac_block_stacking",
        evaluationMode="trained_model_evaluation",
        numEpisodes=1,
        taskName="FLOW_TEST_custom_task_name_isaac_01",
    )
    task_name, generated = resolve_evaluation_task_name(
        request,
        "isaac_block_stacking",
        "trained_model_evaluation",
    )
    assert task_name == "FLOW_TEST_custom_task_name_isaac_01"
    assert "物块堆叠" in generated
    assert task_name != generated


def test_resolve_evaluation_task_name_falls_back_to_generated():
    request = EvaluateAsyncRequest(
        taskType="cable_threading",
        evaluationMode="expert_policy_evaluation",
        numEpisodes=1,
    )
    task_name, generated = resolve_evaluation_task_name(
        request,
        "cable_threading",
        "expert_policy_evaluation",
    )
    assert task_name == generated
    assert "线缆穿杆" in task_name
