"""Tests for pi0 evaluation job metadata / workspace sync fields (Phase G)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.schemas.evaluation import EvaluateAsyncRequest  # noqa: E402
from app.services.benchmark_adapters.cable_threading_adapter import CableThreadingTaskAdapter  # noqa: E402
from app.services.evaluation.evaluation_request_resolver import NormalizedEvaluateRequest  # noqa: E402


def test_pi0_evaluation_context_includes_schema(tmp_path: Path):
    adapter = CableThreadingTaskAdapter()
    job_root = tmp_path / "ct_eval_pi0_ctx"
    job_root.mkdir()
    normalized = NormalizedEvaluateRequest(
        task_type="cable_threading",
        task_template_id="cable_threading_single_arm",
        internal_evaluation_mode="policy_evaluation",
        public_evaluation_mode="trained_model_evaluation",
        policy="pi0",
        checkpoint_path="/tmp/model_final.pt",
        model_asset_id="model__123947_ebd2_final",
        eval_executor="joint_position",
        controller_type="JOINT_POSITION",
        action_mode="joint_delta_derived",
        robot="Panda",
        train_config_path="/tmp/train_config.json",
        task_instruction="thread the cable through the pole",
        source_train_job_id="train_20260630_123947_ebd2",
        state_dim=9,
        action_dim=8,
        model_type="pi0",
        policy_runtime="pi0",
    )
    request = EvaluateAsyncRequest(
        taskTemplateId="cable_threading_single_arm",
        evaluationMode="trained_model_evaluation",
        modelAssetId="model__123947_ebd2_final",
        numEpisodes=1,
    )
    adapter._write_evaluation_context(job_root, normalized, request)
    context_path = job_root / "metadata" / "evaluation_context.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["modelType"] == "pi0"
    assert payload["evalExecutor"] == "joint_position"
    assert payload["robot"] == "Panda"
    assert payload["controllerType"] == "JOINT_POSITION"
    assert payload["stateDim"] == 9
    assert payload["actionDim"] == 8
    assert payload["taskInstruction"] == "thread the cable through the pole"
    assert payload["sourceTrainJobId"] == "train_20260630_123947_ebd2"
    assert payload["modelAssetId"] == "model__123947_ebd2_final"
