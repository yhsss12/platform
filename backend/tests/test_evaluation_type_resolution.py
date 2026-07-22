from __future__ import annotations

from app.services.evaluation.evaluation_type import resolve_evaluation_type_from_sources


def test_joint_dp_import_resolves_as_model_evaluation():
    resolution = resolve_evaluation_type_from_sources(
        task_type="cable_threading",
        runner="run.py",
        task_name="Joint-Space DP · 10ep 评测 (horizon=1200)",
        model_asset_id="model_joint_dp_20260624_full_final",
        dataset_id="ds_ct_gen_20260624_joint_space_replay_full",
        metrics={
            "modelAssetId": "model_joint_dp_20260624_full_final",
            "datasetId": "ds_ct_gen_20260624_joint_space_replay_full",
        },
    )
    assert resolution["evaluationType"] == "model"
    assert resolution["evaluationTypeLabel"] == "模型评测"


def test_dataset_offline_still_resolves_as_dataset():
    resolution = resolve_evaluation_type_from_sources(
        task_type="dataset_offline",
        runner="dataset_offline_eval",
        task_name="离线数据集评测 · demo dataset",
        dataset_id="ds_demo",
    )
    assert resolution["evaluationType"] == "dataset"
    assert resolution["evaluationTypeLabel"] == "数据集评测"
