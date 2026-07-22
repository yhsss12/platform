from __future__ import annotations

from app.services.model_asset_db_service import _infer_task_template_id


def test_infer_task_template_id_from_cable_dataset():
    template_id = _infer_task_template_id(
        manifest={"trainingBackend": "diffusion_policy"},
        dataset_id="ds_ct_gen_20260618_095819_8aa6",
        train_job_id="train_20260619_171526_8025",
    )
    assert template_id == "task_cable_threading_v1"
