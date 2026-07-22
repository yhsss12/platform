"""数据血缘写入：dataset → train → checkpoint → eval。"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.database import SessionLocal
from app.models.artifact_lineage import ArtifactLineage

logger = logging.getLogger(__name__)

REL_DATASET_USED_BY = "dataset_used_by"
REL_MODEL_GENERATED_FROM = "model_generated_from"
REL_EVAL_OF = "eval_of"


def record_lineage(
    *,
    parent_id: str,
    child_id: str,
    relation_type: str,
    job_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    parent = (parent_id or "").strip()
    child = (child_id or "").strip()
    rel = (relation_type or "").strip()
    if not parent or not child or not rel:
        return
    try:
        with SessionLocal() as db:
            row = (
                db.query(ArtifactLineage)
                .filter(
                    ArtifactLineage.parent_id == parent,
                    ArtifactLineage.child_id == child,
                    ArtifactLineage.relation_type == rel,
                )
                .one_or_none()
            )
            if row is None:
                db.add(
                    ArtifactLineage(
                        parent_id=parent,
                        child_id=child,
                        relation_type=rel,
                        job_id=job_id,
                        metadata_json=metadata or {},
                    )
                )
            elif metadata:
                merged = dict(row.metadata_json or {})
                merged.update(metadata)
                row.metadata_json = merged
            db.commit()
    except Exception as exc:
        logger.warning("record_lineage failed parent=%s child=%s rel=%s: %s", parent, child, rel, exc)


def sync_lineage_for_training_job(train_job_id: str) -> None:
    from app.models.workspace_index import ModelAsset
    from app.models.workspace_job import WorkspaceJob

    jid = (train_job_id or "").strip()
    if not jid:
        return
    try:
        with SessionLocal() as db:
            job = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == jid).one_or_none()
            if job is None:
                return
            meta = job.metadata_json if isinstance(job.metadata_json, dict) else {}
            train_config = meta.get("trainConfig") if isinstance(meta.get("trainConfig"), dict) else {}
            dataset_manifest = meta.get("datasetManifest") if isinstance(meta.get("datasetManifest"), dict) else {}
            dataset_id = str(
                train_config.get("datasetId")
                or train_config.get("sourceDatasetId")
                or dataset_manifest.get("datasetId")
                or dataset_manifest.get("id")
                or ""
            ).strip()
            if dataset_id:
                record_lineage(
                    parent_id=dataset_id,
                    child_id=jid,
                    relation_type=REL_DATASET_USED_BY,
                    job_id=jid,
                    metadata={"jobType": job.job_type},
                )
            assets = (
                db.query(ModelAsset)
                .filter(ModelAsset.train_job_id == jid, ModelAsset.status != "deleted")
                .all()
            )
            for asset in assets:
                record_lineage(
                    parent_id=jid,
                    child_id=asset.model_asset_id,
                    relation_type=REL_MODEL_GENERATED_FROM,
                    job_id=jid,
                    metadata={"checkpointKind": asset.checkpoint_kind or asset.asset_type},
                )
    except Exception as exc:
        logger.warning("sync_lineage_for_training_job failed job_id=%s: %s", jid, exc)


def sync_lineage_for_eval_job(eval_job_id: str) -> None:
    from app.models.workspace_index import EvalMetricSummary, ModelAsset
    from app.models.workspace_job import WorkspaceJob

    jid = (eval_job_id or "").strip()
    if not jid:
        return
    try:
        with SessionLocal() as db:
            job = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == jid).one_or_none()
            summary = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == jid).one_or_none()
            meta = job.metadata_json if job and isinstance(job.metadata_json, dict) else {}
            eval_req = meta.get("evaluationRequest") if isinstance(meta.get("evaluationRequest"), dict) else {}
            summary_json = summary.summary_json if summary and isinstance(summary.summary_json, dict) else {}

            model_asset_id = str(
                (summary.model_asset_id if summary else "")
                or eval_req.get("modelAssetId")
                or summary_json.get("modelAssetId")
                or ""
            ).strip()
            dataset_id = str(
                eval_req.get("datasetId")
                or eval_req.get("sourceDatasetId")
                or summary_json.get("datasetId")
                or ""
            ).strip()

            if model_asset_id:
                record_lineage(
                    parent_id=model_asset_id,
                    child_id=jid,
                    relation_type=REL_EVAL_OF,
                    job_id=jid,
                    metadata={"evalJobId": jid},
                )
                asset = db.query(ModelAsset).filter(ModelAsset.model_asset_id == model_asset_id).one_or_none()
                if asset and asset.train_job_id:
                    record_lineage(
                        parent_id=asset.train_job_id,
                        child_id=jid,
                        relation_type=REL_EVAL_OF,
                        job_id=jid,
                        metadata={"viaModelAsset": model_asset_id},
                    )
            if dataset_id:
                record_lineage(
                    parent_id=dataset_id,
                    child_id=jid,
                    relation_type=REL_DATASET_USED_BY,
                    job_id=jid,
                    metadata={"context": "evaluation"},
                )
    except Exception as exc:
        logger.warning("sync_lineage_for_eval_job failed job_id=%s: %s", jid, exc)


def sync_lineage_for_dataset_job(generate_job_id: str) -> None:
    jid = (generate_job_id or "").strip()
    if not jid:
        return
    record_lineage(
        parent_id=jid,
        child_id=f"dataset:{jid}",
        relation_type="dataset_ingested",
        job_id=jid,
        metadata={"source": "generate_job"},
    )
