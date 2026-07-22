from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.services.adapter_layer.compatibility_checker import (
    CompatibilityAnalysis,
    analyze_dataset_compatibility,
    recommend_training_models,
)
from app.services.adapter_layer.evaluation_plan_builder import build_evaluation_plan
from app.services.adapter_layer.manifest_schema import DatasetManifest, normalize_dataset_manifest
from app.services.adapter_layer.training_adaptation_service import build_training_adaptation_plan
from app.services.adapter_layer.training_plan_builder import build_training_plan


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_manifest_by_dataset_id(dataset_id: str) -> Optional[dict[str, Any]]:
    """从 workspace 数据集索引查找原始 manifest（只读，不修改现有服务）。"""
    from app.services import workspace_dataset_service as dataset_svc

    target = (dataset_id or "").strip()
    if not target:
        return None

    for row in dataset_svc.list_datasets():
        if str(row.get("id") or "") != target:
            continue
        manifest_path = Path(str(row.get("manifestPath") or ""))
        manifest = _read_json(manifest_path)
        if not manifest:
            manifest = {}
        manifest.setdefault("datasetId", row.get("id"))
        manifest.setdefault("datasetName", row.get("name") or row.get("displayName"))
        manifest.setdefault("taskName", row.get("taskDisplayName") or row.get("taskType"))
        manifest.setdefault("taskType", row.get("taskType"))
        manifest.setdefault("sourceJobId", row.get("sourceJobId"))
        if row.get("episodeCount") is not None:
            manifest.setdefault("episodeCount", row.get("episodeCount"))
        if row.get("successfulEpisodes") is not None:
            manifest.setdefault("successfulEpisodes", row.get("successfulEpisodes"))
        if row.get("simulatorBackend"):
            manifest.setdefault("simulatorBackend", row.get("simulatorBackend"))
        if row.get("datasetFile"):
            artifacts = manifest.setdefault("artifacts", {})
            if isinstance(artifacts, dict) and not artifacts.get("hdf5"):
                artifacts["hdf5"] = row.get("datasetFile")
        return manifest
    return None


def compatibility_analysis_to_dict(analysis: CompatibilityAnalysis) -> dict[str, Any]:
    return {
        "datasetId": analysis.datasetId,
        "compatible": analysis.compatible,
        "manifestVersion": analysis.manifestVersion,
        "recommendedModels": analysis.recommendedModels,
        "blockingReasons": analysis.blockingReasons,
        "results": [
            {
                "modelType": item.modelType,
                "displayName": item.displayName,
                "compatible": item.compatible,
                "score": item.score,
                "reasons": item.reasons,
                "warnings": item.warnings,
                "status": item.status,
            }
            for item in analysis.results
        ],
    }


def get_dataset_compatibility(dataset_id: str) -> dict[str, Any]:
    raw = resolve_manifest_by_dataset_id(dataset_id)
    if raw is None:
        raise LookupError(f"数据集未找到: {dataset_id}")
    analysis = analyze_dataset_compatibility(raw)
    return compatibility_analysis_to_dict(analysis)


__all__ = [
    "normalize_dataset_manifest",
    "analyze_dataset_compatibility",
    "recommend_training_models",
    "build_training_plan",
    "build_evaluation_plan",
    "build_training_adaptation_plan",
    "resolve_manifest_by_dataset_id",
    "get_dataset_compatibility",
    "compatibility_analysis_to_dict",
]
