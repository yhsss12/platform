from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
_LEGACY_REGISTRY_DIR = _REPO_ROOT / "configs" / "experiments" / "nut_assembly" / "pinn"
_DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (_REPO_ROOT / "eai-data")).expanduser()
_MODEL_ASSETS_DIR = _DATA_ROOT / "assets" / "models" / "pinn"


def _resolve_repo_path(rel: str) -> Path:
    if rel.startswith("assets/"):
        return (_DATA_ROOT / rel).resolve()
    return (_REPO_ROOT / rel).resolve()


def _model_assets_dir(model_id: str) -> Path:
    return _MODEL_ASSETS_DIR / model_id


def load_pinn_model_registry(model_id: str) -> dict[str, Any]:
    assets_meta = _model_assets_dir(model_id) / "metadata.json"
    if assets_meta.is_file():
        data = json.loads(assets_meta.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    path = _LEGACY_REGISTRY_DIR / f"{model_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"PINN registry not found: {assets_meta} or {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid PINN registry: {path}")
    return data


def resolve_pinn_model_path(model_id: str) -> Path | None:
    reg = load_pinn_model_registry(model_id)
    assets_dir = _model_assets_dir(model_id)
    model_file = str(reg.get("modelFile") or "model.pt")
    primary = assets_dir / model_file
    if primary.is_file():
        return primary
    for rel in reg.get("modelPaths") or []:
        candidate = _resolve_repo_path(str(rel))
        if candidate.is_file():
            return candidate
    return None


def resolve_pinn_backend(model_id: str) -> dict[str, Any]:
    """Return honest backend resolution: torch_model when model.pt loads, else heuristic fallback."""
    try:
        reg = load_pinn_model_registry(model_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "modelId": model_id,
            "available": False,
            "pinnBackend": "heuristic",
            "modelLoaded": False,
            "modelPath": None,
            "pipelineVersion": None,
            "displayName": model_id,
            "error": str(exc),
        }

    model_path = resolve_pinn_model_path(model_id)
    model_loaded = model_path is not None
    if model_loaded:
        return {
            "modelId": model_id,
            "available": True,
            "pinnBackend": "torch_model",
            "modelLoaded": True,
            "modelPath": str(model_path),
            "pipelineVersion": str(reg.get("pipelineVersionModel") or "model_v1"),
            "displayName": str(reg.get("displayName") or model_id),
            "repairStages": reg.get("repairStages") or [],
            "constraintsEnabled": reg.get("constraintsEnabled") or [],
            "error": None,
        }

    heuristic_ok = str(reg.get("status") or "") == "available" or bool(
        reg.get("pipelineVersionHeuristic") or reg.get("pipelineVersion") == "v1_heuristic"
    )
    return {
        "modelId": model_id,
        "available": heuristic_ok,
        "pinnBackend": "heuristic",
        "modelLoaded": False,
        "modelPath": None,
        "pipelineVersion": str(reg.get("pipelineVersionHeuristic") or reg.get("pipelineVersion") or "v1_heuristic"),
        "displayName": str(reg.get("displayName") or model_id),
        "repairStages": reg.get("repairStages") or [],
        "constraintsEnabled": reg.get("constraintsEnabled") or [],
        "error": None if heuristic_ok else "未检测到 PINN 修复模型，请先完成模型配置。",
    }


def check_pinn_model_availability(model_id: str) -> dict[str, Any]:
    info = resolve_pinn_backend(model_id)
    return {
        "modelId": info.get("modelId"),
        "available": bool(info.get("available")),
        "displayName": info.get("displayName"),
        "modelPath": info.get("modelPath"),
        "pipelineVersion": info.get("pipelineVersion"),
        "pinnBackend": info.get("pinnBackend"),
        "modelLoaded": bool(info.get("modelLoaded")),
        "repairStages": info.get("repairStages") or [],
        "constraintsEnabled": info.get("constraintsEnabled") or [],
        "error": info.get("error"),
    }
