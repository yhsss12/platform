from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import HTTPException, status

from app.services.nut_assembly_service import OUTPUT_ROOT, validate_job_id

logger = logging.getLogger(__name__)

FilterMode = Literal["all", "all_generated_demos", "success_only", "valid_for_training_only"]
DEFAULT_FILTER_MODE: FilterMode = "valid_for_training_only"
NO_TRAINABLE_MESSAGE = "当前数据集中暂无可训练成功轨迹，请先生成更多数据或优化策略。"


def _normalize_filter_mode(filter_mode: str) -> FilterMode:
    mode = (filter_mode or "").strip()
    if mode in {"all", "all_generated_demos", "success_only", "valid_for_training_only"}:
        return mode  # type: ignore[return-value]
    return DEFAULT_FILTER_MODE

_INTEGRATION_ROOT = Path(__file__).resolve().parents[3] / "integrations" / "NutAssemblyMimicGen"
_SOURCE_HDF5 = Path("datasets") / "nut_assembly_generated.hdf5"
_TRAINING_HDF5 = Path("datasets") / "nut_assembly_training.hdf5"
_TRAINING_MANIFEST = Path("datasets") / "training_build_manifest.json"


def _job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / validate_job_id(job_id)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def probe_training_build(
    job_id: str,
    *,
    filter_mode: FilterMode = DEFAULT_FILTER_MODE,
) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    source = job_dir / _SOURCE_HDF5
    script = _INTEGRATION_ROOT / "utils" / "training_hdf5_filter.py"
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; from pathlib import Path; "
            "sys.path.insert(0, %r); "
            "from utils.training_hdf5_filter import probe_training_filter; "
            "print(json.dumps(probe_training_filter(Path(sys.argv[1]), filter_mode=sys.argv[2])))"
        )
        % str(_INTEGRATION_ROOT),
        str(source),
        filter_mode,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr or proc.stdout or "probe_failed"}
    try:
        result = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid_probe_output"}
    result["jobId"] = job_id
    result["sourceHdf5"] = str(source)
    return result


def build_training_dataset(
    job_id: str,
    *,
    filter_mode: FilterMode = DEFAULT_FILTER_MODE,
) -> dict[str, Any]:
    filter_mode = _normalize_filter_mode(filter_mode)
    job_dir = _job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job not found: {job_id}")

    source = job_dir / _SOURCE_HDF5
    output = job_dir / _TRAINING_HDF5
    manifest_path = job_dir / _TRAINING_MANIFEST

    if not source.is_file():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="源 HDF5 不存在，请先生成 NutAssembly 数据。",
        )

    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; from pathlib import Path; "
            "sys.path.insert(0, %r); "
            "from utils.training_hdf5_filter import build_filtered_training_hdf5; "
            "result = build_filtered_training_hdf5("
            "source_hdf5=Path(sys.argv[1]), output_hdf5=Path(sys.argv[2]), "
            "filter_mode=sys.argv[3], source_job_id=sys.argv[4]); "
            "print(json.dumps(result))"
        )
        % str(_INTEGRATION_ROOT),
        str(source),
        str(output),
        filter_mode,
        job_id,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=proc.stderr or proc.stdout or "build_failed",
        )
    try:
        result = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="invalid_build_output",
        ) from exc

    if not result.get("ok"):
        message = str(result.get("error") or NO_TRAINABLE_MESSAGE)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": message,
                "filterMode": filter_mode,
                "probe": result,
            },
        )

    gen_summary = _read_json(job_dir / "results" / "generation_summary.json")
    job_manifest = _read_json(job_dir / "manifest.json")
    build_manifest = {
        "jobId": job_id,
        "taskTemplateId": job_manifest.get("taskTemplateId") or "nut_assembly_single_arm",
        "taskName": job_manifest.get("taskName") or "螺母装配",
        "taskType": "nut_assembly",
        "filterMode": filter_mode,
        "sourceHdf5": str(source),
        "trainingHdf5": str(output),
        "totalDemos": result.get("totalDemos"),
        "successDemos": result.get("successDemos"),
        "validForTrainingDemos": result.get("validForTrainingDemos"),
        "selectedDemos": result.get("selectedDemos"),
        "filteredDemos": result.get("filteredDemos"),
        "builtDemoCount": result.get("builtDemoCount"),
        "trainDemoCount": result.get("builtDemoCount"),
        "generationMode": gen_summary.get("generationMode") or job_manifest.get("generationMode"),
        "policyMode": gen_summary.get("policyMode") or job_manifest.get("policyMode"),
        "sourceGenerationMode": gen_summary.get("generationMode") or job_manifest.get("generationMode"),
        "sourcePolicyMode": gen_summary.get("policyMode") or job_manifest.get("policyMode"),
        "sourceDemoOrigin": gen_summary.get("sourceDemoOrigin") or job_manifest.get("sourceDemoOrigin"),
        "status": "built",
    }
    _write_json(manifest_path, build_manifest)

    job_manifest["trainingFilterMode"] = filter_mode
    job_manifest["trainingHdf5Path"] = str(_TRAINING_HDF5)
    job_manifest["trainingBuildManifest"] = str(_TRAINING_MANIFEST)
    job_manifest["trainableDemoCount"] = result.get("selectedDemos")
    job_manifest["filteredDemoCount"] = result.get("filteredDemos")
    _write_json(job_dir / "manifest.json", job_manifest)

    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

    invalidate_workspace_dataset_list_cache()

    return {
        "jobId": job_id,
        "status": "built",
        "filterMode": filter_mode,
        "trainingHdf5Path": str(output),
        "trainingBuildManifest": str(manifest_path),
        **build_manifest,
    }
