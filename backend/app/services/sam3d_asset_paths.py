"""Asset pipeline job directory layout and path safety."""

from __future__ import annotations

import json
import logging
import secrets
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional

from app.core.config import settings
from app.core.platform_paths import platform_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
JOB_ID_PREFIX = "asset_job_"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_asset_pipeline_root() -> Path:
    raw = (settings.SAM3D_OUTPUT_ROOT or "runs/asset_pipeline/jobs").strip()
    if raw == "runs/asset_pipeline/jobs":
        return platform_paths.runs_root / "asset_pipeline" / "jobs"
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def make_job_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{JOB_ID_PREFIX}{ts}_{suffix}"


def get_job_dir(job_id: str) -> Path:
    candidate = (job_id or "").strip()
    if not candidate.startswith(JOB_ID_PREFIX):
        raise ValueError(f"invalid asset pipeline job id: {job_id!r}")
    return get_asset_pipeline_root() / candidate


def ensure_job_dirs(job_dir: Path) -> None:
    job_dir = Path(job_dir)
    for rel in (
        "input",
        "sam3/masks",
        "sam3/cutouts",
        "sam3d",
        "sam3d/input_masks",
        "exports",
        "exports/mujoco",
        "exports/mujoco/meshes",
        "live",
        "logs",
        "metadata",
    ):
        (job_dir / rel).mkdir(parents=True, exist_ok=True)


def safe_join_job_file(job_dir: Path, rel_path: str) -> Path:
    job_dir = Path(job_dir).resolve()
    rel = (rel_path or "").strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        raise ValueError(f"invalid relative path: {rel_path!r}")
    candidate = (job_dir / rel).resolve()
    if candidate != job_dir and job_dir not in candidate.parents:
        raise ValueError(f"path escapes job directory: {rel_path!r}")
    return candidate


def read_job_json(job_dir: Path) -> dict[str, Any]:
    path = Path(job_dir) / "job.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_job_json(job_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(job_dir) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize_uploaded_image(
    uploaded: BinaryIO,
    *,
    filename: str,
    target_path: Path,
) -> dict[str, Any]:
    """Save upload as input/image.png when possible; otherwise keep original extension."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    raw = uploaded.read()
    if not raw:
        raise ValueError("empty upload file")

    ext = Path(filename or "").suffix.lower()
    converted = False
    actual_rel = "input/image.png"

    try:
        from PIL import Image

        img = Image.open(BytesIO(raw))
        img.save(target_path, format="PNG")
        converted = ext not in {".png"}
    except Exception as exc:
        logger.warning("PIL convert to PNG failed (%s), saving original bytes", exc)
        if ext in {".png", ".jpg", ".jpeg", ".webp"}:
            actual_name = f"image{ext if ext != '.jpeg' else '.jpg'}"
            actual_rel = f"input/{actual_name}"
            actual_path = target_path.parent / actual_name
            actual_path.write_bytes(raw)
            if actual_path != target_path and target_path.exists():
                target_path.unlink(missing_ok=True)
            target_path = actual_path
        else:
            target_path.write_bytes(raw)
            actual_rel = f"input/{Path(filename).name or 'upload.bin'}"

    return {
        "inputImage": actual_rel,
        "inputImagePath": str(target_path),
        "convertedToPng": converted,
        "originalFilename": filename,
    }


def resolve_input_image_path(job_dir: Path) -> Path:
    job = read_job_json(job_dir)
    rel = str(job.get("inputImage") or "input/image.png")
    path = safe_join_job_file(job_dir, rel)
    if path.is_file():
        return path
    fallback = job_dir / "input" / "image.png"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"input image not found for job {job_dir.name}")


def get_pipeline_run_script() -> Path:
    return (PROJECT_ROOT / "integrations" / "Sam3dAssetPipeline" / "run.py").resolve()


def count_sim_asset_jobs(*, reconstructed_only: bool = False) -> int:
    """Count asset pipeline jobs on disk for resource hub simAssets.

    When reconstructed_only=True, only jobs with sam3d output or status reconstructed count.
    """
    root = get_asset_pipeline_root()
    if not root.is_dir():
        return 0
    count = 0
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(JOB_ID_PREFIX):
            continue
        if reconstructed_only:
            has_output = any(
                (child / "sam3d" / name).is_file()
                for name in ("gs.ply", "splat.ply", "object.glb", "mesh.obj")
            )
            status = {}
            status_path = child / "live" / "status.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    status = {}
            if has_output or status.get("status") == "reconstructed":
                count += 1
        else:
            count += 1
    return count


def scan_job_files(job_dir: Path) -> list[dict[str, Any]]:
    job_dir = Path(job_dir)
    patterns = [
        "input/image.png",
        "input/image.jpg",
        "input/image.jpeg",
        "input/image.webp",
        "sam3/overlay.png",
        "sam3/combined_mask.png",
        "sam3/detections.json",
        "sam3/manifest.json",
        "sam3d/gs.ply",
        "sam3d/splat.ply",
        "sam3d/gaussian.ply",
        "sam3d/mesh.obj",
        "sam3d/mesh.stl",
        "sam3d/mesh.ply",
        "sam3d/object.glb",
        "sam3d/metadata.json",
        "sam3d/preview.gif",
        "sam3d/preview.mp4",
        "sam3d/logs/output_summary.txt",
        "logs/segment.log",
        "logs/reconstruct.log",
        "live/latest.png",
        "exports/mujoco/model_preview.xml",
        "exports/mujoco/model.xml",
        "exports/mujoco/mujoco_package.zip",
        "exports/mujoco/metadata.json",
        "exports/mujoco/meshes/visual.obj",
        "exports/mujoco/meshes/visual.stl",
        "exports/mujoco/meshes/collision.obj",
        "exports/mujoco/meshes/collision.stl",
        "exports/mujoco/preview.png",
        "exports/mujoco/turntable.mp4",
    ]
    files: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(rel: str) -> None:
        if rel in seen:
            return
        path = job_dir / rel
        if path.is_file():
            seen.add(rel)
            files.append(
                {
                    "path": rel,
                    "sizeBytes": path.stat().st_size,
                    "exists": True,
                }
            )

    for rel in patterns:
        add(rel)

    for subdir in ("sam3/masks", "sam3/cutouts"):
        d = job_dir / subdir
        if d.is_dir():
            for p in sorted(d.glob("*.png")):
                add(str(p.relative_to(job_dir)).replace("\\", "/"))

    return files
