"""Normalize SAM3 outputs into a cutout-only manifest for frontend and SAM3D."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from status_utils import read_json, utc_now_iso, write_json_atomic


def _rel(job_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(job_dir.resolve())).replace("\\", "/")


def _mask_index_from_filename(name: str) -> int | None:
    match = re.search(r"(?:mask|cutout)_(\d+)\.png$", name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _numeric_cutout_index_from_filename(name: str) -> int | None:
    match = re.match(r"^(\d+)\.png$", name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _resolve_existing_path(job_dir: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(str(raw))
    if candidate.is_file():
        return candidate.resolve()
    rel = job_dir / str(raw).lstrip("/")
    if rel.is_file():
        return rel.resolve()
    return None


def _collect_raw_items(job_dir: Path, sam3_dir: Path, detections: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect detection-aligned items before numeric cutout standardization."""
    items: list[dict[str, Any]] = []
    seen: set[int] = set()

    for det in detections.get("detections") or []:
        raw_index = det.get("index")
        if raw_index is None:
            continue
        mask_index = int(raw_index)

        mask_path = None
        cutout_path = None
        for key, rel_key in (("mask", "maskPath"), ("cutout", "cutoutPath")):
            raw = det.get(key)
            resolved = _resolve_existing_path(job_dir, str(raw) if raw else None)
            if resolved is not None:
                if key == "mask":
                    mask_path = _rel(job_dir, resolved)
                else:
                    cutout_path = _rel(job_dir, resolved)

        if not mask_path:
            guess = sam3_dir / "masks" / f"mask_{mask_index:03d}.png"
            if guess.is_file():
                mask_path = _rel(job_dir, guess)
        if not cutout_path:
            guess = sam3_dir / "cutouts" / f"cutout_{mask_index:03d}.png"
            if guess.is_file():
                cutout_path = _rel(job_dir, guess)

        bbox = det.get("box_xyxy")
        score = det.get("score")
        items.append(
            {
                "maskIndex": mask_index,
                "label": f"mask_{mask_index:03d}",
                "score": float(score) if score is not None else None,
                "bbox": [float(v) for v in bbox] if isinstance(bbox, (list, tuple)) else None,
                "maskPath": mask_path,
                "cutoutPath": cutout_path,
            }
        )
        seen.add(mask_index)

    if not items:
        for mask_file in sorted((sam3_dir / "masks").glob("mask_*.png")):
            mask_index = _mask_index_from_filename(mask_file.name)
            if mask_index is None or mask_index in seen:
                continue
            cutout = sam3_dir / "cutouts" / f"cutout_{mask_index:03d}.png"
            items.append(
                {
                    "maskIndex": mask_index,
                    "label": f"mask_{mask_index:03d}",
                    "score": None,
                    "bbox": None,
                    "maskPath": _rel(job_dir, mask_file),
                    "cutoutPath": _rel(job_dir, cutout) if cutout.is_file() else None,
                }
            )
            seen.add(mask_index)

        for cutout_file in sorted((sam3_dir / "cutouts").glob("cutout_*.png")):
            mask_index = _mask_index_from_filename(cutout_file.name)
            if mask_index is None or mask_index in seen:
                continue
            items.append(
                {
                    "maskIndex": mask_index,
                    "label": f"mask_{mask_index:03d}",
                    "score": None,
                    "bbox": None,
                    "maskPath": None,
                    "cutoutPath": _rel(job_dir, cutout_file),
                }
            )

    items.sort(key=lambda x: int(x.get("maskIndex", 0)))
    return items


def _standardize_cutout_items(job_dir: Path, raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy valid cutouts to numeric filenames and build cutout-only manifest items."""
    cutouts_dir = job_dir / "sam3" / "cutouts"
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    standardized: list[dict[str, Any]] = []
    cutout_index = 0

    for raw in raw_items:
        source_rel = raw.get("cutoutPath")
        if not source_rel:
            continue
        source_path = _resolve_existing_path(job_dir, source_rel)
        if source_path is None or not source_path.is_file():
            continue

        cutout_index += 1
        numeric_name = f"{cutout_index}.png"
        numeric_path = cutouts_dir / numeric_name
        shutil.copy2(source_path, numeric_path)

        original_mask_rel = raw.get("maskPath")
        item: dict[str, Any] = {
            "cutoutIndex": cutout_index,
            "label": str(cutout_index),
            "score": raw.get("score"),
            "bbox": raw.get("bbox"),
            "cutoutPath": _rel(job_dir, numeric_path),
            "previewPath": _rel(job_dir, numeric_path),
            "originalCutoutPath": source_rel,
            "reconstructInputKind": "cutout",
            "selectable": True,
        }
        if original_mask_rel and _resolve_existing_path(job_dir, original_mask_rel):
            item["originalMaskPath"] = original_mask_rel
        standardized.append(item)

    return standardized


def build_sam3_manifest(job_dir: Path) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    sam3_dir = job_dir / "sam3"
    detections_path = sam3_dir / "detections.json"
    detections = read_json(detections_path) if detections_path.is_file() else {}

    job_id = read_json(job_dir / "job.json").get("jobId") or job_dir.name
    image_rel = read_json(job_dir / "job.json").get("inputImage") or "input/image.png"

    overlay = sam3_dir / "overlay.png"
    combined = sam3_dir / "combined_mask.png"

    raw_items = _collect_raw_items(job_dir, sam3_dir, detections)
    items = _standardize_cutout_items(job_dir, raw_items)

    manifest = {
        "jobId": job_id,
        "image": image_rel,
        "overlay": _rel(job_dir, overlay) if overlay.is_file() else None,
        "combinedMask": _rel(job_dir, combined) if combined.is_file() else None,
        "detections": "sam3/detections.json" if detections_path.is_file() else None,
        "items": items,
        "generatedAt": utc_now_iso(),
    }
    return manifest


def write_sam3_manifest(job_dir: Path) -> Path:
    job_dir = Path(job_dir)
    manifest = build_sam3_manifest(job_dir)
    out_path = job_dir / "sam3" / "manifest.json"
    write_json_atomic(out_path, manifest)
    return out_path


def _load_manifest(job_dir: Path) -> dict[str, Any]:
    manifest_path = Path(job_dir) / "sam3" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SAM3 manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_manifest_item_by_cutout_index(job_dir: Path, cutout_index: int) -> dict[str, Any]:
    manifest = _load_manifest(job_dir)
    for item in manifest.get("items") or []:
        if int(item.get("cutoutIndex", -1)) == int(cutout_index):
            return item
    raise FileNotFoundError(f"Selected cutoutIndex {cutout_index} not found in SAM3 manifest.")


def resolve_manifest_item(job_dir: Path, mask_index: int) -> dict[str, Any]:
    """Legacy resolver by maskIndex; prefer resolve_manifest_item_by_cutout_index."""
    manifest = _load_manifest(job_dir)
    for item in manifest.get("items") or []:
        if int(item.get("maskIndex", -1)) == int(mask_index):
            return item
    # New manifests use cutoutIndex (1-based); allow maskIndex+1 fallback.
    fallback = int(mask_index) + 1
    for item in manifest.get("items") or []:
        if int(item.get("cutoutIndex", -1)) == fallback:
            return item
    raise FileNotFoundError(f"Selected maskIndex {mask_index} not found in SAM3 manifest.")
