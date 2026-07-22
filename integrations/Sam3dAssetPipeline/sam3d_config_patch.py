"""Create job-local SAM3D pipeline configs with offline DINO / MoGe paths."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


def _patch_dino_blocks(data: dict[str, Any], *, repo_or_dir: str) -> None:
    module = data.get("module") or {}
    condition = module.get("condition_embedder") or {}
    backbone = condition.get("backbone") or {}
    embedder_list = backbone.get("embedder_list") or []
    for entry in embedder_list:
        if not isinstance(entry, list) or not entry:
            continue
        embedder = entry[0]
        if not isinstance(embedder, dict):
            continue
        target = str(embedder.get("_target_", ""))
        if target.endswith(".Dino"):
            embedder["source"] = "local"
            embedder["repo_or_dir"] = repo_or_dir


def prepare_job_local_sam3d_config(
    *,
    job_dir: Path,
    sam3d_root: Path,
    dinov2_repo: str,
    moge_model_pt: str,
) -> Path:
    job_dir = Path(job_dir).resolve()
    sam3d_root = Path(sam3d_root).resolve()
    config_dir = job_dir / "sam3d" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    hf_dir = sam3d_root / "checkpoints" / "hf"
    for name in ("ss_generator.yaml", "slat_generator.yaml"):
        src = hf_dir / name
        dst = config_dir / name
        with src.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        _patch_dino_blocks(data, repo_or_dir=str(Path(dinov2_repo).resolve()))
        with dst.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

    with (hf_dir / "pipeline.yaml").open("r", encoding="utf-8") as handle:
        pipeline = yaml.safe_load(handle)

    for key in list(pipeline.keys()):
        if key.endswith("_ckpt_path") and isinstance(pipeline[key], str):
            ckpt_name = Path(pipeline[key]).name
            pipeline[key] = str((hf_dir / ckpt_name).resolve())
        elif key.endswith("_config_path") and isinstance(pipeline[key], str):
            cfg_name = Path(pipeline[key]).name
            pipeline[key] = str((config_dir / cfg_name).resolve())

    depth_model = pipeline.get("depth_model") or {}
    model_cfg = depth_model.get("model") or {}
    model_cfg["pretrained_model_name_or_path"] = str(Path(moge_model_pt).resolve())
    depth_model["model"] = model_cfg
    pipeline["depth_model"] = depth_model
    pipeline["compile_model"] = False

    pipeline_path = config_dir / "pipeline.yaml"
    with pipeline_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(pipeline, handle, sort_keys=False, allow_unicode=True)

    for ckpt_key in (
        "ss_decoder.yaml",
        "slat_decoder_gs.yaml",
        "slat_decoder_gs_4.yaml",
        "slat_decoder_mesh.yaml",
    ):
        src = hf_dir / ckpt_key
        if src.is_file():
            shutil.copy2(src, config_dir / ckpt_key)

    return pipeline_path
