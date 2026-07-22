#!/usr/bin/env python3
"""Check SAM3D offline model dependencies before reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _exists(path: str | Path) -> bool:
    return Path(path).expanduser().resolve().is_file() or Path(path).expanduser().resolve().is_dir()


def _find_moge_model(moge_path: str, hf_home: str) -> Path | None:
    direct = Path(moge_path).expanduser()
    if direct.is_file() and direct.name == "model.pt":
        return direct
    if direct.is_dir():
        candidate = direct / "model.pt"
        if candidate.is_file():
            return candidate
    hf_root = Path(hf_home).expanduser() / "hub"
    patterns = [
        hf_root / "models--Ruicheng--moge-vitl",
        hf_root / "models--Ruicheng--moge-vitl" / "snapshots",
    ]
    for base in patterns:
        if not base.is_dir():
            continue
        for model_pt in base.rglob("model.pt"):
            return model_pt
    return None


def _find_dinov2_repo(dinov2_repo: str, torch_home: str) -> Path | None:
    candidates = [
        Path(dinov2_repo).expanduser(),
        Path(torch_home).expanduser() / "hub" / "facebookresearch_dinov2_main",
        Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
    ]
    for candidate in candidates:
        if (candidate / "hubconf.py").is_file():
            return candidate.resolve()
    return None


def check_offline_dependencies(
    *,
    sam3d_root: str | Path,
    dinov2_repo: str,
    dinov2_model: str,
    moge_model_path: str,
    torch_home: str,
    hf_home: str,
) -> dict:
    sam3d_root = Path(sam3d_root).expanduser().resolve()
    missing: list[str] = []

    pipeline_yaml = sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
    ss_generator = sam3d_root / "checkpoints" / "hf" / "ss_generator.yaml"
    slat_generator = sam3d_root / "checkpoints" / "hf" / "slat_generator.yaml"

    for label, path in (
        ("pipeline.yaml", pipeline_yaml),
        ("ss_generator.yaml", ss_generator),
        ("slat_generator.yaml", slat_generator),
    ):
        if not path.is_file():
            missing.append(f"SAM3D {label} missing: {path}")

    dinov2_path = _find_dinov2_repo(dinov2_repo, torch_home)
    if dinov2_path is None:
        missing.append(
            f"DINOv2 local repo missing hubconf.py (checked {dinov2_repo} and torch hub cache)"
        )

    moge_pt = _find_moge_model(moge_model_path, hf_home)
    if moge_pt is None:
        missing.append(
            f"MoGe model.pt not found under {moge_model_path} or HF cache {hf_home}/hub"
        )

    ckpt_names = [
        "ss_generator.ckpt",
        "slat_generator.ckpt",
        "ss_decoder.ckpt",
        "slat_decoder_gs.ckpt",
    ]
    for name in ckpt_names:
        ckpt = sam3d_root / "checkpoints" / "hf" / name
        if not ckpt.is_file():
            missing.append(f"SAM3D checkpoint missing: {ckpt}")

    result = {
        "ok": not missing,
        "sam3dRoot": str(sam3d_root),
        "dinov2Repo": str(dinov2_path) if dinov2_path else None,
        "dinov2Model": dinov2_model,
        "mogeModelPath": str(moge_pt) if moge_pt else None,
        "pipelineYaml": str(pipeline_yaml) if pipeline_yaml.is_file() else None,
        "torchHome": str(Path(torch_home).expanduser()),
        "hfHome": str(Path(hf_home).expanduser()),
        "missing": missing,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAM3D offline dependencies")
    parser.add_argument("--sam3d-root", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--dinov2-model", default="dinov2_vitl14_reg")
    parser.add_argument("--moge-model-path", required=True)
    parser.add_argument("--torch-home", required=True)
    parser.add_argument("--hf-home", required=True)
    args = parser.parse_args()

    result = check_offline_dependencies(
        sam3d_root=args.sam3d_root,
        dinov2_repo=args.dinov2_repo,
        dinov2_model=args.dinov2_model,
        moge_model_path=args.moge_model_path,
        torch_home=args.torch_home,
        hf_home=args.hf_home,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
