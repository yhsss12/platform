#!/usr/bin/env python3
"""Download official MimicGen NutAssembly datasets from HuggingFace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from utils.official_assets import (
    HF_REPO_ID,
    OFFICIAL_ASSETS_ROOT,
    OFFICIAL_CORE_DEFAULT,
    OFFICIAL_CORE_REL,
    OFFICIAL_SOURCE_DEFAULT,
    OFFICIAL_SOURCE_REL,
    PROVENANCE_MANIFEST,
    load_provenance_manifest,
    save_provenance_manifest,
)
from utils.runtime_env import resolve_mimicgen_root

SOURCE_HF_FILE = OFFICIAL_SOURCE_REL
CORE_HF_FILE = OFFICIAL_CORE_REL


def _download_via_hf_hub(filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(filename).name
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed. pip install huggingface_hub "
            "or manually place files under runtime_assets/mimicgen/nut_assembly/"
        ) from exc

    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    kwargs: dict = {
        "repo_id": HF_REPO_ID,
        "filename": filename,
        "repo_type": "dataset",
    }
    if endpoint:
        kwargs["endpoint"] = endpoint

    try:
        cached = hf_hub_download(**kwargs)
    except Exception:
        # Fallback: direct resolve URL via HF mirror when huggingface.co is unreachable.
        import urllib.request

        mirror = os.environ.get("HF_MIRROR", "https://hf-mirror.com").rstrip("/")
        url = f"{mirror}/datasets/{HF_REPO_ID}/resolve/main/{filename}"
        urllib.request.urlretrieve(url, dest_path)
        return dest_path

    shutil.copy2(cached, dest_path)
    return dest_path


def _download_via_mimicgen(filename: str, dest_dir: Path) -> Path:
    mimicgen_root = resolve_mimicgen_root()
    if mimicgen_root is None:
        raise RuntimeError("mimicgen vendor package not found")
    if str(mimicgen_root) not in sys.path:
        sys.path.insert(0, str(mimicgen_root))
    from mimicgen.utils.file_utils import download_file_from_hf

    dest_dir.mkdir(parents=True, exist_ok=True)
    download_file_from_hf(
        repo_id=HF_REPO_ID,
        filename=filename,
        download_dir=str(dest_dir),
        check_overwrite=False,
    )
    return dest_dir / Path(filename).name


def download_file(filename: str, dest_path: Path, *, force: bool = False) -> dict:
    result: dict = {
        "filename": filename,
        "destPath": str(dest_path),
        "downloaded": False,
        "skipped": False,
        "error": None,
    }
    if dest_path.is_file() and not force:
        result["skipped"] = True
        result["fileSizeBytes"] = dest_path.stat().st_size
        return result

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            tmp = _download_via_mimicgen(filename, dest_path.parent)
        except Exception:
            tmp = _download_via_hf_hub(filename, dest_path.parent)
        if tmp.resolve() != dest_path.resolve() and tmp.is_file():
            shutil.move(str(tmp), str(dest_path))
        result["downloaded"] = True
        result["fileSizeBytes"] = dest_path.stat().st_size
    except Exception as exc:
        result["error"] = str(exc)
        result["manualHint"] = (
            f"Offline/manual: place HuggingFace file `{filename}` from repo `{HF_REPO_ID}` "
            f"at `{dest_path}` or set NUT_ASSEMBLY_OFFICIAL_SOURCE_DEMO_PATH / "
            f"NUT_ASSEMBLY_OFFICIAL_CORE_DATASET_PATH"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official MimicGen NutAssembly datasets")
    parser.add_argument("--source", action="store_true", help="Download source/nut_assembly.hdf5")
    parser.add_argument("--core", action="store_true", help="Download core/nut_assembly_d0.hdf5")
    parser.add_argument("--all", action="store_true", help="Download both source and core")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    download_source = args.source or args.all or (not args.source and not args.core)
    download_core = args.core or args.all

    OFFICIAL_ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "hfRepoId": HF_REPO_ID,
        "downloadedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": [],
    }

    if download_source:
        report["results"].append(download_file(SOURCE_HF_FILE, OFFICIAL_SOURCE_DEFAULT, force=args.force))
    if download_core:
        report["results"].append(download_file(CORE_HF_FILE, OFFICIAL_CORE_DEFAULT, force=args.force))

    manifest = load_provenance_manifest()
    manifest["download"] = report
    save_provenance_manifest(manifest)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = [r for r in report["results"] if r.get("error")]
    if failures:
        print("\n=== DOWNLOAD FAILED — manual placement required ===", file=sys.stderr)
        for f in failures:
            print(f"  {f.get('filename')}: {f.get('error')}", file=sys.stderr)
            if f.get("manualHint"):
                print(f"  Hint: {f['manualHint']}", file=sys.stderr)
        return 1

    print(f"\nProvenance manifest: {PROVENANCE_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
