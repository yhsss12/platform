from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from utils.runtime_env import _DATA_ROOT, _REPO_ROOT

SourceDemoSelection = Literal["official", "local", "custom", "auto"]

OFFICIAL_ASSETS_ROOT = _DATA_ROOT / "assets" / "mimicgen" / "nut_assembly"
OFFICIAL_SOURCE_REL = "source/nut_assembly.hdf5"
OFFICIAL_CORE_REL = "core/nut_assembly_d0.hdf5"
OFFICIAL_SOURCE_DEFAULT = OFFICIAL_ASSETS_ROOT / "source" / "nut_assembly.hdf5"
OFFICIAL_CORE_DEFAULT = OFFICIAL_ASSETS_ROOT / "core" / "nut_assembly_d0.hdf5"
PROVENANCE_DIR = OFFICIAL_ASSETS_ROOT / "provenance"
PROVENANCE_MANIFEST = PROVENANCE_DIR / "official_source_manifest.json"
LOCAL_SOURCE_DEMO = _REPO_ROOT / "mnt" / "data" / "demo.hdf5"

HF_REPO_ID = "amandlek/mimicgen_datasets"


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def official_source_demo_path() -> Path:
    return _path_from_env("NUT_ASSEMBLY_OFFICIAL_SOURCE_DEMO_PATH") or OFFICIAL_SOURCE_DEFAULT


def official_core_dataset_path() -> Path:
    return _path_from_env("NUT_ASSEMBLY_OFFICIAL_CORE_DATASET_PATH") or OFFICIAL_CORE_DEFAULT


def load_provenance_manifest() -> dict[str, Any]:
    if not PROVENANCE_MANIFEST.is_file():
        return {}
    try:
        data = json.loads(PROVENANCE_MANIFEST.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_provenance_manifest(payload: dict[str, Any]) -> Path:
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return PROVENANCE_MANIFEST


def is_official_source_validated() -> bool:
    manifest = load_provenance_manifest()
    source_block = manifest.get("source") or {}
    path = official_source_demo_path()
    if not path.is_file():
        return False
    if not source_block.get("validationPassed"):
        return False
    registered_md5 = source_block.get("md5")
    if registered_md5:
        from utils.source_demo_provenance import _file_hashes

        current = _file_hashes(path).get("md5")
        return current == registered_md5
    return bool(source_block.get("validationPassed"))


def resolve_source_demo_for_selection(
    selection: SourceDemoSelection | str | None,
    custom_path: str | None = None,
) -> tuple[Path | None, str | None, SourceDemoSelection]:
    """
    Resolve source demo path by UI/API selection.
    Does not download — missing official file returns error for official selection.
    """
    sel = (selection or "auto").strip().lower()
    if sel not in {"official", "local", "custom", "auto"}:
        sel = "auto"

    if sel == "custom":
        if not custom_path or not str(custom_path).strip():
            return None, "custom source demo path required", "custom"
        raw = Path(str(custom_path).strip())
        resolved = raw if raw.is_absolute() else (_REPO_ROOT / raw).resolve()
        if not resolved.is_file():
            return None, f"source_demo_missing: {resolved}", "custom"
        return resolved, None, "custom"

    if sel == "local":
        local = LOCAL_SOURCE_DEMO
        if local.is_file():
            return local.resolve(), None, "local"
        return None, f"source_demo_missing: local debug demo not found at {local}", "local"

    if sel == "official":
        official = official_source_demo_path()
        if not official.is_file():
            return (
                None,
                "official_source_demo_missing: download or manually place "
                f"source/nut_assembly.hdf5 at {official} "
                "(run integrations/NutAssemblyMimicGen/scripts/download_official_source_demo.py "
                "or set NUT_ASSEMBLY_OFFICIAL_SOURCE_DEMO_PATH)",
                "official",
            )
        return official.resolve(), None, "official"

    # auto: env override > validated official > local (never core)
    env_override = _path_from_env("NUT_ASSEMBLY_SOURCE_DEMO_PATH")
    if env_override and env_override.is_file():
        return env_override.resolve(), None, "auto"

    if is_official_source_validated():
        official = official_source_demo_path()
        if official.is_file():
            return official.resolve(), None, "auto"

    if LOCAL_SOURCE_DEMO.is_file():
        return LOCAL_SOURCE_DEMO.resolve(), None, "auto"

    legacy = Path("/mnt/data/demo.hdf5")
    if legacy.is_file():
        return legacy.resolve(), None, "auto"

    return None, "source_demo_missing: no official or local source demo available", "auto"
