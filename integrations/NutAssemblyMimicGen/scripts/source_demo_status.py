#!/usr/bin/env python3
"""Print JSON catalog of NutAssembly source demo options for API/UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from utils.official_assets import (
    LOCAL_SOURCE_DEMO,
    OFFICIAL_CORE_DEFAULT,
    OFFICIAL_SOURCE_DEFAULT,
    PROVENANCE_MANIFEST,
    is_official_source_validated,
    load_provenance_manifest,
    official_core_dataset_path,
    official_source_demo_path,
)
from utils.source_demo_provenance import audit_source_demo
from utils.source_demo_validation import validate_source_demo_file


def build_source_demo_catalog() -> dict:
    manifest = load_provenance_manifest()
    official_path = official_source_demo_path()
    core_path = official_core_dataset_path()
    local_path = LOCAL_SOURCE_DEMO

    official_val = validate_source_demo_file(official_path) if official_path.is_file() else {
        "path": str(official_path),
        "exists": False,
        "validationPassed": False,
    }
    local_audit = audit_source_demo(local_path) if local_path.is_file() else {
        "sourceDemoPath": str(local_path),
        "sourceDemoOrigin": "local_source_demo",
        "exists": False,
    }

    default_selection = "official" if is_official_source_validated() and official_path.is_file() else "local"
    default_warning = None
    if default_selection == "local":
        default_warning = "当前使用本地 source demo，非官方 MimicGen 数据。"

    return {
        "defaultSelection": default_selection,
        "defaultWarning": default_warning,
        "officialSourceValidated": is_official_source_validated(),
        "manifestPath": str(PROVENANCE_MANIFEST),
        "options": {
            "official": {
                "label": "官方 MimicGen source demo",
                "path": str(official_path),
                "exists": official_path.is_file(),
                "validationPassed": bool(official_val.get("validationPassed")),
                "sourceDemoOrigin": official_val.get("sourceDemoOrigin") or "official_mimicgen_source",
                "demoCount": official_val.get("demoCount"),
                "objectPoseKeys": official_val.get("objectPoseKeys") or [],
                "envName": official_val.get("envName"),
                "md5": official_val.get("md5"),
                "needsPrepare": official_val.get("needsPrepare"),
                "alreadyPrepared": official_val.get("alreadyPrepared"),
                "registryRelativePath": "source/nut_assembly.hdf5",
            },
            "local": {
                "label": "本地 source demo",
                "path": str(local_path),
                "exists": local_path.is_file(),
                "validationPassed": local_path.is_file(),
                "sourceDemoOrigin": local_audit.get("sourceDemoOrigin", "local_source_demo"),
                "demoCount": local_audit.get("demoCount"),
                "objectPoseKeys": local_audit.get("objectPoseKeys") or [],
                "envName": local_audit.get("envName"),
                "md5": local_audit.get("md5"),
                "warning": "当前使用本地 source demo，非官方 MimicGen 数据。",
            },
            "custom": {
                "label": "用户指定路径",
                "requiresPath": True,
            },
        },
        "coreDataset": {
            "path": str(core_path),
            "exists": core_path.is_file(),
            "purpose": "import_training_replay_validation",
            "notDefaultSourceDemo": True,
            "registryRelativePath": "core/nut_assembly_d0.hdf5",
        },
        "manifest": manifest,
    }


def main() -> int:
    print(json.dumps(build_source_demo_catalog(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
