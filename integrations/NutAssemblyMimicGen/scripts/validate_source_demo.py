#!/usr/bin/env python3
"""Validate official MimicGen NutAssembly source demo and core dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from utils.official_assets import (
    OFFICIAL_CORE_DEFAULT,
    OFFICIAL_SOURCE_DEFAULT,
    PROVENANCE_MANIFEST,
    official_core_dataset_path,
    official_source_demo_path,
)
from utils.source_demo_validation import (
    render_validation_report,
    validate_official_assets,
    validate_source_demo_file,
)

REPORT_PATH = _REPO_ROOT / "runs" / "nut_assembly" / "debug" / "official_source_demo_validation_report.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MimicGen NutAssembly official source demo")
    parser.add_argument("--source-path", default=None, help="Override source demo path")
    parser.add_argument("--core-path", default=None, help="Override core dataset path")
    parser.add_argument("--file", default=None, help="Validate arbitrary HDF5 file")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Markdown report output path")
    parser.add_argument("--json-only", action="store_true", help="Print manifest JSON only")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file).resolve()
        result = validate_source_demo_file(path, user_provided=True)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("validationPassed") else 1

    source = Path(args.source_path).resolve() if args.source_path else official_source_demo_path()
    core = Path(args.core_path).resolve() if args.core_path else official_core_dataset_path()
    manifest = validate_official_assets(source_path=source, core_path=core)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_validation_report(manifest), encoding="utf-8")

    if args.json_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nReport: {report_path}")
        print(f"Manifest: {PROVENANCE_MANIFEST}")

    source_ok = bool((manifest.get("source") or {}).get("validationPassed"))
    return 0 if source_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
