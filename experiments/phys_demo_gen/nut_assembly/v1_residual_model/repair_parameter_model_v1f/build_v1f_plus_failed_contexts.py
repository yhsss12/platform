#!/usr/bin/env python3
"""Task 2：从 demo_failed(1) 提取 original_failed_context pool。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_V1F_DIR = Path(__file__).resolve().parent
_V1_DIR = _V1F_DIR.parent
_EXPERIMENT_DIR = _V1_DIR.parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
for path in (_EXPERIMENT_DIR, _V1_DIR, _V1F_DIR, _OFFLINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from repair_common_v1f import extract_baseline_context_v1f  # noqa: E402
from v1f_plus_utils import (  # noqa: E402
    DEFAULT_FAILED_HDF5,
    DEFAULT_PLUS_OUTPUT,
    list_demo_keys,
    load_failure_map,
)

DEFAULT_AUDIT = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"


def build_failed_contexts(
    *,
    failed_hdf5: Path,
    audit_report: Path,
    output: Path,
) -> list[dict[str, Any]]:
    failure_map = load_failure_map(audit_report)
    records: list[dict[str, Any]] = []
    for demo_key in list_demo_keys(failed_hdf5):
        info = failure_map.get(demo_key, {})
        coarse = info.get("coarse_failure_type", "transport_failed")
        search_kind = info.get("search_kind", "transport")
        context = extract_baseline_context_v1f(
            failed_hdf5=failed_hdf5,
            demo_key=demo_key,
            failure_type=coarse,
            search_kind=search_kind,
        )
        records.append(
            {
                "source_file": str(failed_hdf5),
                "source_demo": demo_key,
                "context_mode": "original_failed",
                "original_failure_type": info.get("rough_failure_type", "unknown"),
                "coarse_failure_type": coarse,
                "search_kind": search_kind,
                "sampler": info.get("sampler", "mixed"),
                "context": context,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build new failed context pool for V1-F-aligned-plus")
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLUS_OUTPUT / "new_failed_contexts.jsonl")
    args = parser.parse_args()

    records = build_failed_contexts(
        failed_hdf5=args.failed_hdf5,
        audit_report=args.audit_report,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output), "num_contexts": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
