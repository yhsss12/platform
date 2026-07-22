#!/usr/bin/env python3
"""只读检查 Dataset Registry 健康情况。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services import workspace_dataset_service as svc  # noqa: E402


def main() -> int:
    raw_rows = svc.list_datasets()
    valid_rows: list[dict] = []
    invalid: list[dict] = []

    for row in raw_rows:
        coerced = svc.coerce_dataset_response_row(row)
        if coerced:
            valid_rows.append(coerced)
        else:
            invalid.append(
                {
                    "datasetId": row.get("id"),
                    "sourceJobId": row.get("sourceJobId"),
                    "taskType": row.get("taskType"),
                    "format": row.get("format"),
                    "error": "DatasetResponse coercion failed",
                }
            )

    by_task = Counter(str(r.get("taskType") or "unknown") for r in valid_rows)
    by_format = Counter(str(r.get("format") or "unknown") for r in valid_rows)
    missing_display = sum(1 for r in raw_rows if not (r.get("displayName") or r.get("name")))
    missing_gen_mode = sum(
        1 for r in raw_rows if not r.get("generationMode") and str(r.get("taskType") or "") == "block_stacking"
    )
    missing_file = sum(1 for r in raw_rows if not r.get("datasetFile") and r.get("format") == "hdf5")
    trainable = sum(1 for r in valid_rows if r.get("trainable"))
    replay = sum(1 for r in valid_rows if r.get("replayAvailable"))

    report = {
        "totalRecords": len(raw_rows),
        "validRecords": len(valid_rows),
        "invalidRecords": len(invalid),
        "invalidReasons": invalid,
        "byTaskType": dict(by_task),
        "byFormat": dict(by_format),
        "missingDisplayNameCount": missing_display,
        "missingGenerationModeCount": missing_gen_mode,
        "missingDatasetFileCount": missing_file,
        "trainableCount": trainable,
        "replayAvailableCount": replay,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(valid_rows) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
